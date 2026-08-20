from typing import Dict

import numpy as np
import pandas as pd

from ..backtest.metrics import ic_series
from .attention import attn_price_div, search_heat, search_mom
from .attention_proxy import gdhs_change, lhb_heat, lhb_net_buy
from .classic import amihud, low_vol_20, mom_12_1, rev_5, volume_ratio


def build_factors(panels: Dict[str, pd.DataFrame], si: pd.DataFrame,
                  attention: Dict[str, pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    close, volume, amount = panels["close"], panels["volume"], panels["amount"]
    factors = {
        "search_heat": search_heat(si),
        "search_mom": search_mom(si),
        "attn_price_div": attn_price_div(si, close),
        "mom_12_1": mom_12_1(close),
        "rev_5": rev_5(close),
        "low_vol_20": low_vol_20(close),
        "volume_ratio": volume_ratio(volume),
        "amihud": amihud(close, amount),
    }
    # 真实关注度代理因子: 仅在对应数据已缓存时启用
    if attention:
        if "lhb_events" in attention:
            factors["lhb_heat"] = lhb_heat(attention["lhb_events"])
        if "lhb_net" in attention:
            factors["lhb_net_buy"] = lhb_net_buy(attention["lhb_net"])
        if "gdhs_chg" in attention:
            factors["gdhs_change"] = gdhs_change(attention["gdhs_chg"])
    return factors


def winsorize_mad(df: pd.DataFrame, n: float = 5.0) -> pd.DataFrame:
    med = df.median(axis=1)
    mad = (df.sub(med, axis=0)).abs().median(axis=1) * 1.4826
    lo = med - n * mad
    hi = med + n * mad
    return df.clip(lower=lo, upper=hi, axis=0)


def zscore_cross(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1)
    return df.sub(mu, axis=0).div(sd, axis=0)


def neutralize_cross(df: pd.DataFrame, industry: pd.Series, size: pd.DataFrame,
                     min_obs: int = 30) -> pd.DataFrame:
    """逐日截面回归残差: 对行业哑变量 + 规模代理做 OLS, 返回残差.

    industry: code -> 行业名(静态分类, 可含 NaN);
    size: 与 df 同形状的规模代理面板(如 ln(成交额60日均值)).
    单日有效样本不足 min_obs 的行保持原值不处理.
    """
    ind = industry.reindex(df.columns)
    dummies = pd.get_dummies(ind, dtype=float)
    base = pd.concat([pd.Series(1.0, index=df.columns, name="const"), dummies], axis=1)
    rows = []
    for t, row in df.iterrows():
        x = base.copy()
        x["size"] = size.loc[t] if t in size.index else np.nan
        m = row.notna() & x.notna().all(axis=1)
        if m.sum() < min_obs:
            rows.append(row)
            continue
        beta, *_ = np.linalg.lstsq(x[m].to_numpy(), row[m].to_numpy(), rcond=None)
        resid = row - x @ beta
        rows.append(resid.where(m))
    return pd.DataFrame(rows, index=df.index, columns=df.columns)


def _prep_zscore(factors: Dict[str, pd.DataFrame], entries: dict,
                 neutral: tuple = None) -> Dict[str, pd.DataFrame]:
    """方向对齐 -> (可选)行业/规模中性化 -> MAD去极值 -> 截面z-score."""
    zed = {}
    for name, e in entries.items():
        raw = factors[name] * e.get("direction", 1)
        if neutral is not None:
            raw = neutralize_cross(raw, neutral[0], neutral[1])
        zed[name] = zscore_cross(winsorize_mad(raw))
    return zed


def compute_composite(factors: Dict[str, pd.DataFrame], entries: dict,
                      min_valid: float = 0.5, neutral: tuple = None) -> pd.DataFrame:
    """按 config 中的方向与权重做截面标准化合成.

    - 每个因子先 MAD 去极值 + 截面 z-score, 再乘以方向与权重累加;
    - 缺失因子不再按中性(0)填充, 权重按该股实际可用因子重新归一,
      避免部分因子缺失的股票被系统性拉向中间排名;
    - 可用因子比例低于 min_valid 的股票整行置 NaN, 不参与选股.
    """
    entries = {k: v for k, v in entries.items() if k in factors}
    total_w = sum(abs(e["weight"]) for e in entries.values())
    if total_w <= 0 or not entries:
        raise ValueError("因子配置为空或权重全为0")
    zed = _prep_zscore(factors, entries, neutral)
    weighted_sum, avail_w, n_avail = None, None, None
    for name, e in entries.items():
        z = zed[name]
        w = abs(e["weight"]) / total_w
        mask = z.notna().astype(float)
        weighted_sum = z.mul(w) if weighted_sum is None else weighted_sum.add(z.mul(w), fill_value=0)
        avail_w = mask.mul(w) if avail_w is None else avail_w.add(mask.mul(w), fill_value=0)
        n_avail = mask if n_avail is None else n_avail.add(mask, fill_value=0)
        index = z.index
    comp = weighted_sum.div(avail_w.where(avail_w > 0))
    if min_valid > 0:
        comp = comp.mask(n_avail / len(entries) < min_valid)
    return comp


def rebalance_dates(index_dates: pd.DatetimeIndex, start: str, end: str, rule: str = "month_end") -> pd.DatetimeIndex:
    days = pd.Series(index_dates)
    days = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
    if rule != "month_end":
        raise ValueError(f"不支持的换仓规则: {rule}")
    return pd.DatetimeIndex(days.groupby(days.dt.to_period("M")).max().values)


def compute_composite_ic_weighted(factors: Dict[str, pd.DataFrame], entries: dict,
                                open_: pd.DataFrame, signal_dates: pd.DatetimeIndex,
                                window: int = 12, min_periods: int = 6,
                                min_valid: float = 0.5, weight_mode: str = "ic",
                                neutral: tuple = None) -> pd.DataFrame:
    """滚动IC自适应加权合成.

    - weight_mode="ic": 权重 = 过去 window 期样本外 Rank IC 均值;
      weight_mode="ic_ir": 权重 = IC均值/IC标准差(IC_IR, 对波动大的因子降权, 更稳健);
      两者均截断在 0, 即弱/反向信号因子自动降权至零, 不会主动做空;
    - 仅在信号日输出合成得分; 信号日 s 的权重只由在 s 日收盘前已完全实现的
      历史IC决定(前瞻收益按信号日对齐, 并校验其实现日不晚于 s), 无未来函数;
    - 历史IC不足 min_periods 期的信号日退化为等权合成.
    """
    from ..backtest.engine import _sig_to_exec

    entries = {k: v for k, v in entries.items() if k in factors}
    if not entries:
        raise ValueError("因子配置为空")
    sig2exec = _sig_to_exec(open_.index, signal_dates)
    signal_dates = pd.DatetimeIndex(sorted(sig2exec.keys()))  # 过滤无执行日(区间外)的信号
    exec_to_sig = {v: k for k, v in sig2exec.items()}
    exec_dates = sorted(set(sig2exec.values()))
    # 列=信号日: 信号t的前瞻收益 = open(exec_t) -> open(exec_{t+1})
    fwd_ret = pd.DataFrame({exec_to_sig[o0]: open_.loc[o1] / open_.loc[o0] - 1
                            for o0, o1 in zip(exec_dates[:-1], exec_dates[1:])})

    n = len(entries)
    zed = _prep_zscore(factors, entries, neutral)
    icw = pd.DataFrame({name: ic_series(z, fwd_ret).sort_index()
                        for name, z in zed.items()})
    roll_mean = icw.rolling(window, min_periods=min_periods).mean()
    if weight_mode == "ic_ir":
        roll_std = icw.rolling(window, min_periods=min_periods).std()
        roll = roll_mean.div(roll_std.where(roll_std > 1e-8))
    elif weight_mode == "ic":
        roll = roll_mean
    else:
        raise ValueError(f"不支持的加权模式: {weight_mode}")
    # 信号t的IC在 exec_{t+1} 开盘后才实现, 早于该日不可用(防前瞻)
    sig_sorted = pd.DatetimeIndex(sorted(sig2exec.keys()))
    ic_known_at = pd.Series(sig2exec, index=sig_sorted).shift(-1)  # 对齐信号日t

    columns = zed[next(iter(zed))].columns
    comp = pd.DataFrame(np.nan, index=signal_dates, columns=columns)
    eq_w = pd.Series(1.0 / n, index=list(entries))
    for sig_d in comp.index:
        known = ic_known_at[(ic_known_at <= sig_d) & ic_known_at.index.isin(roll.index)]
        if known.empty:
            w = eq_w
        else:
            w = roll.loc[known.index[-1]].clip(lower=0.0)
            if w.isna().all() or w.sum() <= 0:
                w = eq_w
        row_sum, row_w, n_avail = None, None, None
        for name in entries:
            z_row = zed[name].loc[sig_d]
            m = z_row.notna().astype(float)
            row_sum = z_row.mul(w[name]) if row_sum is None else row_sum.add(z_row.mul(w[name]), fill_value=0)
            row_w = m.mul(w[name]) if row_w is None else row_w.add(m.mul(w[name]), fill_value=0)
            n_avail = m if n_avail is None else n_avail.add(m, fill_value=0)
        comp.loc[sig_d] = row_sum.div(row_w.where(row_w > 0))
        if min_valid > 0:
            comp.loc[sig_d] = comp.loc[sig_d].mask(n_avail / n < min_valid)
    return comp
