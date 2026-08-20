"""月度换仓多头组合回测: T日收盘生成信号, T+1日开盘成交, 全卖全买, 双边计成本."""
from typing import Dict, List

import numpy as np
import pandas as pd


def _sig_to_exec(trading_days: pd.DatetimeIndex, signal_dates: pd.DatetimeIndex) -> Dict[pd.Timestamp, pd.Timestamp]:
    idx = np.searchsorted(trading_days.values, signal_dates.values, side="right")
    out = {}
    for i, t in zip(idx, signal_dates):
        if i < len(trading_days):
            out[pd.Timestamp(t)] = trading_days[i]
    return out


def run_backtest(close: pd.DataFrame, open_: pd.DataFrame, composite: pd.DataFrame,
                 signal_dates: pd.DatetimeIndex, top_n: int, cost_bps_one_way: float,
                 min_history_days: int = 120, start_nav: float = 1.0,
                 buffer_ratio: float = 0.0, limit_check: bool = True,
                 universe_mask: pd.DataFrame = None):
    """
    buffer_ratio: 换手缓冲带 -- 原持仓得分排名仍在 top_n*(1+buffer_ratio) 内的不卖出,
        仅用卖出回笼资金等额买入新入选股(降低换手与成本)。
    limit_check: 以 T+1 开盘涨跌幅 vs 涨跌停阈值(主板±10%/创科±20%)近似判定一字板:
        开盘即涨停视为不可买入, 开盘即跌停视为不可卖出(继续持有到下期)。
    universe_mask: 可选的时点股票池掩码(如成分股 include_date 准点位还原), 与历史长度条件取交集。
    """
    trading_days = close.index
    sig2exec = _sig_to_exec(trading_days, signal_dates)
    exec_dates = set(sig2exec.values())
    exec_to_sig = {v: k for k, v in sig2exec.items()}

    close_fill = close.ffill()
    open_fill = open_.ffill()
    eligible = close.notna().cumsum() >= min_history_days
    if universe_mask is not None:
        eligible = eligible & universe_mask.reindex(index=trading_days, columns=close.columns).fillna(False)
    score = composite.mask(~eligible)

    # 涨跌停判定: 开盘价相对前收的涨跌幅(阈值留 0.2% 容差)
    lim = pd.Series(0.10, index=close.columns)
    lim[[c for c in close.columns if c.startswith(("30", "68"))]] = 0.20
    open_ret = open_ / close.shift(1) - 1

    holdings: List[dict] = []
    shares: Dict[str, float] = {}
    cash = float(start_nav)
    nav = []
    cost_rate = cost_bps_one_way * 1e-4

    for d in trading_days:
        if d in exec_dates:
            po = open_fill.loc[d]
            sig_d = exec_to_sig[d]
            sc = score.loc[sig_d].dropna()
            if len(sc):
                ordered = sc.sort_values(ascending=False).index.tolist()
                rank = pd.Series(np.arange(1, len(ordered) + 1), index=ordered)
                prev = list(shares.keys())
                if buffer_ratio > 0:
                    keep = [c for c in prev if c in rank.index and rank[c] <= top_n * (1 + buffer_ratio)]
                    keep.sort(key=lambda c: rank[c])
                    keep = keep[:top_n]  # 保留超员时降掉得分最低者
                else:
                    keep = []
                target = list(keep)
                for c in ordered:
                    if len(target) >= top_n:
                        break
                    if c not in target:
                        target.append(c)

                buy_ok = (open_ret.loc[d] < lim - 0.002) if limit_check else pd.Series(True, index=close.columns)
                sell_ok = (open_ret.loc[d] > -(lim - 0.002)) if limit_check else pd.Series(True, index=close.columns)

                sells = [c for c in prev if c not in target]
                stuck = [c for c in sells if not bool(sell_ok.get(c, True))]  # 跌停卖不出, 继续持有
                sells = [c for c in sells if c not in stuck]
                sell_val = sum(s * po[c] for c, s in ((c, shares[c]) for c in sells)
                               if c in po.index and np.isfinite(po[c]))
                sell_cost = sell_val * cost_rate
                cash += sell_val - sell_cost

                buys = [c for c in target if c not in shares]
                px = open_.loc[d].reindex(buys).dropna()
                px = px[px.index.map(lambda c: bool(buy_ok.get(c, True)))]  # 涨停买不进
                per_cash = cash / max(len(px), 1) if len(px) else 0.0
                buy_cost = per_cash * len(px) * cost_rate
                shares = {c: s for c, s in shares.items() if c not in sells}
                for c, p in px.items():
                    shares[c] = per_cash * (1 - cost_rate) / p
                cash -= per_cash * len(px)
                cost_paid = sell_cost + buy_cost
            else:
                cost_paid = 0.0
            holdings.append({
                "signal_date": sig_d.strftime("%Y-%m-%d"), "exec_date": d.strftime("%Y-%m-%d"),
                "n_hold": len(shares), "cost_paid": round(cost_paid, 2),
                "tickers": " ".join(sorted(shares.keys())),
            })
        val = cash + sum(s * close_fill.loc[d][c] for c, s in shares.items()
                         if c in close_fill.columns and np.isfinite(close_fill.loc[d][c]))
        nav.append(val)

    nav = pd.Series(nav, index=trading_days, name="strategy")
    hist = pd.DataFrame(holdings)
    name_turnover = None
    if len(hist) > 1:
        sets = [set(h["tickers"].split()) for h in holdings]
        name_turnover = pd.Series(
            [1 - len(a & b) / max(len(a | b), 1) for a, b in zip(sets[:-1], sets[1:])],
            index=hist["exec_date"].iloc[1:].values, name="name_turnover")
    return nav, hist, name_turnover


def quantile_returns(close: pd.DataFrame, factor: pd.DataFrame, signal_dates: pd.DatetimeIndex, q: int = 5):
    """分层诊断: 信号日收盘 -> 下一信号日收盘, 分组等权收益, 未含成本"""
    dates = [d for d in signal_dates if d in close.index]
    recs = []
    for t0, t1 in zip(dates[:-1], dates[1:]):
        ranks = factor.loc[t0].rank(pct=True)
        grp = np.minimum((ranks * q).round().clip(1, q), q).astype("Int64")
        ret = (close.loc[t1] / close.loc[t0] - 1)
        row = ret.groupby(grp).mean()
        row.name = t1
        recs.append(row)
    out = pd.DataFrame(recs).sort_index()
    out.columns = [f"Q{c}" for c in out.columns]
    return out


def benchmark_equal_weight(close: pd.DataFrame, start: str) -> pd.Series:
    ret = close.pct_change(fill_method=None).mean(axis=1)
    ret = ret[ret.index >= pd.Timestamp(start)].fillna(0.0)
    nav = (1 + ret).cumprod()
    return nav / nav.iloc[0]


def forward_open_ret(open_: pd.DataFrame, signal_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """执行口径前瞻收益: open(T+1) -> open(下一期T+1), 列=执行日"""
    trading_days = open_.index
    sig2exec = _sig_to_exec(trading_days, signal_dates)
    exec_dates = sorted(set(sig2exec.values()))
    recs = {}
    for o0, o1 in zip(exec_dates[:-1], exec_dates[1:]):
        recs[o0] = open_.loc[o1] / open_.loc[o0] - 1
    return pd.DataFrame(recs)
