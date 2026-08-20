import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _spearman(a: pd.Series, b: pd.Series) -> float:
    return a.rank().corr(b.rank())


def performance_metrics(nav: pd.Series, rf: float = 0.0) -> dict:
    ret = nav.pct_change(fill_method=None).dropna()
    n = len(ret)
    years = n / TRADING_DAYS
    total = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    ann_vol = ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    dd = nav / nav.cummax() - 1
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    monthly = ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    return {
        "累计收益": f"{total:+.2%}",
        "年化收益": f"{ann_ret:+.2%}",
        "年化波动": f"{ann_vol:.2%}",
        "夏普比率": f"{sharpe:.2f}",
        "最大回撤": f"{max_dd:.2%}",
        "卡玛比率": f"{calmar:.2f}",
        "月度胜率": f"{(monthly > 0).mean():.2%}",
    }


def ic_series(factor: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.Series:
    common_dates = factor.index.intersection(pd.Index(fwd_ret.columns))
    ics = []
    for d in common_dates:
        f = factor.loc[d]
        r = fwd_ret[d]
        mask = f.notna() & r.notna()
        if mask.sum() < 30:
            continue
        rho = _spearman(f[mask], r[mask])
        ics.append((d, rho))
    return pd.Series(dict(ics), name="IC")


def ic_stats(ic: pd.Series) -> dict:
    if ic is None or len(ic) == 0:
        return {"IC均值": np.nan, "ICIR": np.nan, "IC胜率": np.nan, "t值": np.nan}
    mean = ic.mean()
    std = ic.std()
    t = mean / (std / np.sqrt(len(ic))) if std > 0 else np.nan
    return {"IC均值": mean, "ICIR": mean / std if std > 0 else np.nan,
            "IC胜率": (ic > 0).mean(), "t值": t}
