"""搜索关注度因子, 基于搜索指数面板 SI (date x code)."""
import numpy as np
import pandas as pd

from .classic import ret_20


def search_heat(si: pd.DataFrame) -> pd.DataFrame:
    """热度水平: log(SI) - log(SI)的60日均值"""
    log_si = np.log(si.replace(0, np.nan))
    return log_si - log_si.rolling(60, min_periods=30).mean()


def search_mom(si: pd.DataFrame) -> pd.DataFrame:
    """热度动量: log(SI) 20日变化"""
    return np.log(si.replace(0, np.nan)).diff(20)


def attn_price_div(si: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """关注度-价格背离: 截面排序(热度动量) - 截面排序(20日涨幅), >0 表示热度领先价格"""
    mom = search_mom(si)
    r20 = ret_20(close)
    mr = mom.rank(axis=1, pct=True)
    rr = r20.rank(axis=1, pct=True)
    return mr - rr
