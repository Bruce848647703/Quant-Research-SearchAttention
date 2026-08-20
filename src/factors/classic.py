"""价量类因子, 输入为宽表 (date x code), 全部仅使用 t 日及以前信息."""
import numpy as np
import pandas as pd


def mom_12_1(close: pd.DataFrame) -> pd.DataFrame:
    """12-1 月动量: t-252 至 t-21 的区间收益"""
    return close.shift(21) / close.shift(252) - 1.0


def rev_5(close: pd.DataFrame) -> pd.DataFrame:
    """5日收益 (短期反转, 方向由 config.direction 控制)"""
    return close / close.shift(5) - 1.0


def ret_20(close: pd.DataFrame) -> pd.DataFrame:
    return close / close.shift(20) - 1.0


def low_vol_20(close: pd.DataFrame) -> pd.DataFrame:
    """20日已实现波动率"""
    return close.pct_change(fill_method=None).rolling(20).std()


def volume_ratio(volume: pd.DataFrame) -> pd.DataFrame:
    """5日均量 / 60日均量"""
    return volume.rolling(5).mean() / volume.rolling(60).mean()


def amihud(close: pd.DataFrame, amount: pd.DataFrame) -> pd.DataFrame:
    """Amihud (2002) 非流动性: 20日均 |ret| / 20日均成交额, 放大便于阅读"""
    return close.pct_change(fill_method=None).abs().rolling(20).mean() / amount.rolling(20).mean() * 1e9
