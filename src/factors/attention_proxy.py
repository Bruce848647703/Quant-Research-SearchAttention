"""真实关注度代理因子 (龙虎榜事件 + 股东户数变化).

输入面板已在数据层按可知日 shift(1), 本模块仅做时序变换, 不引入未来信息.
"""
import pandas as pd


def lhb_heat(events: pd.DataFrame, span: int = 20) -> pd.DataFrame:
    """龙虎榜关注热度: 上榜事件的指数衰减强度 (span 控制衰减半衰尺度)."""
    return events.ewm(span=span, min_periods=1).mean()


def lhb_net_buy(net_amt: pd.DataFrame, span: int = 20) -> pd.DataFrame:
    """龙虎榜资金净买入强度: 净买额(元)的指数衰减均值."""
    return net_amt.ewm(span=span, min_periods=1).mean()


def gdhs_change(panel: pd.DataFrame) -> pd.DataFrame:
    """股东户数增减比例(%): 数据层已按公告日 ffill, 增加=筹码向散户分散."""
    return panel
