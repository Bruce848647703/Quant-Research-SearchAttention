"""真实关注度代理数据: 龙虎榜事件 + 股东户数变化.

真实搜索指数无公开历史API, 本模块提供两类可回溯的真实"投资者关注"代理:

1. 龙虎榜 (东财, akshare stock_lhb_detail_em): 个股因涨跌幅/换手率异常等触发上榜,
   上榜当日通常伴随搜索量与讨论量激增, 是真实的强关注事件, 历史覆盖完整回测区间.
2. 股东户数 (东财, akshare stock_zh_a_gdhs_detail_em): 季频披露, 股东户数增加代表
   筹码向散户分散/关注度上升, 是A股文献中验证充分的散户参与度代理.

时点口径 (无未来函数):
- 龙虎榜: 上榜日晚间才披露, 事件自上榜日次日起可用 (shift 1);
- 股东户数: 以"公告日期"为市场可知日, 自公告次日起可用 (shift 1).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils import ensure_dir


def fetch_lhb(cache_dir, start: str, end: str, force: bool = False) -> pd.DataFrame:
    """抓取区间内龙虎榜明细并缓存整表(单接口覆盖全区间)."""
    att_dir = ensure_dir(Path(cache_dir) / "attention")
    fp = att_dir / "lhb_detail.parquet"
    if fp.exists() and not force:
        df = pd.read_parquet(fp)
        print(f"[attention] 龙虎榜使用缓存 {fp} ({len(df)}条)")
        return df

    import akshare as ak

    raw = ak.stock_lhb_detail_em(start_date=start.replace("-", ""), end_date=end.replace("-", ""))
    df = pd.DataFrame({
        "code": raw["代码"].astype(str).str.zfill(6),
        "date": pd.to_datetime(raw["上榜日"]),
        "net_amt": pd.to_numeric(raw["龙虎榜净买额"], errors="coerce"),
        "turnover": pd.to_numeric(raw["换手率"], errors="coerce"),
        "reason": raw["上榜原因"].astype(str),
    })
    df = df.sort_values(["date", "code"]).reset_index(drop=True)
    df.to_parquet(fp)
    print(f"[attention] 龙虎榜 {start}~{end}: {len(df)}条, "
          f"{df['code'].nunique()}只股票, 缓存至 {fp}")
    return df


def fetch_gdhs(codes, cache_dir, force: bool = False, sleep: float = 0.15):
    """逐股抓取股东户数历史并缓存(每股一个文件, 支持断点续抓)."""
    import akshare as ak

    gdhs_dir = ensure_dir(Path(cache_dir) / "attention" / "gdhs")
    failed = []
    for code in tqdm(codes, desc="股东户数"):
        fp = gdhs_dir / f"{code}.parquet"
        if fp.exists() and not force:
            continue
        try:
            raw = ak.stock_zh_a_gdhs_detail_em(symbol=code)
            df = pd.DataFrame({
                "cutoff_date": pd.to_datetime(raw["股东户数统计截止日"]),
                "announce_date": pd.to_datetime(raw["股东户数公告日期"]),
                "holders": pd.to_numeric(raw["股东户数-本次"], errors="coerce"),
                "chg_pct": pd.to_numeric(raw["股东户数-增减比例"], errors="coerce"),
            }).dropna(subset=["announce_date"]).sort_values("announce_date")
            df.to_parquet(fp)
            time.sleep(sleep)
        except Exception as e:
            failed.append(code)
            tqdm.write(f"[attention] {code} 股东户数抓取失败: {e}")
    n = len(list(gdhs_dir.glob("*.parquet")))
    print(f"[attention] 股东户数完成. 缓存 {n} 只, 失败 {len(failed)} 只")
    return failed


def load_attention_panels(cache_dir, index: pd.DatetimeIndex, codes=None, gdhs_span: int = 4) -> dict:
    """读取缓存并构造日频面板 (date x code), 均已按可知日 shift(1) 防前瞻.

    返回: {"lhb_events": 上榜事件计数, "lhb_net": 净买额(元),
          "gdhs_chg": 最近 gdhs_span 期股东户数累计变化%}
    缺失的数据源不会出现在返回 dict 中.
    """
    att_dir = Path(cache_dir) / "attention"
    if codes is not None:
        codes = set(str(c).zfill(6) for c in codes)
    out = {}

    fp = att_dir / "lhb_detail.parquet"
    if fp.exists():
        df = pd.read_parquet(fp)
        if codes is not None:
            df = df[df["code"].isin(codes)]
        df = df[df["date"].isin(index)]
        events = df.groupby(["date", "code"]).size().unstack(fill_value=0)
        net = df.groupby(["date", "code"])["net_amt"].sum().unstack(fill_value=0.0)
        events = events.reindex(index=index).fillna(0.0).shift(1).fillna(0.0)
        net = net.reindex(index=index).fillna(0.0).shift(1).fillna(0.0)
        if codes is not None:
            events, net = events.reindex(columns=sorted(codes)), net.reindex(columns=sorted(codes))
        out["lhb_events"], out["lhb_net"] = events.fillna(0.0), net.fillna(0.0)

    gdhs_dir = att_dir / "gdhs"
    if gdhs_dir.exists():
        series = {}
        for f in sorted(gdhs_dir.glob("*.parquet")):
            code = f.stem.zfill(6)
            if codes is not None and code not in codes:
                continue
            df = pd.read_parquet(f)
            if df.empty:
                continue
            # 同一公告日多条保留最后一条; 以公告日为可知时点
            df = df.drop_duplicates("announce_date", keep="last").set_index("announce_date").sort_index()
            h = df["holders"]
            # 长窗口累计变化(默认近4期≈一年): 诊断上比单期变化更有预测力
            s = (h / h.shift(gdhs_span) - 1) * 100 if gdhs_span > 1 else df["chg_pct"]
            series[code] = s[~s.index.duplicated(keep="last")]
        if series:
            panel = pd.DataFrame(series).reindex(index).ffill().shift(1)
            out["gdhs_chg"] = panel

    return out
