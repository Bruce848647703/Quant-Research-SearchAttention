"""申万行业分类映射(巨潮源): code -> 申万一级行业名.

用途: 因子行业中性化的截面回归行业哑变量. 分类为当前静态快照,
不随历史变化(与成分股快照的局限一致, 见 README).
"""
import datetime as dt
import time
from pathlib import Path

import pandas as pd

from tqdm import tqdm

_SW_STANDARD = "申银万国行业分类标准"


def _build_l1_map(ak) -> dict:
    """从申万分类目录构造 行业名(任意层级) -> 申万一级行业名 的映射."""
    cat = ak.stock_industry_category_cninfo(symbol="申银万国行业分类标准")
    cat = cat[cat["类目编码"].astype(str).str.match(r"^S\d")]
    l1 = cat[cat["类目编码"].astype(str).str.len() == 3][["类目编码", "类目名称"]].drop_duplicates()
    l1_name = dict(zip(l1["类目编码"], l1["类目名称"]))
    name2l1 = {}
    for _, r in cat.iterrows():
        c = str(r["类目编码"])
        if len(c) >= 3 and c[:3] in l1_name:
            name2l1[str(r["类目名称"])] = l1_name[c[:3]]
    return name2l1


def _to_l1(ind, name2l1: dict):
    """巨潮返回的行业名可能带层级后缀(如 证券Ⅲ), 去后缀后查一级行业."""
    if not isinstance(ind, str):
        return ind
    key = ind.rstrip("ⅠⅡⅢⅣ").strip()
    return name2l1.get(key, name2l1.get(ind, ind))


def _fetch_one(ak, code: str, end_date: str):
    """查单只股票的行业归属变更史, 返回申万一级行业名(无申万记录时退巨潮大类)."""
    for attempt in range(2):
        try:
            df = ak.stock_industry_change_cninfo(symbol=code, start_date="20050101", end_date=end_date)
            if df is None or df.empty:
                return None
            sw = df[df["分类标准"] == _SW_STANDARD]
            if len(sw):
                return sw.iloc[-1]["行业大类"]
            cn = df[df["分类标准"].str.contains("巨潮", na=False)]
            if len(cn):  # 申万缺失时以巨潮大类兜底(中性化只需分组依据)
                return cn.iloc[-1]["行业大类"]
            return None
        except Exception:
            time.sleep(1.0)
    return None


def fetch_industry_map(codes, cache_dir, force: bool = False, sleep: float = 0.05) -> pd.DataFrame:
    """逐股抓取行业分类并缓存为 industry_map.csv(code, industry). 支持增量续抓."""
    fp = Path(cache_dir) / "industry_map.csv"
    cached = pd.read_csv(fp, dtype={"code": str}) if fp.exists() else pd.DataFrame(columns=["code", "industry"])
    recs = {} if force else dict(zip(cached["code"], cached["industry"]))
    need = [c for c in map(str, codes) if c not in recs]

    import akshare as ak
    end_date = dt.datetime.now().strftime("%Y%m%d")
    name2l1 = _build_l1_map(ak)
    if not need:  # 缓存全覆盖: 仅重新做一级行业映射(兼容旧缓存存了细级名)
        df = pd.DataFrame({"code": list(recs), "industry": [_to_l1(v, name2l1) for v in recs.values()]})
        df.to_csv(fp, index=False)
        return df
    fp.parent.mkdir(parents=True, exist_ok=True)
    for i, code in enumerate(tqdm(need, desc="industry_map")):
        ind = _fetch_one(ak, code, end_date)
        if ind:
            recs[code] = ind
        if sleep > 0:
            time.sleep(sleep)
        if i % 30 == 29:  # 增量落盘, 防中断丢失
            pd.DataFrame({"code": list(recs), "industry": list(recs.values())}).to_csv(fp, index=False)

    df = pd.DataFrame({"code": list(recs), "industry": [_to_l1(v, name2l1) for v in recs.values()]})
    df.to_csv(fp, index=False)
    missing = len(codes) - len(df)
    print(f"[industry] 覆盖 {len(df)}/{len(codes)} 只, 缺 {missing} 只(按无行业处理)")
    return df
