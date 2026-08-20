import datetime as dt
from pathlib import Path

import pandas as pd


def fetch_constituents(cache_dir, index_code: str = "000300", force: bool = False) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{index_code}_constituents.csv"
    if cache_file.exists() and not force:
        df = pd.read_csv(cache_file, dtype={"code": str})
        print(f"[constituents] 使用缓存 {cache_file} ({len(df)}只)")
        return df

    import akshare as ak

    cons = ak.index_stock_cons(symbol=index_code)
    df = pd.DataFrame(
        {
            "code": cons["品种代码"].astype(str).str.zfill(6),
            "name": cons["品种名称"],
            "include_date": cons["纳入日期"],
        }
    )
    df.to_csv(cache_file, index=False)
    print(f"[constituents] {index_code} 成分股 {len(df)} 只, 抓取于 {dt.date.today()}, 缓存至 {cache_file}")
    print("[constituents] 注意: 使用当前成分股回测存在幸存者偏差, 结果仅供研究参考")
    return df
