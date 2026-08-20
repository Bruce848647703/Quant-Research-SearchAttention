"""抓取股票池的申万行业分类映射(用于因子行业中性化)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import load_config, resolve_dir
from src.data.industry import fetch_industry_map


def main():
    cfg = load_config()
    cache_dir = resolve_dir(cfg, "cache_dir")
    cons = pd.read_csv(Path(cache_dir) / f"{cfg['universe']['index_code']}_constituents.csv",
                       dtype={"code": str})
    df = fetch_industry_map(cons["code"].tolist(), cache_dir)
    print(df["industry"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
