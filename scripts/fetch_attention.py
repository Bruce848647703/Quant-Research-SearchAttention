"""抓取真实关注度代理数据: 龙虎榜明细 + 股东户数历史, 缓存至 data/cache/attention."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import load_config, resolve_dir
from src.data import fetch_gdhs, fetch_lhb
from src.data.constituents import fetch_constituents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="忽略缓存重新抓取")
    ap.add_argument("--skip-gdhs", action="store_true", help="仅抓龙虎榜")
    args = ap.parse_args()

    cfg = load_config()
    cache_dir = resolve_dir(cfg, "cache_dir")
    start = pd.to_datetime(cfg["data"]["start_date"]).strftime("%Y-%m-%d")
    end = (cfg["data"]["end_date"] or pd.Timestamp.today().strftime("%Y-%m-%d"))

    fetch_lhb(cache_dir, start, end, force=args.force)

    if not args.skip_gdhs:
        cons = fetch_constituents(cache_dir, cfg["universe"]["index_code"])
        codes = cons["code"].drop_duplicates().tolist()
        fetch_gdhs(codes, cache_dir, force=args.force)


if __name__ == "__main__":
    main()
