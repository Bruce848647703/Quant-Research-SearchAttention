"""抓取沪深300成分股 + 日线行情(腾讯源, 后复权hfq) + 基准指数, 缓存至 data/cache."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import PROJECT_ROOT, load_config, resolve_dir
from src.data import fetch_all_prices, fetch_benchmark_index, fetch_constituents
from src.data.prices import validate_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="忽略缓存重新抓取")
    args = ap.parse_args()

    cfg = load_config()
    cache_dir = resolve_dir(cfg, "cache_dir")
    start = pd.to_datetime(cfg["data"]["start_date"]).strftime("%Y-%m-%d")
    end = cfg["data"]["end_date"] or pd.Timestamp.today().strftime("%Y-%m-%d")

    cons = fetch_constituents(cache_dir, cfg["universe"]["index_code"], force=args.force)
    codes = cons["code"].drop_duplicates().tolist()
    failed = fetch_all_prices(codes, start, end, cache_dir, force=args.force)
    fetch_benchmark_index(cache_dir, "sh" + cfg["universe"]["index_code"], start, end, force=args.force)

    bad = validate_cache(cache_dir)
    if bad:
        print(f"[validate] {len(bad)} 只缓存含较多脏K线, 重新抓取: {bad}")
        from pathlib import Path
        for c in bad:
            (Path(cache_dir) / "prices" / f"{c}.parquet").unlink(missing_ok=True)
        failed += fetch_all_prices(bad, start, end, cache_dir, force=True)
    if failed:
        print(f"以下股票抓取失败: {set(failed)}")


if __name__ == "__main__":
    main()
