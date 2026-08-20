import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from ..utils import code_to_symbol, ensure_dir
from .constituents import fetch_constituents

TX_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
MAX_JUMP = 0.26  # 涨跌停20%+缓冲, 超过视为脏数据(腾讯源偶发返回损坏K线)


def _bad_mask(df: pd.DataFrame) -> pd.Series:
    bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1) | (df["high"] < df["low"])
    ret = df["close"].pct_change(fill_method=None).abs()
    return bad | (ret > MAX_JUMP)


def _fetch_window(symbol: str, start: str, end: str, adjust: str):
    all_rows = []
    cur_end = end
    for _ in range(12):
        rows = _request_kline(symbol, start, cur_end, count=800, adjust=adjust)
        if not rows:
            break
        all_rows = rows + all_rows
        first_date = rows[0][0]
        if first_date <= start:
            break
        cur_end = (pd.Timestamp(first_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if not all_rows:
        return pd.DataFrame()
    return _rows_to_df(all_rows)


def fetch_kline(symbol: str, start: str, end: str, adjust: str = "qfq", max_retries: int = 3) -> pd.DataFrame:
    """按 800 根/页向前翻页拉取完整区间; 发现脏数据(负价格/单日波动超限)则整体重抓"""
    df = _fetch_window(symbol, start, end, adjust)
    if df.empty:
        return df
    bad = _bad_mask(df)
    attempt = 0
    while bad.sum() > 0 and attempt < max_retries:
        attempt += 1
        time.sleep(0.8 * attempt)
        df_new = _fetch_window(symbol, start, end, adjust)
        if df_new.empty:
            break
        df, bad = df_new, _bad_mask(df_new)
    if bad.sum() > 0:
        tqdm.write(f"[prices] {symbol} 重试后仍有 {int(bad.sum())} 根脏K线, 已置为NaN")
        df.loc[bad, ["open", "high", "low", "close"]] = np.nan
    return df


def validate_cache(cache_dir, threshold: float = 0.005):
    """扫描缓存, 返回脏K线比例超阈值的代码(需重抓)"""
    cache_dir = Path(cache_dir) / "prices"
    bad_codes = []
    for fp in sorted(cache_dir.glob("*.parquet")):
        df = pd.read_parquet(fp)
        bad = _bad_mask(df)
        if bad.mean() > threshold:
            bad_codes.append(fp.stem.zfill(6))
    return bad_codes


def _request_kline(symbol: str, start: str, end: str, count: int = 800, adjust: str = "qfq"):
    adj = adjust or ""
    params = {"param": f"{symbol},day,{start},{end},{count},{adj}"}
    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(TX_URL, params=params, timeout=20, headers=HEADERS)
            r.raise_for_status()
            js = r.json()["data"][symbol]
            rows = js.get("qfqday") or js.get("hfqday") or js.get("day") or []
            return rows
        except Exception as e:
            last_err = e
            time.sleep(1.0 + attempt)
    raise ConnectionError(f"{symbol} K线请求失败: {last_err}")


def _rows_to_df(rows):
    recs = []
    for row in rows:
        date, o, c, h, l, v = row[0], *row[1:6]
        amount = float(v) * float(c)
        if len(row) >= 7 and isinstance(row[6], (int, float, str)):
            try:
                amount = float(row[6])
            except ValueError:
                pass
        recs.append((date, float(o), float(c), float(h), float(l), float(v), amount))
    df = pd.DataFrame(recs, columns=["date", "open", "close", "high", "low", "volume", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset="date", keep="first").sort_values("date")
    return df.set_index("date")


def fetch_all_prices(codes, start: str, end: str, cache_dir, force: bool = False, sleep: float = 0.12):
    cache_dir = ensure_dir(Path(cache_dir) / "prices")
    failed = []
    for code in tqdm(codes, desc="行情"):
        fp = cache_dir / f"{code}.parquet"
        if fp.exists() and not force:
            continue
        try:
            df = fetch_kline(code_to_symbol(code), start, end, adjust="hfq")
            if df.empty:
                failed.append(code)
                continue
            df.to_parquet(fp)
            time.sleep(sleep)
        except Exception as e:
            failed.append(code)
            tqdm.write(f"[prices] {code} 失败: {e}")
    print(f"[prices] 完成. 缓存 {len(list(cache_dir.glob('*.parquet')))} 只, 失败 {len(failed)} 只")
    return failed


def fetch_benchmark_index(cache_dir, symbol: str = "sh000300", start: str = "2020-01-01", end: str = "", force: bool = False) -> pd.DataFrame:
    cache_dir = ensure_dir(Path(cache_dir) / "index")
    fp = cache_dir / f"{symbol}.parquet"
    if fp.exists() and not force:
        return pd.read_parquet(fp)
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    df = fetch_kline(symbol, start, end, adjust=None)
    df.to_parquet(fp)
    print(f"[benchmark] {symbol} 指数行情 {len(df)} 根K线")
    return df


def load_panels(cache_dir, codes=None, fields=("open", "close", "high", "low", "volume", "amount")):
    """读取缓存的个股K线, 返回 {field: DataFrame(date x code)}"""
    cache_dir = Path(cache_dir) / "prices"
    files = sorted(cache_dir.glob("*.parquet"))
    if codes is not None:
        codes = set(str(c).zfill(6) for c in codes)
        files = [f for f in files if f.stem.zfill(6) in codes]
    data = {}
    for fp in tqdm(files, desc="加载缓存"):
        df = pd.read_parquet(fp)
        data[fp.stem.zfill(6)] = df
    panels = {}
    for field in fields:
        panels[field] = pd.DataFrame({code: d[field] for code, d in data.items()}).sort_index()
    bad = ((panels["close"] <= 0) | (panels["open"] <= 0)
           | (panels["high"] < panels["low"])
           | (panels["close"].pct_change(fill_method=None).abs() > MAX_JUMP))
    n_bad = int(bad.values.sum())
    if n_bad:
        print(f"[panels] 清洗 {n_bad} 个异常价格点(负价格/单日波动>{MAX_JUMP:.0%}), 置为NaN")
        for field in ("open", "high", "low", "close"):
            panels[field] = panels[field].mask(bad)
    print(f"[panels] {len(files)} 只股票, {len(panels['close'])} 个交易日 "
          f"({panels['close'].index.min().date()} ~ {panels['close'].index.max().date()})")
    return panels


def load_benchmark_index(cache_dir, symbol: str = "sh000300") -> pd.DataFrame:
    fp = Path(cache_dir) / "index" / f"{symbol}.parquet"
    return pd.read_parquet(fp)
