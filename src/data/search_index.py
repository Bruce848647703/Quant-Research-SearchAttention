"""搜索指数数据层.

优先读取真实数据: data/search_index/{ticker}.csv (列: date, index, 可含百度指数等导出).
缺失时使用合成搜索指数 (默认), 仅用于验证框架, 不代表真实市场数据.
合成逻辑 (无未来函数, 仅用 t-1 及更早信息):
    log(SI_t) = base_i + rho * (log(SI_{t-1}) - base_i)
                + beta_r * |ret_{t-1}| + beta_p * max(ret_{t-1}, 0)
                + beta_v * clip(log(量比_{t-1}), -1, 2) + event_t + eps_t
即: 大涨大跌、放量、偶发热点事件会推高后续搜索热度 (散户关注滞后于行情, 与行为金融文献一致).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def _synth_one(close_s: pd.Series, volume_s: pd.Series, base: float, rng: np.random.Generator,
               rho=0.90, beta_r=4.0, beta_p=2.5, beta_v=0.30, sigma=0.10,
               event_prob=0.008, event_scale=1.0) -> pd.Series:
    ret = close_s.pct_change(fill_method=None).fillna(0.0).to_numpy()
    vol_ma = volume_s.rolling(60, min_periods=20).mean()
    vol_ratio = (volume_s / vol_ma).fillna(1.0).to_numpy()
    n = len(close_s)
    x = np.full(n, base)
    ev = 0.0
    for t in range(1, n):
        lag_r = ret[t - 1]
        lag_v = float(np.clip(np.log(max(vol_ratio[t - 1], 1e-6)), -1.0, 2.0))
        ev = ev * 0.85
        if rng.random() < event_prob:
            ev += rng.exponential(event_scale)
        x[t] = base + rho * (x[t - 1] - base) + beta_r * abs(lag_r) + beta_p * max(lag_r, 0.0) \
            + beta_v * lag_v + ev + rng.normal(0.0, sigma)
    return pd.Series(np.exp(x), index=close_s.index, name=close_s.name)


def generate_synthetic(close: pd.DataFrame, volume: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = {}
    for code in close.columns:
        c = close[code].dropna()
        v = volume[code].reindex(c.index)
        if len(c) < 120:
            continue
        out[code] = _synth_one(c, v, base=0.0, rng=rng)
    si = pd.DataFrame(out, index=close.index)
    lvl = np.log1p(close.mul(volume).mean()).rank(pct=True)
    base_map = 5.5 + 1.8 * lvl
    si = si * np.exp(base_map).reindex(si.columns)
    return si


def _load_real(code: str, path: Path, date_col: str, index_col: str) -> pd.Series:
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    s = df.set_index(date_col)[index_col].astype(float)
    return s[~s.index.duplicated(keep="last")].sort_index()


def load_search_index(cfg, close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    s_cfg = cfg["search_index"]
    mode = s_cfg.get("mode", "auto")
    si_dir = Path(cfg["data"]["search_index_dir"])
    if not si_dir.is_absolute():
        from ..config import PROJECT_ROOT
        si_dir = PROJECT_ROOT / si_dir

    date_col = s_cfg.get("real_file_date_col", "date")
    index_col = s_cfg.get("real_file_index_col", "index")
    n_real = n_synth = 0
    real_parts, synth_codes = {}, []
    for code in tqdm(close.columns, desc="搜索指数"):
        fp = si_dir / f"{code}.csv"
        s = None
        if fp.exists() and mode in ("auto", "real"):
            try:
                s = _load_real(code, fp, date_col, index_col)
                n_real += 1
            except Exception as e:
                print(f"[search_index] {code} 真实数据读取失败: {e}")
                s = None
        if s is None:
            if mode == "real":
                continue
            synth_codes.append(code)
        else:
            real_parts[code] = s

    si = pd.DataFrame(index=close.index)
    if real_parts:
        si = si.join(pd.DataFrame(real_parts))
    if synth_codes and mode in ("auto", "synthetic"):
        sub_close = close[synth_codes]
        synth = generate_synthetic(sub_close, volume[synth_codes], seed=s_cfg.get("synthetic_seed", 42))
        si = si.join(synth)
        n_synth = len(synth_codes)

    print(f"[search_index] 真实数据 {n_real} 只, 合成数据 {n_synth} 只"
          + (" (合成数据仅用于框架验证)" if n_synth > 0 else ""))
    si = si.reindex(columns=close.columns)
    if si.empty or not si.notna().any().any():
        raise RuntimeError(
            f"搜索指数为空: 请确认 mode={mode} 配置, 或在 {si_dir} 下放置真实指数CSV({{code}}.csv)")
    n_missing = int(si.isna().all().sum())
    if n_missing:
        print(f"[search_index] 警告: {n_missing} 只股票无任何搜索指数数据, 将被排除出选股")
    return si
