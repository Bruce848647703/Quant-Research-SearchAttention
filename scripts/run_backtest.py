"""端到端: 加载数据 -> 搜索指数 -> 因子 -> 月度换仓回测 -> 报告"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.backtest import (benchmark_equal_weight, forward_open_ret, ic_series, ic_stats,
                          performance_metrics, quantile_returns, run_backtest)
from src.backtest.engine import _sig_to_exec
from src.backtest.report import run_report
from src.config import PROJECT_ROOT, load_config, resolve_dir
from src.data import load_attention_panels, load_search_index
from src.data.prices import load_benchmark_index, load_panels
from src.factors import build_factors, compute_composite
from src.factors.pipeline import compute_composite_ic_weighted, rebalance_dates


def _build_universe_mask(cache_dir, index_code, close):
    """准点位还原: 仅允许信号日已纳入成分股(include_date)的股票参与选股.

    注: 成分股快照无剔除日期, 已剔除股票仍会保留, 属部分还原.
    """
    fp = Path(cache_dir) / f"{index_code}_constituents.csv"
    cons = pd.read_csv(fp, dtype={"code": str})
    if "include_date" not in cons.columns:
        return None
    inc = pd.to_datetime(cons["include_date"], errors="coerce")
    cols = [c for c in cons["code"] if c in close.columns]
    mask = pd.DataFrame(
        {r.code: (close.index >= d) if pd.notna(d) else True
         for r, d in zip(cons[cons["code"].isin(cols)].itertuples(),
                         inc[cons["code"].isin(cols)])},
        index=close.index)
    return mask if not mask.all().all() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    cache_dir = resolve_dir(cfg, "cache_dir")
    start = pd.to_datetime(cfg["data"]["start_date"]).strftime("%Y-%m-%d")
    end = cfg["data"]["end_date"] or None

    panels = load_panels(cache_dir)
    close = panels["close"]
    if end:
        panels = {k: v[v.index <= pd.Timestamp(end)] for k, v in panels.items()}
        close = panels["close"]

    bench = load_benchmark_index(cache_dir, "sh" + cfg["universe"]["index_code"])
    bench_close = bench["close"].reindex(close.index).ffill()

    si = load_search_index(cfg, close, panels["volume"])
    attention = load_attention_panels(cache_dir, close.index, close.columns,
                                      gdhs_span=cfg.get("attention", {}).get("gdhs_span", 4))
    if attention:
        print(f"[attention] 真实关注度代理: {', '.join(attention.keys())}")
    else:
        print("[attention] 未缓存真实关注度数据(可运行 scripts/fetch_attention.py), 仅用搜索指数因子")
    factors = build_factors(panels, si, attention)
    entries = cfg["factors"]["entries"]
    min_valid = cfg["factors"].get("min_valid_ratio", 0.5)

    # 行业/规模中性化: 申万一级哑变量 + ln(成交额60日均值)作规模代理
    neutral = None
    if cfg["factors"].get("neutralize", False):
        ind_fp = Path(cache_dir) / "industry_map.csv"
        if ind_fp.exists():
            ind_map = pd.read_csv(ind_fp, dtype={"code": str}).set_index("code")["industry"]
            size = np.log(panels["amount"].rolling(60, min_periods=20).mean())
            neutral = (ind_map, size)
            miss = set(close.columns) - set(ind_map.index)
            print(f"[factors] 行业/规模中性化: 启用 (覆盖 {len(close.columns) - len(miss)}/{len(close.columns)} 只)")
        else:
            print("[factors] 中性化已启用但缺少 industry_map.csv(运行 scripts/fetch_industry.py), 跳过")

    bt = cfg["backtest"]
    rebal = rebalance_dates(bench_close.index, start, pd.Timestamp(close.index.max()), bt["rebalance"])
    print(f"[backtest] 换仓日 {len(rebal)} 期: {rebal[0].date()} ~ {rebal[-1].date()}")

    weighting = cfg["factors"].get("weighting", "fixed")
    if weighting == "ic_adaptive":
        wmode = cfg["factors"].get("ic_weight_mode", "ic")
        composite = compute_composite_ic_weighted(
            factors, entries, panels["open"], rebal,
            window=cfg["factors"].get("ic_weight_window", 12), min_valid=min_valid,
            weight_mode=wmode, neutral=neutral)
        print(f"[factors] 合成方式: 滚动IC自适应加权 mode={wmode} "
              f"(window={cfg['factors'].get('ic_weight_window', 12)}期)")
    elif weighting == "fixed":
        composite = compute_composite(factors, entries, min_valid=min_valid, neutral=neutral)
        print("[factors] 合成方式: 固定权重")
    else:
        raise ValueError(f"不支持的因子加权方式: {weighting}")

    universe_mask = _build_universe_mask(cache_dir, cfg["universe"]["index_code"], close)
    if universe_mask is not None:
        print("[backtest] 股票池: 准点位还原(include_date过滤)")
    nav, hist, name_turnover = run_backtest(
        close, panels["open"], composite, rebal,
        top_n=bt["top_n"], cost_bps_one_way=bt["cost_bps_one_way"],
        min_history_days=bt["min_history_days"],
        buffer_ratio=bt.get("buffer_ratio", 0.0),
        limit_check=bt.get("limit_check", True),
        universe_mask=universe_mask)

    fwd = forward_open_ret(panels["open"], rebal)
    ic_df = pd.DataFrame({name: ic_series(f, fwd) for name, f in factors.items()})
    # 合成得分在信号日取值, 映射到执行日以对齐前瞻收益口径
    sig2exec = _sig_to_exec(panels["open"].index, rebal)
    comp_at_sig = composite.loc[rebal[rebal.isin(composite.index)]]
    ic_comp = ic_series(comp_at_sig.rename(index=sig2exec), fwd)
    ic_df["composite"] = ic_comp

    qret = quantile_returns(close, composite, rebal, q=bt["quantiles"])

    ew_pool = benchmark_equal_weight(close, start)
    idx_nav = bench_close / bench_close.iloc[0]
    idx_nav.name = "CSI300"
    nav_all = pd.concat([nav, ew_pool.rename("EW_Pool"), idx_nav], axis=1).dropna(how="all").ffill()

    metrics_all = {
        f"策略(Top{bt['top_n']})": performance_metrics(nav_all["strategy"]),
        "等权股票池": performance_metrics(nav_all["EW_Pool"]),
        "沪深300": performance_metrics(nav_all["CSI300"]),
    }
    # 样本外分段检验: 前后两半区间分别统计(观察是否过拟合于某段行情)
    mid = nav_all.index[len(nav_all) // 2]
    split_metrics = {
        f"策略·前半(~{mid.date()})": performance_metrics(nav_all["strategy"][nav_all.index < mid]),
        f"策略·后半({mid.date()}~)": performance_metrics(nav_all["strategy"][nav_all.index >= mid]),
        f"等权池·前半(~{mid.date()})": performance_metrics(nav_all["EW_Pool"][nav_all.index < mid]),
        f"等权池·后半({mid.date()}~)": performance_metrics(nav_all["EW_Pool"][nav_all.index >= mid]),
    }
    ic_stats_all = {name: ic_stats(ic_df[name]) for name in ic_df.columns}

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    hist.to_csv(results_dir / "holdings.csv", index=False)
    (results_dir / "name_turnover.csv").write_text(
        "" if name_turnover is None else name_turnover.to_csv(header=False))

    data_note = ("搜索指数: 优先真实CSV(data/search_index), 缺失列为合成数据"
                 if cfg["search_index"]["mode"] != "synthetic" else "搜索指数: 全部为合成数据")
    if weighting == "ic_adaptive":
        data_note += f"; 因子加权: 滚动{cfg['factors'].get('ic_weight_window', 12)}期样本外IC自适应(表中权重仅为配置基准)"
    else:
        data_note += "; 因子加权: 固定权重"
    run_report(cfg, nav_all, qret, ic_df, ic_comp, name_turnover,
               metrics_all, ic_stats_all, entries, len(close), data_note,
               split_metrics=split_metrics)

    print("\n" + "=" * 72)
    for name, m in metrics_all.items():
        print(f"{name:<14} " + "  ".join(f"{k}{v}" for k, v in m.items()))
    print("-" * 72)
    for name, m in split_metrics.items():
        print(f"{name:<20} " + "  ".join(f"{k}{v}" for k, v in m.items()))
    print("=" * 72)
    print("因子IC: " + "  ".join(f"{k}={v['IC均值']:+.3f}(t={v['t值']:+.1f})" for k, v in ic_stats_all.items()))


if __name__ == "__main__":
    main()
