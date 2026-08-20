"""运行: python tests/test_factors.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import ic_series, ic_stats, run_backtest
from src.data.attention_proxy import load_attention_panels
from src.data.search_index import generate_synthetic
from src.factors import build_factors, compute_composite
from src.factors.attention_proxy import lhb_heat
from src.factors.pipeline import compute_composite_ic_weighted, neutralize_cross, rebalance_dates

ENTRIES = {
    "search_heat": {"weight": 0.28, "direction": 1},
    "search_mom": {"weight": 0.10, "direction": 1},
    "attn_price_div": {"weight": 0.12, "direction": 1},
    "mom_12_1": {"weight": 0.15, "direction": 1},
    "rev_5": {"weight": 0.10, "direction": -1},
    "low_vol_20": {"weight": 0.10, "direction": -1},
    "volume_ratio": {"weight": 0.05, "direction": -1},
    "amihud": {"weight": 0.10, "direction": 1},
}


def make_panels(n_days=400, n_codes=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    codes = [f"{i:06d}" for i in range(n_codes)]
    ret = rng.normal(0.0005, 0.02, size=(n_days, n_codes))
    close = pd.DataFrame(100 * np.exp(np.cumsum(ret, axis=0)), index=dates, columns=codes)
    volume = pd.DataFrame(rng.lognormal(13, 0.6, size=(n_days, n_codes)), index=dates, columns=codes)
    amount = close * volume
    return {"open": close.shift(1) * 1.001, "close": close, "volume": volume, "amount": amount}


def test_no_lookahead():
    panels = make_panels()
    si = generate_synthetic(panels["close"], panels["volume"], seed=1)
    factors = build_factors(panels, si)
    t = panels["close"].index[300]
    vals_before = {k: f.loc[t].copy() for k, f in factors.items()}

    panels2 = {k: v.copy() for k, v in panels.items()}
    mask = panels2["close"].index > t
    for k in panels2:
        panels2[k].loc[mask] = panels2[k].loc[mask] * 1.5
    si2 = generate_synthetic(panels2["close"], panels2["volume"], seed=1)
    factors2 = build_factors(panels2, si2)
    for k in factors:
        pd.testing.assert_series_equal(factors[k].loc[t], vals_before[k], check_names=False,
                                       obj=f"lookahead check for {k}")
    print("PASS no_lookahead")


def test_composite_properties():
    panels = make_panels()
    si = generate_synthetic(panels["close"], panels["volume"], seed=1)
    factors = build_factors(panels, si)
    comp = compute_composite(factors, ENTRIES)
    late = comp.iloc[260:]
    assert late.notna().values.sum() > 0, "composite 全为 NaN"
    row_std = late.std(axis=1).dropna()
    assert (row_std < 2).all(), "合成因子截面方差异常"
    cs_mean = late.mean(axis=1).abs()
    assert (cs_mean < 0.2).all(), "合成因子截面均值应接近0"
    print("PASS composite_properties")


def test_composite_missing_factors():
    """因子缺失时: 权重按可用因子重新归一, 缺失过多则整行 NaN"""
    panels = make_panels()
    si = generate_synthetic(panels["close"], panels["volume"], seed=1)
    factors = build_factors(panels, si)
    comp_full = compute_composite(factors, ENTRIES)

    t = panels["close"].index[320]
    code = panels["close"].columns[5]
    # 剔除 2 个因子(仍超过 50% 覆盖): 该股得分不应被拉向 0
    factors2 = {k: v.copy() for k, v in factors.items()}
    factors2["amihud"].loc[t, code] = np.nan
    factors2["low_vol_20"].loc[t, code] = np.nan
    comp2 = compute_composite(factors2, ENTRIES)
    assert np.isfinite(comp2.loc[t, code]), "覆盖过半的因子缺失不应置NaN"
    assert abs(comp2.loc[t, code]) > 1e-8 or abs(comp_full.loc[t, code]) < 1e-8, \
        "缺失因子不应等价于中性填充"

    # 剔除过半因子: 该股应置 NaN
    factors3 = {k: v.copy() for k, v in factors.items()}
    for name in ["amihud", "low_vol_20", "mom_12_1", "volume_ratio", "rev_5"]:
        factors3[name].loc[t, code] = np.nan
    comp3 = compute_composite(factors3, ENTRIES)
    assert np.isnan(comp3.loc[t, code]), "可用因子不足50%应置NaN"
    print("PASS composite_missing_factors")


def test_rebalance_dates():
    days = pd.bdate_range("2023-01-02", "2023-06-30")
    rebal = rebalance_dates(days, "2023-01-01", "2023-06-30", "month_end")
    assert len(rebal) == 6
    assert all(d == days[days.month == m].max() for m, d in zip(range(1, 7), rebal)), \
        "换仓日应为每月最后一个交易日"
    print("PASS rebalance_dates")


def test_ic_stats_direction():
    """因子与前瞻收益强正相关时, IC 应显著为正"""
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2023-01-02", periods=24)
    codes = [f"{i:06d}" for i in range(60)]
    ret = rng.normal(0, 0.05, size=(len(dates), len(codes)))
    fwd_ret = pd.DataFrame(ret.T, index=codes, columns=dates)  # 行=股票, 列=信号日
    factor = pd.DataFrame(ret * 2 + rng.normal(0, 0.02, ret.shape), index=dates, columns=codes)
    ic = ic_series(factor, fwd_ret)
    s = ic_stats(ic)
    assert len(ic) == len(dates)
    assert s["IC均值"] > 0.5 and s["t值"] > 5, f"IC 统计异常: {s}"
    print("PASS ic_stats_direction")


def test_ic_weighted_no_lookahead():
    """IC自适应加权: 修改未来数据不影响历史信号日的合成得分"""
    panels = make_panels()
    si = generate_synthetic(panels["close"], panels["volume"], seed=1)
    factors = build_factors(panels, si)
    days = panels["close"].index
    rebal = rebalance_dates(days, days[60], days[-1], "month_end")
    comp = compute_composite_ic_weighted(factors, ENTRIES, panels["open"], rebal,
                                         window=4, min_periods=2)
    t = rebal[len(rebal) // 2]
    assert comp.loc[t].notna().any(), "历史信号日合成得分不应全为NaN"

    panels2 = {k: v.copy() for k, v in panels.items()}
    mask = panels2["close"].index > t
    for k in panels2:
        panels2[k].loc[mask] = panels2[k].loc[mask] * 1.5
    si2 = generate_synthetic(panels2["close"], panels2["volume"], seed=1)
    factors2 = build_factors(panels2, si2)
    comp2 = compute_composite_ic_weighted(factors2, ENTRIES, panels2["open"], rebal,
                                          window=4, min_periods=2)
    pd.testing.assert_series_equal(comp.loc[t], comp2.loc[t], check_names=False,
                                   obj="ic-weighted lookahead check")
    print("PASS ic_weighted_no_lookahead")


def test_engine_costs():
    """T日信号/T+1开盘成交口径: 无成本时净值=买入股票表现, 有成本时精确扣减双边费用"""
    days = pd.bdate_range("2023-01-02", periods=24)
    codes = ["000001", "000002", "000003"]
    close = pd.DataFrame({
        "000001": np.concatenate([np.full(21, 10.0), [11.0, 11.0, 11.0]]),
        "000002": np.full(24, 5.0), "000003": np.full(24, 8.0),
    }, index=days)
    open_ = close.shift(1).fillna(10.0)
    open_.loc[days[21]] = [10.0, 5.0, 8.0]  # T+1 开盘价
    composite = pd.DataFrame(0.0, index=days, columns=codes)
    composite.loc[days[20], "000001"] = 1.0  # 仅选中 000001
    sig = pd.DatetimeIndex([days[20]])

    nav0, hist, _ = run_backtest(close, open_, composite, sig, top_n=1,
                                 cost_bps_one_way=0, min_history_days=5)
    assert len(hist) == 1 and hist["tickers"].iloc[0] == "000001"
    assert nav0.iloc[20] == 1.0
    assert abs(nav0.iloc[-1] - 1.1) < 1e-9, "无成本时净值应等于持仓涨幅"

    cost = 15.0
    nav1, _, _ = run_backtest(close, open_, composite, sig, top_n=1,
                              cost_bps_one_way=cost, min_history_days=5)
    # 仅一次买入(扣买入成本), 期末未卖出
    expected = (1 - cost * 1e-4) * 1.1
    assert abs(nav1.iloc[-1] - expected) < 1e-9, \
        f"有成本净值应为 {expected}, 实际 {nav1.iloc[-1]}"
    print("PASS engine_costs")


def test_attention_proxy_no_lookahead(tmp_path=None):
    """真实关注度代理: 龙虎榜事件次日才可用, 股东户数公告次日才可用"""
    import tempfile

    days = pd.bdate_range("2023-01-02", periods=30)
    codes = ["000001", "000002"]
    with tempfile.TemporaryDirectory() as tmp:
        att = Path(tmp) / "attention"
        (att / "gdhs").mkdir(parents=True)
        # 龙虎榜: 000001 在 days[10] 上榜
        lhb = pd.DataFrame({
            "code": ["000001"], "date": [days[10]], "net_amt": [1e7],
            "turnover": [5.0], "reason": ["测试"]})
        lhb.to_parquet(att / "lhb_detail.parquet")
        # 股东户数: 000002 在 days[12] 公告户数+10%
        gdhs = pd.DataFrame({
            "cutoff_date": [days[5]], "announce_date": [days[12]],
            "holders": [10000], "chg_pct": [10.0]})
        gdhs.to_parquet(att / "gdhs" / "000002.parquet")

        panels = load_attention_panels(Path(tmp), days, codes, gdhs_span=1)
        ev, net, g = panels["lhb_events"], panels["lhb_net"], panels["gdhs_chg"]
        # 上榜当日不可知, 次日起可用
        assert ev.loc[days[10], "000001"] == 0 and ev.loc[days[11], "000001"] == 1
        assert net.loc[days[11], "000001"] == 1e7
        # 公告当日不可知, 次日起 ffill
        assert np.isnan(g.loc[days[12], "000002"]) and g.loc[days[13], "000002"] == 10.0
        assert g.loc[days[20], "000002"] == 10.0
        # lhb_heat 仅用历史事件
        heat = lhb_heat(ev)
        assert heat.loc[days[10]].max() == 0 and heat.loc[days[11], "000001"] > 0
        assert heat.loc[days[25], "000001"] < heat.loc[days[11], "000001"], "热度应随时间衰减"
    print("PASS attention_proxy_no_lookahead")


def test_engine_buffer_limits():
    """换手缓冲带保留原持仓; 开盘涨停买不进/跌停卖不出"""
    days = pd.bdate_range("2023-01-02", periods=45)
    codes = ["000001", "000002", "000003", "000004"]
    close = pd.DataFrame(10.0, index=days, columns=codes)
    open_ = close.shift(1).fillna(10.0)
    composite = pd.DataFrame(0.0, index=days, columns=codes)
    composite.loc[days[20], ["000001", "000002"]] = [4.0, 3.0]   # 一期: 持 1,2
    composite.loc[days[30], codes] = [2.0, 1.0, 4.0, 3.0]        # 二期: 1 排第3仍在缓冲带内
    sig = pd.DatetimeIndex([days[20], days[30]])

    # 二期开盘: 000002 跌停(不可卖), 000003 涨停(不可买)
    open_.loc[days[31], ["000002", "000003"]] = [8.99, 11.01]
    nav, hist, _ = run_backtest(close, open_, composite, sig, top_n=2,
                                cost_bps_one_way=15, min_history_days=5,
                                buffer_ratio=0.6, limit_check=True, start_nav=1e6)
    assert hist["tickers"].iloc[0] == "000001 000002"
    # 000001 缓冲带保留不交易, 000002 跌停卖不出, 000003 涨停买不进 → 无成交无成本
    assert hist["tickers"].iloc[1] == "000001 000002"
    assert hist["cost_paid"].iloc[1] == 0, "无成交日不应产生成本"

    # 关闭涨跌停限制后: 000002 卖出, 000001 保留, 000003 买入
    nav2, hist2, _ = run_backtest(close, open_, composite, sig, top_n=2,
                                  cost_bps_one_way=15, min_history_days=5,
                                  buffer_ratio=0.6, limit_check=False, start_nav=1e6)
    assert hist2["tickers"].iloc[1] == "000001 000003"
    assert hist2["cost_paid"].iloc[1] > 0
    print("PASS engine_buffer_limits")


def test_universe_mask():
    """时点股票池掩码: 被排除的股票即使得分最高也不参与选股"""
    days = pd.bdate_range("2023-01-02", periods=24)
    codes = ["000001", "000002", "000003"]
    close = pd.DataFrame(10.0, index=days, columns=codes)
    open_ = close.shift(1).fillna(10.0)
    composite = pd.DataFrame(0.0, index=days, columns=codes)
    composite.loc[days[20], "000001"] = 10.0  # 得分最高但被掩码排除
    composite.loc[days[20], "000002"] = 1.0
    mask = pd.DataFrame(True, index=days, columns=codes)
    mask["000001"] = False
    nav, hist, _ = run_backtest(close, open_, composite, pd.DatetimeIndex([days[20]]),
                                top_n=1, cost_bps_one_way=0, min_history_days=5,
                                universe_mask=mask)
    assert hist["tickers"].iloc[0] == "000002", "掩码排除的股票不应入选"
    print("PASS universe_mask")


def test_neutralize_cross():
    """中性化后: 残差的行业组间均值差与规模相关性应归零"""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2023-01-02", periods=5)
    codes = [f"{i:06d}" for i in range(60)]
    industry = pd.Series(["A"] * 30 + ["B"] * 30, index=codes)
    size = pd.DataFrame(rng.normal(0, 1, (5, 60)), index=dates, columns=codes)
    # 因子 = 行业效应 + 规模效应 + 噪声
    ind_eff = pd.Series([1.0] * 30 + [-1.0] * 30, index=codes)
    factor = pd.DataFrame(
        rng.normal(0, 0.1, (5, 60)) + 2.0 * size.to_numpy()
        + np.tile(ind_eff.values, (5, 1)), index=dates, columns=codes)
    resid = neutralize_cross(factor, industry, size)
    for t in dates:
        r = resid.loc[t]
        assert abs(r[industry == "A"].mean() - r[industry == "B"].mean()) < 1e-9, "行业效应未剥离"
        assert abs(np.corrcoef(r.values, size.loc[t].values)[0, 1]) < 1e-9, "规模效应未剥离"
    print("PASS neutralize_cross")


if __name__ == "__main__":
    test_no_lookahead()
    test_composite_properties()
    test_composite_missing_factors()
    test_rebalance_dates()
    test_ic_stats_direction()
    test_ic_weighted_no_lookahead()
    test_engine_costs()
    test_attention_proxy_no_lookahead()
    test_engine_buffer_limits()
    test_universe_mask()
    test_neutralize_cross()
    print("ALL TESTS PASSED")
