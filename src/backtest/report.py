import datetime as dt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..config import PROJECT_ROOT

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_nav(nav: pd.DataFrame, plot_dir: Path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    x = nav.index.to_numpy()
    for col in nav.columns:
        axes[0].plot(x, nav[col].to_numpy(), label=col, linewidth=1.3)
    axes[0].set_title("Strategy NAV vs Benchmarks")
    axes[0].legend()
    dd = nav.div(nav.cummax()) - 1
    for col in dd.columns:
        axes[1].fill_between(x, dd[col].to_numpy(), 0, alpha=0.5, label=col)
    axes[1].set_title("Drawdown")
    axes[1].legend()
    _save(fig, plot_dir / "nav.png")


def plot_quantiles(qret: pd.DataFrame, plot_dir: Path):
    ann = qret.mean() * 12
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ann.plot.bar(ax=ax, color=["#d0573e", "#e8a459", "#c9c9c9", "#8aa9c4", "#3d78b5"])
    ls = (qret.iloc[:, -1] - qret.iloc[:, 0]).mean() * 12
    ax.bar(["L/S"], [ls], color="#2e2e2e", alpha=0.8)
    ax.set_title(f"Quantile Annualized Returns (monthly rebalance), L/S={ls:+.2%}")
    ax.set_ylabel("ann. return")
    _save(fig, plot_dir / "quantiles.png")


def plot_ic(ic_df: pd.DataFrame, plot_dir: Path):
    stats = pd.DataFrame({
        "mean": ic_df.mean(), "t": ic_df.mean() / (ic_df.std() + 1e-12) * len(ic_df) ** 0.5,
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    stats["mean"].plot.bar(ax=axes[0], color="#3d78b5")
    axes[0].set_title("Rank IC mean (period return vs factor)")
    stats["t"].plot.bar(ax=axes[1], color="#d0573e")
    axes[1].axhline(2, ls="--", c="k", lw=0.8)
    axes[1].axhline(-2, ls="--", c="k", lw=0.8)
    axes[1].set_title("IC t-stat")
    _save(fig, plot_dir / "ic.png")


def plot_ic_cumsum(ic_comp: pd.Series, name_turnover: pd.Series, plot_dir: Path):
    if name_turnover is None or len(name_turnover) == 0:
        return
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=False)
    axes[0].plot(ic_comp.index.to_numpy(), ic_comp.cumsum().to_numpy(), lw=1.4, color="#3d78b5")
    axes[0].set_title(f"Composite Rank IC cumsum (mean={ic_comp.mean():.3f})")
    axes[1].plot(name_turnover.index.to_numpy(), name_turnover.to_numpy(), color="#d0573e", lw=1.2)
    axes[1].set_title("Portfolio name turnover at rebalance")
    _save(fig, plot_dir / "ic_turnover.png")


def write_report(results_dir: Path, cfg, metrics_all: dict, ic_stats_all: dict,
                 qret_ann: pd.Series, factors_meta: dict, n_days: int, data_note: str,
                 split_metrics: dict = None):
    lines = [
        f"# 搜索关注度多因子选股 回测报告",
        "",
        f"- 生成时间: {dt.datetime.now():%Y-%m-%d %H:%M}",
        f"- 股票池: 沪深{int(cfg['universe']['index_code'])}成分股 (含include_date准点位还原; 无剔除日期, 幸存者偏差部分残留)",
        f"- 回测区间: {n_days} 个交易日",
        f"- 数据说明: {data_note}",
        "",
        "## 因子设置",
        "",
        "| 因子 | 权重 | 方向 | 说明 |",
        "|---|---|---|---|",
    ]
    for name, e in factors_meta.items():
        lines.append(f"| {name} | {e['weight']} | {e.get('direction', 1):+d} | {e.get('desc', '')} |")
    metric_names = list(next(iter(metrics_all.values())).keys())
    lines += ["", "## 净值表现", "",
              "| 策略 | " + " | ".join(metric_names) + " |",
              "|---|" + "---|" * len(metric_names)]
    for k, m in metrics_all.items():
        lines.append("| " + " | ".join([k] + [m[n] for n in metric_names]) + " |")
    if split_metrics:
        lines += ["", "## 分段样本表现(过拟合检查)", "",
                  "| 分段 | " + " | ".join(metric_names) + " |",
                  "|---|" + "---|" * len(metric_names)]
        for k, m in split_metrics.items():
            lines.append("| " + " | ".join([k] + [m[n] for n in metric_names]) + " |")
    lines += ["", "## 因子Rank IC (执行口径前瞻收益)", "", "| 因子 | IC均值 | ICIR | IC胜率 | t值 |", "|---|---|---|---|---|"]
    for name, s in ic_stats_all.items():
        lines.append(f"| {name} | {s['IC均值']:+.3f} | {s['ICIR']:+.3f} | {s['IC胜率']:.1%} | {s['t值']:+.2f} |")
    lines += ["", "## 分层年化收益", "", "| 分组 | 年化 |", "|---|---|"]
    for g, v in qret_ann.items():
        lines.append(f"| {g} | {v:+.2%} |")
    lines += [
        "", "## 图表", "",
        "![nav](plots/nav.png)", "", "![quantiles](plots/quantiles.png)", "",
        "![ic](plots/ic.png)", "", "![ic_turnover](plots/ic_turnover.png)", "",
        "## 重要声明", "",
        "- 成分股使用当前沪深300名单, 存在幸存者偏差; 个股为后复权行情(收益率口径不受影响), 基准指数为不复权行情.",
        "- 若使用合成搜索指数(默认), 结果仅用于验证策略框架, 不代表可交易机会.",
        "- 涨跌停不可成交已做一字板近似处理(开盘即涨/跌停视为不可买/卖); 停牌复牌冲击与部分成交未建模, 实际成交成本可能更高.",
    ]
    (results_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_report(cfg, nav_all: pd.DataFrame, qret: pd.DataFrame, ic_df: pd.DataFrame,
               ic_comp: pd.Series, name_turnover: pd.Series, metrics_all: dict,
               ic_stats_all: dict, factors_meta: dict, n_days: int, data_note: str,
               split_metrics: dict = None):
    results_dir = PROJECT_ROOT / "results"
    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    nav_all.to_csv(results_dir / "nav.csv")
    qret.to_csv(results_dir / "quantile_returns.csv")
    ic_df.to_csv(results_dir / "factor_ic.csv")

    plot_nav(nav_all, plot_dir)
    plot_quantiles(qret, plot_dir)
    plot_ic(ic_df, plot_dir)
    plot_ic_cumsum(ic_comp, name_turnover, plot_dir)
    qret_ann = qret.mean() * 12
    write_report(results_dir, cfg, metrics_all, ic_stats_all, qret_ann, factors_meta, n_days, data_note,
                 split_metrics=split_metrics)
    print(f"[report] 已输出至 {results_dir}/")
