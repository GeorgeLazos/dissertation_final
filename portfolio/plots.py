"""
portfolio/plots.py — figures from the run artifacts, nothing recomputed.

Reads the per-arm Parquet paths, weights and manifests that run.py wrote and
renders the study's figures into figures/{split}/, in the format(s) named by
FORMATS. Each run clears the split's folder and rewrites README.md, so the
directory always holds exactly the current set — same artifacts, same figures.

    equity          all arms' value paths, log scale
    drawdown        distance below each arm's running peak
    profile_*       one card per arm: equity, drawdown, per-year bars,
                    class shares and the summary numbers on one page
    sensitivity     Sharpe against the cost grid, per arm
    per_year        calendar-year returns, grouped bars

    python -m portfolio.plots                 # every split with artifacts
    python -m portfolio.plots --splits train
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from config import portfolio as cfg
from config.tickers import all_classes
from portfolio import baselines, engine
from portfolio.run import OUT_DIR, SPLIT_ORDER

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

# Fixed identity colors (categorical slots, assigned per entity, never
# cycled): arms keep one color everywhere; sleeves likewise in their figure.
ARM_COLOR = {
    "one_over_n": "#2a78d6",
    "markowitz": "#eb6834",
    "min_variance": "#1baf7a",
    "risk_parity": "#eda100",
    "fixed_mix": "#e87ba4",
}
# Reserve slots for arms not named above, assigned stably by name order.
RESERVE_COLORS = ("#008300", "#4a3aa7", "#e34948", "#52514e")


def arm_color(name: str, roster: list) -> str:
    if name in ARM_COLOR:
        return ARM_COLOR[name]
    unknown = sorted(n for n in roster if n not in ARM_COLOR)
    return RESERVE_COLORS[unknown.index(name) % len(RESERVE_COLORS)]


CLASS_COLOR = {
    "equities": "#2a78d6",
    "bonds": "#eb6834",
    "commodities": "#eda100",
    "reits": "#1baf7a",
    "CASH": "#e87ba4",
}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_C = "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": BASELINE_C,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "font.size": 9,
})


def _style(ax, title: str, sub: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK,
                 pad=22 if sub else 10)
    if sub:
        ax.text(0.0, 1.04, sub, transform=ax.transAxes,
                fontsize=8, color=MUTED, va="bottom")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# Market episodes shaded on the time charts, per split.
EVENTS = {
    "train": [("2008-09-01", "2009-03-31", "2008–09 crisis"),
              ("2018-10-01", "2018-12-31", "late-2018 selloff")],
    "val": [("2020-02-15", "2020-03-31", "COVID crash")],
    "test": [("2022-01-01", "2022-10-31", "2022 rate shock")],
}


def _mark_events(ax, split: str) -> None:
    for start, end, label in EVENTS.get(split, []):
        a, b = pd.Timestamp(start), pd.Timestamp(end)
        ax.axvspan(a, b, color=GRID, alpha=0.45, zorder=0)
        ax.text(a + (b - a) / 2, 0.985, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7, color=MUTED)


FORMATS = ("png",)   # switch to ("pdf",) for the LaTeX build


def _save(fig, split: str, name: str) -> None:
    out = FIG_DIR / split
    out.mkdir(parents=True, exist_ok=True)
    for ext in FORMATS:
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  figures/{split}/{name}." + "|".join(FORMATS))


def _load(split: str) -> dict:
    arms = {}
    order = list(baselines.ARMS) + sorted(
        p.stem[:-9] for p in (OUT_DIR / split).glob("*_manifest.json")
        if p.stem[:-9] not in baselines.ARMS)
    for name in order:
        d = OUT_DIR / split
        if (d / f"{name}_manifest.json").exists():
            arms[name] = {
                "path": pd.read_parquet(d / f"{name}_path.parquet"),
                "weights": pd.read_parquet(d / f"{name}_weights.parquet"),
                "manifest": json.loads(
                    (d / f"{name}_manifest.json").read_text()),
            }
    return arms


def plot_equity(split: str, arms: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name, a in arms.items():
        v = a["path"]["value"]
        col = arm_color(name, list(arms))
        ax.plot(v.index, v.values, lw=2, color=col, label=name)
        ax.annotate(f" {name}  {v.iloc[-1]:.2f}x", (v.index[-1], v.iloc[-1]),
                    color=col, fontsize=8, va="center")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.FixedLocator([1, 1.5, 2, 3, 4, 6, 8]))
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:g}x")
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylabel("portfolio value (log, start = 1)")
    ax.margins(x=0.14)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _mark_events(ax, split)
    _style(ax, f"Growth of 1 unit — {split}, net of costs",
           "What 1 unit invested at the start is worth each day.")
    _save(fig, split, "equity")


def plot_drawdown(split: str, arms: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for name, a in arms.items():
        v = a["path"]["value"]
        dd = v / v.cummax() - 1.0
        ax.plot(dd.index, dd.values, lw=2, color=arm_color(name, list(arms)),
                label=f"{name}  (max {dd.min() * 100:.0f}%)")
    ax.set_ylabel("distance below running peak")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x * 100:.0f}%")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _mark_events(ax, split)
    _style(ax, f"Drawdown — {split}",
           "How far below its own record high each arm sits; 0% = new high.")
    _save(fig, split, "drawdown")


def _class_share_ax(ax, w) -> None:
    classes = all_classes()
    shares = pd.DataFrame(index=w.index)
    for cls, members in classes.items():
        shares[cls] = w[[t for t in members if t in w.columns]].sum(axis=1)
    shares["CASH"] = w[engine.CASH]
    ax.stackplot(shares.index, [shares[c].values for c in shares.columns],
                 labels=list(shares.columns),
                 colors=[CLASS_COLOR[c] for c in shares.columns],
                 edgecolor=SURFACE, linewidth=0.4)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x * 100:.0f}%")
    ax.legend(frameon=False, fontsize=8, ncol=5,
              loc="upper center", bbox_to_anchor=(0.5, -0.15))


# One card per arm: everything about a single strategy on one page.
def plot_profiles(split: str, arms: dict) -> None:
    for name, a in arms.items():
        col = arm_color(name, list(arms))
        v = a["path"]["value"]
        man = a["manifest"]
        fig = plt.figure(figsize=(10, 8.5))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 0.9],
                              hspace=0.5, wspace=0.28)

        ax = fig.add_subplot(gs[0, 0])
        ax.plot(v.index, v.values, lw=2, color=col)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(
            mticker.FixedLocator([1, 1.5, 2, 3, 4, 6, 8]))
        ax.yaxis.set_major_formatter(lambda x, _: f"{x:g}x")
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        _style(ax, "Growth of 1 unit (log)")

        ax = fig.add_subplot(gs[0, 1])
        py = man["per_year"]
        years = list(py)
        vals = [py[y]["return"] * 100 for y in years]
        ax.bar(np.arange(len(years)), vals, width=0.7, color=col)
        ax.axhline(0, color=BASELINE_C, lw=1)
        ax.set_xticks(np.arange(len(years)))
        ax.set_xticklabels([y[2:] for y in years], fontsize=7)
        _style(ax, "Per-year return (%)")

        ax = fig.add_subplot(gs[1, 0])
        dd = v / v.cummax() - 1.0
        ax.plot(dd.index, dd.values, lw=2, color=col)
        ax.yaxis.set_major_formatter(lambda x, _: f"{x * 100:.0f}%")
        _style(ax, f"Drawdown  (max {dd.min() * 100:.0f}%)")

        ax = fig.add_subplot(gs[1, 1])
        ax.axis("off")
        s_ = man["summary"]
        rows = [
            ("annual return", f"{s_['annual_return'] * 100:+.2f}%"),
            ("volatility", f"{s_['volatility'] * 100:.2f}%"),
            ("Sharpe", f"{s_['sharpe']:.3f}"),
            ("Sortino", f"{s_['sortino']:.3f}"),
            ("max drawdown", f"{s_['max_drawdown'] * 100:.1f}%"),
            ("Calmar", f"{s_['calmar']:.3f}"),
            ("turnover / yr", f"{s_['turnover_annual'] * 100:.0f}%"),
            ("cost drag / yr", f"{s_['costs_annual_bp']:.1f} bp"),
            ("largest position", f"{s_['max_asset_weight'] * 100:.1f}%"),
            ("rebalances", f"{man['rebalances']} ({man['clock']})"),
        ]
        ax.set_title("Summary", loc="left", fontsize=11, color=INK, pad=10)
        for i, (k, val) in enumerate(rows):
            y = 0.95 - i * 0.10
            ax.text(0.02, y, k, fontsize=9, color=MUTED, va="top")
            ax.text(0.98, y, val, fontsize=9, color=INK, va="top",
                    ha="right", fontfamily="sans-serif")

        ax = fig.add_subplot(gs[2, :])
        _class_share_ax(ax, a["weights"])
        _style(ax, "Class allocation at each rebalance")

        fig.suptitle(f"{name} — {split}", x=0.01, ha="left",
                     fontsize=13, color=INK)
        _save(fig, split, f"profile_{name}")


def plot_sensitivity(split: str, arms: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    grid = [f"{bp:g}" for bp in cfg.SENSITIVITY_BP]
    x = [float(g) for g in grid]
    for name, a in arms.items():
        y = [a["manifest"]["sensitivity"][g]["sharpe"] for g in grid]
        col = arm_color(name, list(arms))
        ax.plot(x, y, lw=2, marker="o", ms=4, color=col)
        ax.annotate(f" {name}", (x[-1], y[-1]), color=col,
                    fontsize=8, va="center")
    ax.set_xticks(x)
    ax.set_xlabel("one-way cost (bp)")
    ax.set_ylabel("Sharpe")
    ax.margins(x=0.25)
    _style(ax, f"Cost sensitivity — {split}, same decisions re-costed")
    _save(fig, split, "sensitivity")


def plot_per_year(split: str, arms: dict) -> None:
    years = sorted({y for a in arms.values() for y in a["manifest"]["per_year"]})
    fig, ax = plt.subplots(figsize=(8, 3.6))
    n = len(arms)
    width = 0.8 / n
    for j, (name, a) in enumerate(arms.items()):
        py = a["manifest"]["per_year"]
        vals = [py.get(y, {}).get("return", np.nan) for y in years]
        pos = np.arange(len(years)) + (j - (n - 1) / 2) * width
        ax.bar(pos, [v * 100 if v == v else 0 for v in vals],
               width=width * 0.92, color=arm_color(name, list(arms)),
               label=name)
    ax.axhline(0, color=BASELINE_C, lw=1)
    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels([y if int(y) % 2 == 1 else "" for y in years]
                       if len(years) > 8 else years, fontsize=8)
    ax.set_ylabel("calendar-year return (%)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _style(ax, f"Per-year returns — {split}",
           "Each bar: that calendar year's return for that arm.")
    _save(fig, split, "per_year")


README = """# Figures

Generated by `python -m portfolio.plots` from the artifacts in
`data/processed/portfolio_runs/` — regenerated in full on every run,
never edited by hand. One folder per split.

| figure | what it shows |
|---|---|
| equity | what 1 unit invested at the start is worth each day (log scale, so equal slopes = equal % growth) |
| drawdown | how far below its own record high each arm sits; depth = worst loss, width = time underwater |
| per_year | each calendar year's return per arm — was the average earned steadily or by one big year? |
| sensitivity | Sharpe with the SAME decisions re-priced at 0/5/10/25bp costs — does the ranking survive? |
| profile_{arm} | one page per arm: its equity, drawdown, yearly bars, summary numbers and class allocation |

Shaded bands on the time charts mark market episodes (2008-09 crisis,
COVID crash) so dips can be tied to events.
"""


if __name__ == "__main__":
    from collectors._core import console_utf8
    console_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+",
                    default=[s for s in SPLIT_ORDER if (OUT_DIR / s).exists()])
    args = ap.parse_args()
    if not args.splits:
        raise SystemExit("no run artifacts in data/processed/portfolio_runs/ "
                         "— run python -m portfolio.run first")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "README.md").write_text(README, encoding="utf-8")
    for split in args.splits:
        arms = _load(split)
        if not arms:
            print(f"{split}: no artifacts, skipped")
            continue
        out = FIG_DIR / split
        if out.exists():
            for f in out.glob("*"):
                f.unlink()
        print(f"{split}:")
        plot_equity(split, arms)
        plot_drawdown(split, arms)
        plot_profiles(split, arms)
        plot_sensitivity(split, arms)
        plot_per_year(split, arms)
