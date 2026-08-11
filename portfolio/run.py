"""
portfolio/run.py — the front door: the only file that knows the study.

loads the panel, filters to the investable universe, 
builds the cash column once for both the engine and the metrics,
runs the arms on their own clocks, and writes one artifact set,
per arm — the weights frame, the daily path, and a manifest carrying the
summary statistics. The report is rendered FROM the manifests on disk, so
a partial re-run updates its own arms and the report stays consistent with
everything ever computed.

    python -m portfolio.run --splits train val
    python -m portfolio.run --splits train --arms one_over_n markowitz
    python -m portfolio.run --splits test --acknowledge-single-test-pass

The test split refuses to run without the acknowledgement flag: the study
touches test once, at the end, every strategy together.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import portfolio as cfg
from config.splits import get_split
from config.tickers import all_classes
from dataset import loader as dl
from portfolio import baselines, engine, metrics

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "portfolio_runs"
REPORT = Path(__file__).resolve().parent.parent / "reports" / "baselines_report.md"
SPLIT_ORDER = ("train", "val", "test")

CLOCKS = {
    "monthly": cfg.month_starts,
    "daily": lambda window: window,
}

# Portfolio results metrics
SUMMARY_COLS = ("annual_return", "volatility", "sharpe", "sortino",
                "max_drawdown", "calmar", "turnover_annual",
                "costs_annual_bp", "costs_total", "first_trade_cost",
                "max_asset_weight")


# One load of everything a run needs, over the FULL panel: estimation must
# reach back across split boundaries, so slicing happens per split later.
def load_bundle() -> dict:
    m = dl.matrices(None)
    classes = all_classes()
    inv = [t for members in classes.values() for t in members]
    ret = m["ret"][inv]
    tradeable = m["tradeable"][inv]

    # Ragged ENTRY only: an investable ticker must never stop trading after
    # its first session — the engine has no forced-liquidation path.
    tr = tradeable.to_numpy(dtype=bool)
    first = tr.argmax(axis=0)
    for j, t in enumerate(inv):
        if tr[:, j].any() and not tr[first[j]:, j].all():
            raise AssertionError(f"{t} goes non-tradeable after entry — the " 
                                 f"universe broke the ragged-entry property")

    return {"ret": ret, "tradeable": tradeable,
            "cash": cfg.cash_daily(m["macro"]), "classes": classes,
            "columns": inv + [engine.CASH], "dates": ret.index}

# Three files per arm: the decisions, the daily money path, and every
# number computed from them. The RL environment records into the same
# weights format, so any recorded run replays through engine.run().
def write_artifacts(out_dir: Path, name: str, weights: pd.DataFrame,
                    run: dict, manifest: dict) -> None:
    weights.to_parquet(out_dir / f"{name}_weights.parquet")
    pd.DataFrame({k: run[k] for k in ("value", "ret", "cost", "turnover")}
                 ).to_parquet(out_dir / f"{name}_path.parquet")
    (out_dir / f"{name}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")


# Run one split, one arm at a time. The producer never sees a cost rate and
# the engine never learns which producer made the weights.
def run_split(bundle: dict, split: str, arm_names: list, band: float) -> None:
    start, end = get_split(split)
    dates = bundle["dates"]
    window = dates[(dates >= start) & (dates <= end)]

    # What the engine sees: in-window returns with cash appended last. The
    # same cash series becomes the metrics risk-free, so holding cash can
    # never move Sharpe for free.
    ret_eval = bundle["ret"].loc[window].copy()
    ret_eval[engine.CASH] = bundle["cash"].reindex(window).values
    rates = cfg.cost_rates(bundle["columns"])
    rf = bundle["cash"].reindex(window)

    out_dir = OUT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in arm_names:
        spec = baselines.ARMS[name]
        reb = CLOCKS[spec["clock"]](window)

        # The first rebalance must have a complete estimation window.
        i = dates.get_loc(reb[0])
        if i + 1 < baselines.WINDOW:
            raise AssertionError(f"{split}/{name}: only {i + 1} sessions "
                                 f"precede the first rebalance; estimation "
                                 f"needs {baselines.WINDOW}")

        # Decide, simulate, score. The producer never sees a cost rate and
        # the engine never learns which producer made the weights.
        W, info = spec["producer"](bundle, reb, **spec.get("kwargs", {}))
        run = engine.run(W, ret_eval, rates, band=band)
        summary = metrics.summary(run, rf, start_value=1.0)
        per_year = metrics.per_year(run)

        # Sensitivity: the same decisions re-costed at the declared grid —
        # uniform one-way rate on every risky asset, cash capped at its own.
        sens = {}
        for bp in cfg.SENSITIVITY_BP:
            grid = pd.Series(bp / 1e4, index=bundle["columns"])
            grid[engine.CASH] = min(bp, cfg.COST_BP["cash"]) / 1e4
            r2 = engine.run(W, ret_eval, grid, band=band)
            sens[f"{bp:g}"] = metrics.summary(r2, rf, start_value=1.0)

        manifest = {
            "arm": name,
            "split": split,
            "clock": spec["clock"],
            "band": band,
            "rebalances": len(reb),
            "window": f"{window.min().date()} -> {window.max().date()}",
            "cost_bp": cfg.COST_BP,
            "info": info,
            "summary": summary,
            "sensitivity": sens,
            "per_year": per_year.to_dict(orient="index"),
        }
        write_artifacts(out_dir, name, W, run, manifest)

        print(f"  {name:14s} sharpe {summary['sharpe']:6.3f}  "
              f"ann {summary['annual_return'] * 100:6.2f}%  "
              f"mdd {summary['max_drawdown'] * 100:6.1f}%  "
              f"{info if info else ''}")

# The report is DERIVED from the manifests on disk — every arm and split
# ever computed, not just this invocation's.
def render_report() -> None:
    splits = [s for s in SPLIT_ORDER if (OUT_DIR / s).exists()]
    L = ["# Baselines report",
         "",
         "Generated by `python -m portfolio.run` from the manifests in "
         "`data/processed/portfolio_runs/` — never hand-edited.",
         "",
         f"Costs (one-way bp): {cfg.COST_BP}; monthly = first trading day "
         f"of the month; independent per-split runs starting in cash; the "
         f"first-trade cost column is the initial buy-in from cash, reported "
         f"so it can be netted out across splits of different length.", ""]
    for split in splits:
        manifests = {}
        order = list(baselines.ARMS) + sorted(
            p.stem[:-9] for p in (OUT_DIR / split).glob("*_manifest.json")
            if p.stem[:-9] not in baselines.ARMS)
        for name in order:
            p = OUT_DIR / split / f"{name}_manifest.json"
            if p.exists():
                manifests[name] = json.loads(p.read_text())
        if not manifests:
            continue
        L += [f"## {split}", ""]
        head = "| arm | clock | " + " | ".join(
            c.replace("_", " ") for c in SUMMARY_COLS) + " |"
        L += [head, "|" + "---|" * (len(SUMMARY_COLS) + 2)]
        for name, man in manifests.items():
            s = man["summary"]
            L.append("| " + " | ".join(
                [name, man["clock"]] + [f"{s[c]:.4f}" for c in SUMMARY_COLS])
                + " |")
        L.append("")

        grid = [f"{bp:g}" for bp in cfg.SENSITIVITY_BP]
        L += ["Sensitivity (Sharpe at uniform one-way cost):", "",
              "| arm | " + " | ".join(f"{g}bp" for g in grid) + " |",
              "|" + "---|" * (len(grid) + 1)]
        for name, man in manifests.items():
            L.append("| " + name + " | " + " | ".join(
                f"{man['sensitivity'][g]['sharpe']:.3f}" for g in grid) + " |")
        L.append("")

        years = sorted({y for man in manifests.values()
                        for y in man["per_year"]})
        L += ["Per-year returns (%; p = partial year):", "",
              "| year | " + " | ".join(manifests) + " |",
              "|" + "---|" * (len(manifests) + 1)]
        for y in years:
            row = [y]
            for man in manifests.values():
                py = man["per_year"].get(y)
                row.append("—" if py is None else
                           f"{py['return'] * 100:+.2f}"
                           + (" p" if "(partial)" in py["label"] else ""))
            L.append("| " + " | ".join(row) + " |")
        L.append("")

        notes = {n: m["info"] for n, m in manifests.items() if m["info"]}
        if notes:
            L += ["Diagnostics: " + "; ".join(
                f"{n}: {v}" for n, v in notes.items()), ""]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"report -> {REPORT}")

if __name__ == "__main__":
    from collectors._core import console_utf8
    console_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--arms", nargs="+", default=list(baselines.ARMS))
    ap.add_argument("--band", type=float, default=0.0)
    ap.add_argument("--acknowledge-single-test-pass", action="store_true")
    args = ap.parse_args()

    if "test" in args.splits and not args.acknowledge_single_test_pass:
        raise SystemExit("test is touched ONCE, at the end, every strategy "
                         "together. Pass --acknowledge-single-test-pass if "
                         "this is that moment.")
    unknown = [a for a in args.arms if a not in baselines.ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; have {list(baselines.ARMS)}")

    bundle = load_bundle()
    for split in args.splits:
        print(f"{split}:")
        run_split(bundle, split, args.arms, args.band)
    render_report()
