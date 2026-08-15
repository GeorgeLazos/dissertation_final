"""
portfolio/report.py — every written report in one place.

Three report families share this module. RUN REPORTS document a single
training run's folder under agent_runs/: report.md (the run's numbers
against its naive reference), curves.png (the run against itself),
equity_val.png (validation equity against the reference), history.json
and meta.json. PROBE REPORTS rank a harness's walk-forward arms from
its cell files (tune/flat/allocate). The STUDY REPORT renders
the baselines table from the manifests on disk, so a partial re-run
updates its own arms and the report stays consistent with everything
ever computed. Figures draw their look from portfolio.figstyle; study
FIGURES (equity, drawdown, profiles) remain portfolio/plots.py's job.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from portfolio.figstyle import INK, MUTED, RC, style

plt.rcParams.update(RC)

C_BEST, C_FINAL, C_NAIVE = "#4a3aa7", "#e34948", "#2a78d6"
# Labels beyond the three known ones draw from this reserve, in order.
EXTRA_COLORS = ("#008300", "#52514e")


def _style(ax, title):
    style(ax, title, size=10)


# ── run reports ─────────────────────────────────────────────────────────

# The run against itself: episode reward per update, validation Sharpe at
# each evaluation, the kept checkpoint marked.
def curves_png(run_dir: Path, history: dict) -> None:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5.2), sharex=True)
    up = history["update"]
    a1.plot(up, history["ep_reward"], lw=1, color=MUTED, alpha=0.6)
    k = max(1, len(up) // 25)
    smooth = np.convolve(history["ep_reward"], np.ones(k) / k, mode="valid")
    a1.plot(up[k - 1:], smooth, lw=2, color=C_BEST)
    a1.set_ylabel("episode reward")
    _style(a1, "Training reward (smoothed)")

    if history["eval_update"]:
        a2.plot(history["eval_update"], history["eval_sharpe"], lw=2,
                marker="o", ms=4, color=C_BEST)
        sh = np.array(history["eval_sharpe"], dtype=float)
        if np.isfinite(sh).any():          # NaN evals keep no checkpoint
            i = int(np.nanargmax(sh))
            a2.annotate("  kept", (history["eval_update"][i], sh[i]),
                        color=INK, fontsize=8)
    a2.set_ylabel("validation Sharpe")
    a2.set_xlabel("update")
    _style(a2, "Validation Sharpe (best checkpoint kept)")
    fig.savefig(run_dir / "curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# The run against the sleeve's naive baseline on validation: growth of 1.
def equity_png(run_dir: Path, paths: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    known = {"agent (best)": C_BEST, "agent (final)": C_FINAL,
             "1/N in sleeve": C_NAIVE}
    extra = 0
    for name, v in paths.items():
        c = known.get(name)
        if c is None:                      # an unknown label still renders
            c = EXTRA_COLORS[extra % len(EXTRA_COLORS)]
            extra += 1
        ax.plot(v.index, v.values, lw=2, color=c, label=name)
        ax.annotate(f" {v.iloc[-1]:.2f}x", (v.index[-1], v.iloc[-1]),
                    color=c, fontsize=8, va="center")
    ax.set_ylabel("growth of 1 (validation)")
    ax.margins(x=0.08)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax, "Validation equity — agent vs naive")
    fig.savefig(run_dir / "equity_val.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _row(name: str, s: dict, train_sharpe: float | None) -> str:
    ts = f"{train_sharpe:+.3f}" if train_sharpe is not None else "—"
    return (f"| {name} | {ts} | {s['sharpe']:+.3f} "
            f"| {s['annual_return'] * 100:+.2f}% "
            f"| {s['volatility'] * 100:.2f}% "
            f"| {s['max_drawdown'] * 100:.1f}% "
            f"| {s['turnover_annual'] * 100:.0f}% |")


# The written record: configuration, results against naive, the training
# gauges, and what the folder contains.
def report_md(run_dir: Path, meta: dict, results: dict, history: dict) -> None:
    L = [f"# {meta['sleeve']} agent — run {run_dir.name}", ""]
    L += [f"seed {meta['seed']}; network {meta['network']}; "
          f"env {meta['env']}; {len(meta['features'])} features.", ""]
    L += ["| policy | train Sharpe | val Sharpe | val ann | val vol "
          "| val maxDD | val turnover |",
          "|---|---|---|---|---|---|---|"]
    for name, (summary, train_sharpe) in results.items():
        L.append(_row(name, summary, train_sharpe))
    L += ["", "Train Sharpe far above validation = memorised, not learned.",
          ""]
    if history["update"]:
        last = slice(max(0, len(history["update"]) - 50), None)
        L += ["Training gauges (last 50 updates): "
              f"KL {np.mean(history['kl'][last]):.4f}, "
              f"clip fraction {np.mean(history['clipfrac'][last]):.2f}, "
              f"explained variance {np.mean(history['ev'][last]):.2f}. "
              "Healthy: KL ~0.01-0.03, clip ~0.1-0.2, EV toward 1.", ""]
    L += ["Files: best.pt / final.pt (checkpoints), curves.png (the run "
          "against itself), equity_val.png (against naive), history.json "
          "(raw series), meta.json (full configuration).", ""]
    (run_dir / "report.md").write_text("\n".join(L), encoding="utf-8")


def write_all(run_dir: Path, meta: dict, results: dict, history: dict,
              paths: dict) -> None:
    (run_dir / "history.json").write_text(json.dumps(history),
                                          encoding="utf-8")
    (run_dir / "meta.json").write_text(json.dumps(meta, default=str,
                                                  indent=2),
                                       encoding="utf-8")
    curves_png(run_dir, history)
    if paths:
        equity_png(run_dir, paths)
    report_md(run_dir, meta, results, history)


# ── probe reports ───────────────────────────────────────────────────────

# One walk-forward probe table: cells grouped by their own recorded grid,
# so a later edit of a harness's ARMS list cannot relabel stored results.
def probe_report(probe_dir: Path, title: str, delta_note: str) -> None:
    files = sorted(probe_dir.glob("c*_f*.json")) if probe_dir.exists() else []
    groups: dict = {}
    for p in files:
        c = json.loads(p.read_text(encoding="utf-8"))
        groups.setdefault(json.dumps(c["grid"], sort_keys=True),
                          []).append(c)
    if not groups:
        print(f"no cells in {probe_dir}")
        return
    rows = []
    for cells in groups.values():
        g = cells[0]["grid"]
        label = g.get("name") or json.dumps(g, sort_keys=True)
        deltas = [c["delta"] for c in cells]
        rows.append({"config": label,
                     "mean_delta": float(np.mean(deltas)),
                     "worst_delta": float(np.min(deltas)),
                     "folds_won": sum(d > 0 for d in deltas),
                     "n_folds": len(cells),
                     "mean_turnover": float(np.mean(
                         [c["agent_turnover"] for c in cells])),
                     "per_fold": {str(c["fold_year"]): round(c["delta"], 3)
                                  for c in cells}})
    rows.sort(key=lambda r: -r["mean_delta"])
    L = [f"# {title}", "", delta_note, "",
         "| config | mean delta | worst | folds won | turnover | per fold |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['config']} | {r['mean_delta']:+.3f} "
                 f"| {r['worst_delta']:+.3f} "
                 f"| {r['folds_won']}/{r['n_folds']} "
                 f"| {r['mean_turnover'] * 100:.0f}% | {r['per_fold']} |")
    (probe_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


# ── the study report ────────────────────────────────────────────────────

# Derived from the manifests on disk — every arm and split ever computed,
# not just one invocation's. The caller owns the locations and column
# order, so this module needs nothing from portfolio.run.
def study_report(out_dir: Path, report_path: Path, split_order,
                 summary_cols, arms_order, cost_bp: dict,
                 sensitivity_bp) -> None:
    splits = [s for s in split_order if (out_dir / s).exists()]
    L = ["# Baselines report",
         "",
         "Generated by `python -m portfolio.run` from the manifests in "
         "`data/processed/portfolio_runs/` — never hand-edited.",
         "",
         f"Costs (one-way bp): {cost_bp}; monthly = first trading day "
         f"of the month; independent per-split runs starting in cash; the "
         f"first-trade cost column is the initial buy-in from cash, reported "
         f"so it can be netted out across splits of different length.", ""]
    for split in splits:
        manifests = {}
        order = list(arms_order) + sorted(
            p.stem[:-9] for p in (out_dir / split).glob("*_manifest.json")
            if p.stem[:-9] not in arms_order)
        for name in order:
            p = out_dir / split / f"{name}_manifest.json"
            if p.exists():
                manifests[name] = json.loads(p.read_text())
        if not manifests:
            continue
        L += [f"## {split}", ""]
        head = "| arm | clock | " + " | ".join(
            c.replace("_", " ") for c in summary_cols) + " |"
        L += [head, "|" + "---|" * (len(summary_cols) + 2)]
        for name, man in manifests.items():
            s = man["summary"]
            L.append("| " + " | ".join(
                [name, man["clock"]] + [f"{s[c]:.4f}" for c in summary_cols])
                + " |")
        L.append("")

        grid = [f"{bp:g}" for bp in sensitivity_bp]
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
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text("\n".join(L), encoding="utf-8")
    print(f"report -> {report_path}")
