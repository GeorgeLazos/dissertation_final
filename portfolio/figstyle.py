"""
portfolio/figstyle.py — one look for every figure the study renders.

The palette, the matplotlib defaults and the axis style live here once;
portfolio/plots.py (study figures) and portfolio/report.py (run
reports) both draw with them, so the two figure families cannot drift.
"""
from __future__ import annotations

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_C = "#c3c2b7"

RC = {
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
}


# Left-aligned title, optional subtitle above the axes, top/right spines off.
def style(ax, title: str, sub: str | None = None, size: int = 11) -> None:
    ax.set_title(title, loc="left", fontsize=size, color=INK,
                 pad=22 if sub else 10)
    if sub:
        ax.text(0.0, 1.04, sub, transform=ax.transAxes,
                fontsize=8, color=MUTED, va="bottom")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
