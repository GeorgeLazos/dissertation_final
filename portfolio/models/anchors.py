"""
portfolio/models/anchors.py — closed-form anchor weight paths for
anchored-tilt agents.

An anchor is a baseline strategy's monthly weight rows over the full
panel, cached under data/processed/. A tilt agent's executed weights
pivot multiplicatively around the anchor's live row, so the anchor
must use only trailing data — the baselines' own estimation
discipline already guarantees that.

The artifact builds itself on first use and rebuilds when the panel
underneath it is newer, so training and evaluation need no separate
step; --build forces a rebuild.

    python -m portfolio.models.anchors --build markowitz
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from config import portfolio as cfg
from portfolio import baselines

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"

# Anchor names resolve through the baseline registry, so an anchor is
# always one of the study's own graded strategies.
ANCHORS = tuple(baselines.ARMS)


def path(name: str) -> Path:
    return PROCESSED / f"anchor_{name}.parquet"


# The anchor's monthly weight rows from the agents' first training
# session through the end of the panel, one row per rebalance.
def build(name: str) -> pd.DataFrame:
    if name not in ANCHORS:
        raise ValueError(f"unknown anchor {name!r}; have {ANCHORS}")
    from portfolio.run import load_bundle
    b = load_bundle()
    dates = b["dates"]
    window = dates[dates >= cfg.AGENT_TRAIN_START]
    reb = cfg.month_starts(window)
    W, info = baselines.ARMS[name]["producer"](b, reb)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    W.to_parquet(path(name))
    print(f"anchor_{name}: {W.shape[0]} rebalances x {W.shape[1]} "
          f"columns  {info}")
    return W


# Ensure the anchor is built and up to date
def ensure(name: str) -> Path:
    p = path(name)
    panel = PROCESSED / "price_dataset.parquet"
    if not p.exists():
        print(f"anchor_{name} is not built - building it now")
        build(name)
    elif panel.exists() and p.stat().st_mtime < panel.stat().st_mtime:
        print(f"anchor_{name} is older than the panel - rebuilding")
        build(name)
    return p


# Load the anchor's monthly weight rows from disk, reindexing to the
# given columns and filling missing columns with zeros.
def load(name: str, columns: list) -> pd.DataFrame:
    A = pd.read_parquet(ensure(name))
    missing = [c for c in columns if c not in A.columns]
    if len(missing) > 1:
        raise ValueError(f"anchor {name} lacks columns {missing[:5]}")
    return A.reindex(columns=columns).fillna(0.0)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, choices=list(ANCHORS))
    build(ap.parse_args().build)
