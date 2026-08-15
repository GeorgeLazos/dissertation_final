"""
portfolio/models/scaling.py — per-feature scaling fitted on train only.

An agent's observations are scaled with statistics measured on the TRAINING
years of its own feature list, then applied unchanged to every split —
using later years would leak their distribution into training. Nothing
scaled is ever stored: fit() writes a small self-describing JSON (the
per-feature mean and spread plus provenance), and apply() uses it on raw
values at observation time.

Asset-grain features pool across the agent's assets; market-grain features
are one row per date. NaNs are excluded from fitting and handled at apply
time: scaled values carry NaN -> 0 AFTER scaling (zero = the average), and
a parallel 0/1 was-blank flag array preserves the information.

    fit(name, features, assets)   -> stats dict, written to disk
    load(name)                    -> stats dict
    vectors(stats, features)      -> (mu, sd) aligned to the feature order
    apply(raw, mu, sd)            -> (scaled, flags), both float32
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from config import feature_registry as registry
from config.splits import get_split
from features import loader as fl

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "scaling"


# Fit per-feature mean and spread on a training window of this feature
# list, pooled over the given assets for asset-grain features. The window
# defaults to the train split; a walk-forward fold passes its own years.
# Any window must stay inside the train split — statistics fitted on later
# years leak their distribution into training. Writes and returns the
# stats dict.
def fit(name: str, features: list, assets: list,
        window: tuple | None = None) -> dict:
    a_names = [n for n in features if registry.spec(n)["grain"] == "asset"]
    m_names = [n for n in features if registry.spec(n)["grain"] == "market"]
    start, end = window if window is not None else get_split("train")
    t0, t1 = get_split("train")
    if pd.Timestamp(start) < pd.Timestamp(t0) \
            or pd.Timestamp(end) > pd.Timestamp(t1):
        raise ValueError(f"scaling window {start}..{end} leaves the train "
                         f"split {t0}..{t1}")

    #Build a dict of per-feature stats, keyed by feature name.
    per_feature = {}
    if a_names:
        af = fl.features_asset(None, a_names)
        af = af[(af["date"] >= start) & (af["date"] <= end)]
        af = af[af["ticker"].isin(assets)]
        for n in a_names:
            x = af[n].to_numpy(dtype=float)
            per_feature[n] = _stats(x)
    if m_names:
        mf = fl.features_market(None, m_names)
        mf = mf[(mf["date"] >= start) & (mf["date"] <= end)]
        for n in m_names:
            per_feature[n] = _stats(mf[n].to_numpy(dtype=float))

    stats = {
        "name": name,
        "features": list(features),
        "assets": list(assets),
        "window": [str(start), str(end)],
        "per_feature": per_feature,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}.json").write_text(json.dumps(stats, indent=2),
                                          encoding="utf-8")
    return stats

# A constant column has no spread to divide by; it scales to zero and is
# marked so the fact is visible in the file.
def _stats(x: np.ndarray) -> dict:
    ok = x[~np.isnan(x)]
    if len(ok) < 2:
        return {"mean": 0.0, "std": 1.0, "count": int(len(ok)),
                "degenerate": True}
    std = float(np.std(ok, ddof=1))
    return {"mean": float(np.mean(ok)), "std": std if std > 0 else 1.0,
            "count": int(len(ok)), "degenerate": bool(std == 0)}

# Load a stats dict from disk, given its name.
def load(name: str) -> dict:
    return json.loads((OUT_DIR / f"{name}.json").read_text(encoding="utf-8"))


# Vectors of means/spreads aligned to a feature order, for fast apply.
def vectors(stats: dict, features: list) -> tuple:
    missing = [n for n in features if n not in stats["per_feature"]]
    if missing:
        raise ValueError(f"scaling '{stats['name']}' has no stats for "
                         f"{missing[:5]}")
    mu = np.array([stats["per_feature"][n]["mean"] for n in features])
    sd = np.array([stats["per_feature"][n]["std"] for n in features])
    return mu, sd


# raw: ndarray whose LAST axis runs over `features` in order. Returns the
# scaled values (NaN -> 0 after scaling) and the was-blank flags.
def apply(raw: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> tuple:
    flags = np.isnan(raw)
    scaled = (raw - mu) / sd
    scaled[flags] = 0.0
    return scaled.astype(np.float32), flags.astype(np.float32)
