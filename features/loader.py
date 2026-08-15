"""
features/loader.py — the single read interface: layers 3 and 4 read
features through here.

Every request is a NAME LIST validated against the registry — an unknown or
repeated name is an error, never a silent extra column. Split slicing reuses
config.splits so no model code ever writes a date literal.

    features_asset(split, names)    (date, ticker) + the asset columns
    features_market(split, names)   (date) + the market columns
    observations(names, split)      one frame, market columns broadcast to
                                    every (date, ticker) row, columns in the
                                    requested order

    python -m features.loader --check     # verify the loader
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config.splits import get_split
from config import feature_registry as registry
from features import asset_features, market_features

# Validated split of a request into asset and market names, order kept.
def _resolve(names: list) -> tuple:
    if len(set(names)) != len(names):
        dup = [n for n in names if names.count(n) > 1]
        raise ValueError(f"repeated feature names: {sorted(set(dup))}")
    unknown = [n for n in names if n not in registry.REGISTRY]
    if unknown:
        raise ValueError(f"unknown feature names: {unknown}")
    a = [n for n in names if registry.spec(n)["grain"] == "asset"]
    m = [n for n in names if registry.spec(n)["grain"] == "market"]
    return a, m

# Slice a frame by split, or return the whole frame if split is None.
def _slice(df: pd.DataFrame, split: str | None) -> pd.DataFrame:
    if split is None:
        return df
    start, end = get_split(split)
    return df[(df["date"] >= start) & (df["date"] <= end)]

# The asset table, optionally one split and a subset of columns.
def features_asset(split: str | None = None, names: list | None = None) -> pd.DataFrame:
    a = _slice(asset_features.load(), split)
    if names is not None:
        picked, stray = _resolve(names)[0], _resolve(names)[1]
        if stray:
            raise ValueError(f"market-grain names in an asset request: {stray}")
        a = a[["date", "ticker"] + picked]
    return a.reset_index(drop=True)

# The market table, optionally one split and a subset of columns.
def features_market(split: str | None = None, names: list | None = None) -> pd.DataFrame:
    m = _slice(market_features.load(), split)
    if names is not None:
        stray, picked = _resolve(names)
        if stray:
            raise ValueError(f"asset-grain names in a market request: {stray}")
        m = m[["date"] + picked]
    return m.reset_index(drop=True)

# One observation frame: asset columns per (date, ticker), market columns
# broadcast by date, column order exactly as requested.
def observations(names: list, split: str | None = None) -> pd.DataFrame:
    a_names, m_names = _resolve(names)
    out = _slice(asset_features.load(), split)[["date", "ticker"] + a_names]
    if m_names:
        m = _slice(market_features.load(), split)[["date"] + m_names]
        out = out.merge(m, on="date", how="left")
    return out[["date", "ticker"] + list(names)].reset_index(drop=True)

if __name__ == "__main__":
    from collectors._core import console_utf8
    console_utf8()
    if "--check" in sys.argv:
        bad = []
        a = features_asset("train")
        m = features_market("train")
        print(f"features_asset('train')  : {a.shape}, "
              f"{a['date'].min().date()} -> {a['date'].max().date()}")
        print(f"features_market('train') : {m.shape}")
        obs = observations(["mom_63", "vix", "ep_ttm", "term_spread"], "val")
        print(f"observations(4, 'val')   : {obs.shape}, "
              f"columns {list(obs.columns)}")
        one = obs[obs["date"] == obs["date"].iloc[0]]
        if one["vix"].nunique() != 1 or one["term_spread"].nunique() != 1:
            bad.append("market columns not constant within a date")
        for should_fail in (["mom_63", "mom_63"], ["not_a_feature"]):
            try:
                observations(should_fail, "train")
                bad.append(f"request {should_fail} was not rejected")
            except ValueError:
                pass
        print("broadcast constant within date: "
              f"{'yes' if not bad else 'NO'}; bad requests rejected: yes")
        if bad:
            for b in bad:
                print(f"  {b}")
        raise SystemExit(1 if bad else 0)
    print(f"registry: {len(registry.REGISTRY)} features; use --check")
