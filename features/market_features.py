"""
features/market_features.py — the per-day market table: one row per trading
day, no ticker. State descriptors the policy conditions on, never return
forecasts.

Columns are exactly registry.names("market"), in registry order. Macro
columns arrive from layer 1 with publication lags already applied and are
NEVER re-shifted here; price-derived columns use returns through the close of
day t, which are knowable at t.

INPUT
    dataset.loader.macro()    (5,658 x 14) lags baked in
    dataset.loader.prices()   (695,934 x 12) the daily panel

OUTPUT  data/processed/market_features.parquet   (5,658 x 19)
    date + the 18 registry market columns, float64

    python -m features.market_features            # build
    python -m features.market_features --show     # print the stored table
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from config.tickers import equities
from dataset import loader
from dataset._core import cli, load_table
from features import registry

NAME = "market_features"

# Cumulative simple return over the trailing w sessions, NaN until warm.
def _cumulative(ret: pd.Series, w: int) -> pd.Series:
    return (1 + ret).rolling(w, min_periods=w).apply(np.prod, raw=True) - 1

# One instrument's return series aligned to the calendar index.
def _ret_of(p: pd.DataFrame, ticker: str, cal: pd.DatetimeIndex) -> pd.Series:
    s = p[p["ticker"] == ticker].set_index("date")["ret"]
    return s.reindex(cal)

# The market table: 18 columns on the trading calendar.
def build() -> pd.DataFrame:
    m = loader.macro().set_index("date")
    p = loader.prices()
    cal = m.index

    spy = _ret_of(p, "SPY", cal)
    tlt = _ret_of(p, "TLT", cal)
    shy = _ret_of(p, "SHY", cal)
    lqd = _ret_of(p, "LQD", cal)
    ief = _ret_of(p, "IEF", cal)

    out = pd.DataFrame(index=cal)

    #Fear
    out["vix"] = m["vixcls"]
    out["vix_chg_21"] = m["vixcls"].diff(21)
    # implied variance for the next 21 sessions vs realized over the last 21,
    # horizon-matched (21/252 of an annualized square)
    implied = (m["vixcls"] / 100.0) ** 2 * (21 / 252)
    realized = (spy ** 2).rolling(21, min_periods=21).sum()
    out["vrp_21"] = implied - realized

    #Rates
    out["dtb3"] = m["dtb3"]
    out["dtb3_chg_63"] = m["dtb3"].diff(63)
    out["dff"] = m["dff"]
    out["term_spread"] = m["dgs10"] - m["dtb3"]
    out["def_spread"] = m["dbaa"] - m["daaa"]

    #Macro
    out["cpi_yoy"] = m["cpi_yoy"]
    out["unrate"] = m["unrate"]
    out["unrate_chg_12m"] = m["unrate_chg_12m"]
    out["gdpc1_yoy"] = m["gdpc1_yoy"]

    out["cred_ig_21"] = _cumulative(lqd, 21) - _cumulative(ief, 21)
    out["term_ret_21"] = _cumulative(tlt, 21) - _cumulative(shy, 21)
    out["sb_corr_63"] = spy.rolling(63, min_periods=63).corr(tlt)
    eq = p[p["ticker"].isin(equities()) & p["tradeable"]]
    disp = eq.groupby("date")["ret"].std().reindex(cal)
    out["cs_disp_21"] = disp.rolling(21, min_periods=21).mean()

    out["spy_mom_252"] = _cumulative(spy, 252)
    bear = _cumulative(spy, 504)
    out["spy_bear_504"] = (bear < 0).astype(float).where(bear.notna())

    cols = registry.names("market")
    built = set(out.columns)
    if built != set(cols):
        raise ValueError(
            f"built columns != registry: extra {sorted(built - set(cols))}, "
            f"missing {sorted(set(cols) - built)}")
    return out[cols].astype("float64").rename_axis("date").reset_index()

# The stored market table: one row per trading day.
def load() -> pd.DataFrame:
    return load_table(NAME, deps=("calendar", "price_dataset",
                                  "macro_dataset"), package="features")

if __name__ == "__main__":
    cli(NAME, build, "Build the per-day market feature table.")