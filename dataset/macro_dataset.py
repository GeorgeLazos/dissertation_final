"""
dataset/macro_dataset.py — one row per trading day, publication lags baked in.

A value in row t was publicly known by the 4:00pm ET close of
day t. Lags are applied HERE, once — downstream code must never shift a macro
column again.

    dtb3, vixcls, dgs10, dbaa, daaa
                    lag 1 trading day (published ~4:15pm ET, after the close)
    dff             lag 2 (published the next FED business day; a 1-day lag
                    leaks on holidays where NYSE trades but the Fed is shut)
    cpiaucsl, unrate, gdp, gdpc1
                    keyed on the ALFRED release date (8:30am ET, public by
                    that day's close) and carried forward to the next
                    release; the headline is always the NEWEST period —
                    revisions of older periods never overwrite it

Three DERIVED columns (cpi_yoy, unrate_chg_12m, gdpc1_yoy) are computed here
and not downstream, because a year-over-year change must compare two values
FROM THE SAME VINTAGE: differencing the served daily series across a BEA
base-year re-expression reads the level shift as growth. Only this module
sees the vintage interior, so the transform lives here.

NaN before a series' first in-window release — never back-filled.

INPUT
    calendar.load()       (5,658 x 1) 'date'
    fred_macro.load(id)   list of observation dicts, VALUES AS STRINGS,
                          '.' marking unpublished:
                          daily series   {date, value}, one vintage
                          revised series {date, realtime_start, realtime_end,
                                         value}, one row per (period, release),
                                         periods reaching 13 months before the
                                         window so every vintage holds its own
                                         year-ago figure

OUTPUT  data/processed/macro_dataset.parquet   (5,658 x 14)
    date + dtb3 dff vixcls dgs10 dbaa daaa cpiaucsl unrate gdp gdpc1
         + cpi_yoy unrate_chg_12m gdpc1_yoy                          float

    python -m dataset.macro_dataset            # build
    python -m dataset.macro_dataset --show     # print the stored table
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from collectors import fred_macro
from config.macro import REVISED_SERIES, all_series
from dataset import calendar
from dataset._core import cli, load_table

NAME = "macro_dataset"

# Trading-day publication lags for the single-vintage daily series.
DAILY_LAGS = {"DTB3": 1, "VIXCLS": 1, "DFF": 2,
              "DGS10": 1, "DBAA": 1, "DAAA": 1}

# Derived point-in-time transforms: column -> (series, months back, diff).
# diff=True serves a difference, otherwise a ratio minus one.
DERIVED = {
    "cpi_yoy": ("CPIAUCSL", 12, False),
    "unrate_chg_12m": ("UNRATE", 12, True),
    "gdpc1_yoy": ("GDPC1", 12, False),
}

# One daily series as a (date, value) frame, '.' holiday markers dropped,
# string values made float.
def _daily(series_id: str) -> pd.DataFrame:
    rows = [(o["date"], o["value"]) for o in fred_macro.load(series_id)
            if o["value"] != "."]
    df = pd.DataFrame(rows, columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[us]")
    df["value"] = df["value"].astype(float)
    return df.sort_values("date")

# One revised series as a (release, value) event stream: what the headline
# figure BECAME on each release day. Only a release carrying the newest
# period updates it, so a revision of an old period never overwrites.
def _releases(series_id: str) -> pd.DataFrame:
    rows = [(o["realtime_start"], o["date"], o["value"])
            for o in fred_macro.load(series_id) if o["value"] != "."]
    df = pd.DataFrame(rows, columns=["release", "period", "value"])
    df["release"] = pd.to_datetime(df["release"]).astype("datetime64[us]")
    df["period"] = pd.to_datetime(df["period"])
    df["value"] = df["value"].astype(float)
    df = df.sort_values(["release", "period"])

    events, newest = [], pd.Timestamp.min
    for r in df.itertuples():
        if r.period >= newest:
            newest = r.period
            events.append((r.release, r.value))
    ev = pd.DataFrame(events, columns=["release", "value"])
    return ev.groupby("release", as_index=False).last()

# Headline events for a derived transform: at each accepted release, the new
# figure against the SAME vintage's own year-ago figure. The prior value is
# the row whose realtime interval covers the release day, so both legs sit on
# one base and a re-expression of the whole series cancels out.
def _derived_events(series_id: str, months: int, diff: bool) -> pd.DataFrame:
    rows = [(o["realtime_start"], o["realtime_end"], o["date"], o["value"])
            for o in fred_macro.load(series_id) if o["value"] != "."]
    df = pd.DataFrame(rows, columns=["release", "until", "period", "value"])
    for c in ("release", "until", "period"):
        df[c] = pd.to_datetime(df[c])
    df["value"] = df["value"].astype(float)

    by_period = {p: list(g[["release", "until", "value"]].itertuples(index=False))
                 for p, g in df.groupby("period")}

    def _as_of(period: pd.Timestamp, day: pd.Timestamp):
        for rs, until, v in by_period.get(period, ()):
            if rs <= day <= until:
                return v
        return None

    events, newest = [], pd.Timestamp.min
    for r in df.sort_values(["release", "period"]).itertuples():
        if r.period >= newest:
            newest = r.period
            prior = _as_of(r.period - pd.DateOffset(months=months), r.release)
            if prior is None:
                val = float("nan")
            else:
                val = r.value - prior if diff else r.value / prior - 1
            events.append((r.release.as_unit("us"), val))
    ev = pd.DataFrame(events, columns=["release", "value"])
    return ev.groupby("release", as_index=False).last()


# The macro table: every series aligned to the calendar at its own lag.
def build() -> pd.DataFrame:
    cal = calendar.load()
    out = cal.copy()
    for sid in all_series():
        col = sid.lower()
        if sid in REVISED_SERIES:
            ev = _releases(sid).rename(columns={"release": "date"})
            out[col] = pd.merge_asof(cal, ev, on="date",
                                     direction="backward")["value"]
        else:
            s = _daily(sid)
            aligned = pd.merge_asof(cal, s, on="date",
                                    direction="backward")["value"]
            out[col] = aligned.shift(DAILY_LAGS[sid])
    for col, (sid, months, diff) in DERIVED.items():
        ev = _derived_events(sid, months, diff).rename(columns={"release": "date"})
        out[col] = pd.merge_asof(cal, ev, on="date",
                                 direction="backward")["value"]
    return out


# The stored macro table: one row per calendar day, lags already applied.
def load() -> pd.DataFrame:
    return load_table(NAME)


if __name__ == "__main__":
    cli(NAME, build, "Build the point-in-time macro table.")
