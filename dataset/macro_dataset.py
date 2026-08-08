"""
dataset/macro_dataset.py — one row per trading day, publication lags baked in.

A value in row t was publicly known by the 4:00pm ET close of
day t. Lags are applied HERE, once — downstream code must never shift a macro
column again.

    dtb3, vixcls    lag 1 trading day (published ~4:15pm ET, after the close)
    dff             lag 2 (published the next FED business day; a 1-day lag
                    leaks on holidays where NYSE trades but the Fed is shut)
    cpiaucsl, unrate, gdp, gdpc1
                    keyed on the ALFRED release date (8:30am ET, public by
                    that day's close) and carried forward to the next
                    release; the headline is always the NEWEST period —
                    revisions of older periods never overwrite it

NaN before a series' first in-window release — never back-filled.

INPUT
    calendar.load()       (5,658 x 1) 'date'
    fred_macro.load(id)   list of observation dicts, VALUES AS STRINGS,
                          '.' marking unpublished:
                          daily series   {date, value}, one vintage
                          revised series {date, realtime_start, value},
                                         one row per (period, release)

OUTPUT  data/processed/macro_dataset.parquet   (5,658 x 8)
    date + dtb3 dff vixcls cpiaucsl unrate gdp gdpc1   float

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
DAILY_LAGS = {"DTB3": 1, "VIXCLS": 1, "DFF": 2}

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
    return out


# The stored macro table: one row per calendar day, lags already applied.
def load() -> pd.DataFrame:
    return load_table(NAME)


if __name__ == "__main__":
    cli(NAME, build, "Build the point-in-time macro table.")
