"""
collectors/fred_macro.py — macro series via FRED/ALFRED.

Each series in config.macro at native frequency, as levels. The four revised
series (CPI, unemployment, GDP x2) are fetched with their FULL vintage history
(ALFRED ALL_RELEASES): one row per (observation date, release), where
`realtime_start` is the day that value became public — the point-in-time key.
The daily market series (DTB3, DFF, VIXCLS) are never revised; requesting
their vintages would exceed FRED's 2,000-vintage cap and add nothing.

Needs FRED_KEY in .env (free: https://fredaccount.stlouisfed.org/apikeys).

    python -m collectors.fred_macro            # collect; skip jobs already ok
    python -m collectors.fred_macro --force    # re-fetch everything
    python -m collectors.fred_macro --verify   # coverage report, no fetching
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from dotenv import load_dotenv
from collectors._core import cli, collect as _collect, read_stored
from config.macro import REVISED_SERIES, all_series
from config.splits import FETCH_END, FETCH_START

load_dotenv()
FRED_KEY = os.getenv("FRED_KEY")

SOURCE = "fred_macro"
WINDOW = f"{FETCH_START}_{FETCH_END}"
EXT = "json"
JOBS = [(s, WINDOW) for s in all_series()]

URL = "https://api.stlouisfed.org/fred/series/observations"
# ALFRED ALL_RELEASES — the widest real-time range FRED allows.
REALTIME_START = "1776-07-04"
REALTIME_END = "9999-12-31"

# One series bounded to the fetch window; all vintages if it is revised.
def fetch_one(series_id: str, window: str) -> dict:
    params = {
        "series_id": series_id, "api_key": FRED_KEY, "file_type": "json",
        "observation_start": FETCH_START, "observation_end": FETCH_END,
    }
    if series_id in REVISED_SERIES:
        params["realtime_start"] = REALTIME_START
        params["realtime_end"] = REALTIME_END
    r = requests.get(URL, params=params, timeout=60)
    if not r.ok:
        try:
            detail = r.json().get("error_message", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"FRED {r.status_code}: {detail[:150]}")   # keeps the key out of errors
    return r.json()

# Stored observations for one series as a list of dicts, [] if never collected.
def load(series_id: str) -> list:
    stored = read_stored(SOURCE, series_id, WINDOW, EXT)
    return stored["observations"] if stored else []

# Collect every series in JOBS. Returns the run tally: ok/empty/error/skip.
def collect(force: bool = False) -> dict:
    if not FRED_KEY:
        raise SystemExit("FRED_KEY not set in .env")
    return _collect(SOURCE, JOBS, fetch_one, EXT, sleep=0.5, force=force)

# Run the CLI for collecting FRED/ALFRED macro series.
if __name__ == "__main__":
    cli(SOURCE, JOBS, collect, "Collect macro series from FRED/ALFRED.")
