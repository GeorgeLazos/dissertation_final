"""
collectors/edgar_fundamentals.py — raw XBRL fundamentals from SEC EDGAR.

The independent cross-check for sharadar_fundamentals: every financial fact a
company ever filed, stored raw and unstandardized (~500 us-gaap tags per
company vs Sharadar's 112 fixed fields). Facts carry per-filing dates (`filed`)
and form types, which Sharadar's one availability date per quarter does not.

Companies that changed registrant through a reorg file under MORE THAN ONE
CIK; each CIK is its own job and its own file (window = the CIK), and the
build layer merges them by filing date.

Needs SEC_USER_AGENT in .env ("Full Name email@example.com") — SEC returns 403
without an identifying User-Agent, and fair use is ~10 requests/second.

    python -m collectors.edgar_fundamentals            # collect; skip jobs already ok
    python -m collectors.edgar_fundamentals --force    # re-fetch everything
    python -m collectors.edgar_fundamentals --verify   # coverage report, no fetching
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from dotenv import load_dotenv
from collectors._core import cli, collect as _collect, read_stored
from config.tickers import sharadar_tickers

load_dotenv()
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT")

SOURCE = "edgar_fundamentals"
EXT = "json"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Companies filing under more than one CIK (reorgs). LIN is NOT split: the 2018
# Praxair/Linde merger of equals continues under Linde plc's own registration.
CIK_OVERRIDES = {
    "XOM":   ["0000034088"],                    # one ID pinned — full history
    "BLK":   ["0001364742", "0002012383"],      # BlackRock Inc + 2024 reorg entity
    "GOOGL": ["0001288776", "0001652044"],      # Google Inc + Alphabet
    "DIS":   ["0001001039", "0001744489"],      # old Disney + current
    "MDT":   ["0000064670", "0001613103"],      # Medtronic Inc + plc
}

# One job per (ticker, CIK): overrides expand to one file per CIK; everyone
# else resolves through the SEC ticker map at fetch time under window 'all'.
JOBS = [(t, cik)
        for t in sharadar_tickers() if t in CIK_OVERRIDES
        for cik in CIK_OVERRIDES[t]]
JOBS += [(t, "all") for t in sharadar_tickers() if t not in CIK_OVERRIDES]
JOBS.sort()

_cik_map: dict = {}

# Check that SEC_USER_AGENT is set, and return a headers dict for requests.
def _headers() -> dict:
    if not SEC_USER_AGENT:
        raise SystemExit("SEC_USER_AGENT not set in .env — 'Full Name email@example.com'")
    return {"User-Agent": SEC_USER_AGENT}

# Zero-padded 10-digit CIK from the SEC ticker map, fetched once on the first
# lookup and reused for every one after.
def _resolve(ticker: str) -> str:
    if not _cik_map:
        r = requests.get(TICKERS_URL, headers=_headers(), timeout=60)
        r.raise_for_status()
        _cik_map.update({row["ticker"].upper(): f"{int(row['cik_str']):010d}"
                         for row in r.json().values()})
    cik = _cik_map.get(ticker.upper())
    if not cik:
        raise RuntimeError(f"no CIK for {ticker} in the SEC ticker map")
    return cik

# companyfacts for one (ticker, CIK) job -- every XBRL fact ever filed. An
# override job carries its CIK in the window; everyone else resolves here.
def fetch_one(ticker: str, window: str) -> dict:
    cik = window if window != "all" else _resolve(ticker)
    r = requests.get(FACTS_URL.format(cik=cik), headers=_headers(), timeout=60)
    if not r.ok:
        raise RuntimeError(f"SEC {r.status_code}: {r.text[:150]}")
    return r.json()

# All stored companyfacts blobs for one ticker, [] if never collected. A list
# rather than one blob: a reorg company has one file per CIK.
def load(ticker: str) -> list:
    windows = CIK_OVERRIDES.get(ticker, ["all"])
    return [b for w in windows if (b := read_stored(SOURCE, ticker, w, EXT))]

# Collect every (ticker, CIK) job. Returns the run tally: ok/empty/error/skip.
def collect(force: bool = False) -> dict:
    _headers()                                   # fail before looping, not inside it
    return _collect(SOURCE, JOBS, fetch_one, EXT, sleep=0.15, force=force)

# Run the CLI for collecting SEC EDGAR companyfacts (raw XBRL).
if __name__ == "__main__":
    cli(SOURCE, JOBS, collect, "Collect SEC EDGAR companyfacts (raw XBRL).")
