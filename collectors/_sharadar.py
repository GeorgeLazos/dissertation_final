"""
collectors/_sharadar.py — shared access to the Sharadar API.

All three Sharadar adapters hit the same host with the same key, the same
paging cursor and the same ticker spellings, so that lives here once. An
adapter supplies only its endpoint and its own parameters.

Not a collector: fetches nothing on its own, writes nothing to disk.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("SHARADAR_KEY")

BASE = "https://api.sharadar.com/v1.0/data"
PAGE = 10000                        # API maximum rows per request
TIMEOUT = 90

# Our spelling -> Sharadar's. Applies to every Sharadar endpoint.
ALIAS = {"BRK-B": "BRK.B"}

# Every row for one ticker from one endpoint, following the paging cursor.
# Returns {"data": rows}; sort_key, if given, orders them before returning.
def get(endpoint: str, ticker: str, sort_key=None, **params) -> dict:
    sym = ALIAS.get(ticker, ticker)
    rows, skip = [], 0
    while True:
        r = requests.get(
            f"{BASE}/{endpoint}",
            params={"api_key": KEY, "ticker": sym, "format": "json",
                    "limit": PAGE, "skip": skip, **params},
            timeout=TIMEOUT,
        )
        if not r.ok:
            raise RuntimeError(f"sharadar {endpoint} {r.status_code}: {r.text[:150]}")
        page = r.json().get("data", [])
        rows.extend(page)
        if len(page) < PAGE:
            if sort_key is not None:
                rows.sort(key=sort_key)
            return {"data": rows}
        skip += PAGE
