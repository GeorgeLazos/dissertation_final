"""
collectors/digest.py — ledger health readout.

Queries ledger.db only (never the raw files, never any API) and prints a
one-glance summary of the whole collection layer, so silent failures surface.

    python -m collectors.digest
"""

from __future__ import annotations
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collectors._core import DEFAULT_DB, console_utf8

# Looks at the ledger.db and prints a summary of the fetches, including counts of ok, empty, and error statuses,
#  as well as the number of records and the newest fetch time for each source. It also lists any problems encountered during fetching.
def digest() -> None:
    conn = sqlite3.connect(str(DEFAULT_DB))
    try:
        total = conn.execute("SELECT count(*) FROM fetches").fetchone()[0]
        print(f"ledger digest — {total} rows\n")

        agg = defaultdict(lambda: {"ok": 0, "empty": 0, "error": 0, "records": 0, "newest": ""})
        for src, status, n, recs, newest in conn.execute(
            "SELECT source, status, count(*), sum(n_records), max(fetched_at) "
            "FROM fetches GROUP BY source, status"
        ):
            agg[src][status] = n
            agg[src]["records"] += recs or 0
            if newest and newest > agg[src]["newest"]:
                agg[src]["newest"] = newest

        print(f"  {'source':24s} {'ok':>5s} {'empty':>6s} {'error':>6s} {'records':>12s}   newest (UTC)")
        for src in sorted(agg):
            a = agg[src]
            print(f"  {src:24s} {a['ok']:5d} {a['empty']:6d} {a['error']:6d} "
                  f"{a['records']:12,}   {a['newest'][:19]}")

        problems = conn.execute(
            "SELECT source, symbol, window, status, error FROM fetches "
            "WHERE status IN ('error', 'empty') ORDER BY source, symbol"
        ).fetchall()
        print(f"\nproblems (error/empty): {len(problems)}" + ("" if problems else "  — none"))
        for src, sym, win, st, err in problems[:15]:
            print(f"    {st:5s} {src:22s} {sym:8s} {win:12s} {(err or '')[:48]}")
    finally:
        conn.close()


if __name__ == "__main__":
    console_utf8()
    digest()
