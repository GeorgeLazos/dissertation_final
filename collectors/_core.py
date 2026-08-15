"""
collectors/_core.py — the shared storage and ledger core of the collection
layer.

Every adapter fetches differently but stores and reports IDENTICALLY through
this module.

    data/raw/{source}/{symbol}/{window}.{ext}     one file per response
    ledger.db                                     one row per (source, symbol, window)

Run `python -m collectors._core` for an offline round-trip test.
"""

from __future__ import annotations
import argparse
import io
import json
import os
import re
import sqlite3
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DEFAULT_DB = ROOT / "ledger.db"
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]") 
_SCHEMA = """
CREATE TABLE IF NOT EXISTS fetches (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT    NOT NULL,
    symbol      TEXT    NOT NULL,
    window      TEXT    NOT NULL,
    path        TEXT    NOT NULL,
    status      TEXT    NOT NULL,   -- 'ok' | 'empty' | 'error'
    n_records   INTEGER,
    error       TEXT,
    fetched_at  TEXT    NOT NULL
);
CREATE INDEX        IF NOT EXISTS idx_symbol ON fetches(symbol);
CREATE INDEX        IF NOT EXISTS idx_source ON fetches(source);
CREATE UNIQUE INDEX IF NOT EXISTS idx_job    ON fetches(source, symbol, window);
"""

# Switch stdout/stderr to UTF-8. Windows consoles default to cp1252, which
# cannot encode the arrows and dashes the progress lines print.
def console_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError, OSError):
            pass

# Returns the current time as a string
def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

# Filesystem-safe form of a symbol, for the on-disk path only ('^VIX' -> '_VIX').
# The real symbol always goes in the ledger.
def _safe_symbol(symbol: str) -> str:
    return _UNSAFE.sub("_", symbol)

# Where a response goes: data/raw/{source}/{safe_symbol}/{window}.{ext}.
# Creates the folders on the way.
def build_path(source: str, symbol: str, window: str, ext: str) -> Path:
    directory = DATA_RAW / source / _safe_symbol(symbol)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{window}.{ext}"

# Atomically writes text to a file.
def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)

# Atomically writes bytes to a file.
def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# ── ledger ───────────────────────────────────────────────────────────────

# Open a ledger connection: create the table if absent, commit on success,
# always close. Every ledger read and write goes through here.
@contextmanager
def _ledger(db_path: Path = DEFAULT_DB):
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()

# Write a row to the ledger, or update an existing one. 
def _upsert(conn, source, symbol, window, path, status, n_records, error) -> None:
    conn.execute(
        """
        INSERT INTO fetches (source, symbol, window, path, status, n_records, error, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source, symbol, window) DO UPDATE SET
            path = excluded.path, status = excluded.status,
            n_records = excluded.n_records, error = excluded.error,
            fetched_at = excluded.fetched_at
        """,
        (source, symbol, window, str(path), status, n_records, error, _now_utc()),
    )

# True only if this job has an 'ok' row — the skip check. A job that errored or
# came back empty is not done, so it gets retried on the next run.
def already_done(source: str, symbol: str, window: str) -> bool:
    with _ledger() as conn:
        row = conn.execute(
            "SELECT 1 FROM fetches WHERE source=? AND symbol=? AND window=? AND status='ok'",
            (source, symbol, window),
        ).fetchone()
    return row is not None

# Write an 'error' row so the failure surfaces in the digest.
def record_error(source: str, symbol: str, window: str, message: str) -> None:
    nominal = DATA_RAW / source / _safe_symbol(symbol) / window
    with _ledger() as conn:
        _upsert(conn, source, symbol, window, nominal, "error", None, str(message))

# True only if EVERY job has an 'ok' row. A job never attempted has no row at
# all, so this is what turns a partial run into a non-zero exit code.
def complete(source: str, jobs: list) -> bool:
    with _ledger() as conn:
        rows = conn.execute(
            "SELECT symbol, window FROM fetches WHERE source=? AND status='ok'",
            (source,),
        ).fetchall()
    return set(jobs) <= {(s, w) for s, w in rows}


# ── counting, saving, reading ────────────────────────────────────────────

#Counts the number of records in a response, handling different shapes.
def count_records(response) -> int:
    if response is None:
        return 0
    try:
        import pandas as pd
        if isinstance(response, pd.DataFrame):
            return int(len(response))
    except ImportError:
        pass
    if isinstance(response, list):
        return len(response)
    if isinstance(response, dict):
        facts = response.get("facts")                 # EDGAR companyfacts
        if isinstance(facts, dict):
            return sum(len(unit_rows)
                       for taxonomy in facts.values()
                       for concept in taxonomy.values()
                       for unit_rows in concept.get("units", {}).values())
        for key in ("data", "observations"):          # Sharadar, FRED
            val = response.get(key)
            if isinstance(val, list):
                return len(val)
    return 0


# Store one response and index it in the ledger. DataFrames go to Parquet,
# everything else to a JSON envelope {"_fetched_at", "_response"}. Returns the
# record count, which decides 'ok' vs 'empty'.
# The ledger row is written only AFTER the file lands, so a crash leaves a
# re-fetchable job rather than a success record pointing at nothing.
def save(source: str, symbol: str, window: str, response, ext: str) -> int:
    n = count_records(response)
    status = "ok" if n > 0 else "empty"
    path = build_path(source, symbol, window, ext)

    if ext == "parquet":
        import pandas as pd
        out = response
        # Written with index=False, so a DatetimeIndex must become a column
        # first or the dates are silently dropped.
        if not isinstance(out.index, pd.RangeIndex):
            name = out.index.name or "Date"
            out = out.reset_index().rename(columns={"index": name})
        buf = io.BytesIO()
        out.to_parquet(buf, engine="pyarrow", index=False, compression="snappy")
        _atomic_write_bytes(path, buf.getvalue())
    else:
        envelope = {"_fetched_at": _now_utc(), "_response": response}
        _atomic_write_text(path, json.dumps(envelope, ensure_ascii=False, default=str))

    with _ledger() as conn:
        _upsert(conn, source, symbol, window, path, status, n, None)
    return n

# The stored response for one job: a DataFrame for parquet, the unwrapped
# _response for json, None if never collected. Backs every adapter's load(),
# so the build layer never constructs a raw path itself.
def read_stored(source: str, symbol: str, window: str, ext: str):
    path = DATA_RAW / source / _safe_symbol(symbol) / f"{window}.{ext}"
    if not path.exists():
        return None
    if ext == "parquet":
        import pandas as pd
        return pd.read_parquet(path)
    return json.loads(path.read_text(encoding="utf-8"))["_response"]


# ── the collect loop and CLI ─────────────────────────────────────────────

# THE loop, shared by every adapter: for each (symbol, window) job — skip if
# already ok, else fetch it with the adapter's own fetch_one, save it, tally it.
# One job failing is logged and the run carries on. Returns the tally.
def collect(source: str, jobs: list, fetch_one, ext: str,
            sleep: float = 0.3, force: bool = False) -> dict:
    total = len(jobs)
    print(f"{source} — {total} jobs\n")
    counts = {"ok": 0, "empty": 0, "error": 0, "skip": 0}
    varied = len({w for _s, w in jobs}) > 1          # per-CIK windows get labelled

    for i, (symbol, window) in enumerate(jobs, start=1):
        label = f"{symbol} {window}" if varied and window != "all" else symbol
        if not force and already_done(source, symbol, window):
            counts["skip"] += 1
            continue
        try:
            n = save(source, symbol, window, fetch_one(symbol, window), ext)
        except Exception as exc:
            record_error(source, symbol, window, repr(exc))
            counts["error"] += 1
            print(f"  [{i:3d}/{total}] {label:16s} ERROR  {exc}")
            time.sleep(sleep)
            continue
        counts["ok" if n else "empty"] += 1
        print(f"  [{i:3d}/{total}] {label:16s} {'ok' if n else 'EMPTY':5s} {n:8,} records")
        time.sleep(sleep)

    print(f"\nsummary  ok={counts['ok']}  empty={counts['empty']}  "
          f"error={counts['error']}  skipped={counts['skip']}  (of {total})")
    return counts

# Report what one source collected against what it was asked for: status counts,
# record total, and the jobs still missing by name. Prints only, checks nothing.
def verify(source: str, jobs: list) -> None:
    with _ledger() as conn:
        rows = conn.execute(
            "SELECT symbol, window, status, n_records FROM fetches WHERE source=?",
            (source,),
        ).fetchall()
    by_status = {}
    for _s, _w, st, _n in rows:
        by_status[st] = by_status.get(st, 0) + 1
    ok = {(s, w) for s, w, st, _n in rows if st == "ok"}
    missing = [j for j in jobs if j not in ok]
    total_records = sum(n or 0 for _s, _w, st, n in rows if st == "ok")

    print(f"{source} — coverage")
    print(f"  jobs expected : {len(jobs)}")
    print(f"  ledger        : " + "  ".join(f"{k}={v}" for k, v in sorted(by_status.items())))
    print(f"  records total : {total_records:,}")
    print(f"  missing       : {[f'{s} {w}' for s, w in missing] or 'none'}")

# The one command-line entry point, shared by every adapter: --verify reports,
# no flag collects, --force re-fetches. Exits 0 only if the source is complete,
# so a partial run cannot look finished to a script reading exit codes.
def cli(source: str, jobs: list, collect_fn, description: str) -> None:
    console_utf8()
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--force", action="store_true", help="re-fetch even jobs already ok")
    ap.add_argument("--verify", action="store_true", help="coverage report, no fetching")
    args = ap.parse_args()
    if args.verify:
        verify(source, jobs)
    else:
        collect_fn(force=args.force)
    raise SystemExit(0 if complete(source, jobs) else 1)

# ── offline round-trip test: python -m collectors._core ──────────────────
if __name__ == "__main__":
    import shutil
    import pandas as pd

    console_utf8()
    src = "_selftest"
    print("spine round-trip test (no network)\n")

    # JSON shape (Sharadar-like) through the full loop
    fake = {"data": [{"date": "2020-01-02", "value": 1.0}, {"date": "2020-01-03", "value": 2.0}]}
    collect(src, [("FAKE", "all")], lambda s, w: fake, ext="json", sleep=0)
    back = read_stored(src, "FAKE", "all", "json")
    print(f"  json round-trip : {back == fake}")

    # DataFrame shape with a DatetimeIndex through the full loop
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0], "Volume": [100, 200, 300]}, index=idx)
    df.index.name = "Date"
    collect(src, [("FAKEDF", "w")], lambda s, w: df, ext="parquet", sleep=0)
    bdf = read_stored(src, "FAKEDF", "w", "parquet")
    print(f"  parquet columns : {list(bdf.columns)}   volume dtype {bdf['Volume'].dtype}")

    # empty and error classification + completeness
    collect(src, [("EMPTY", "all")], lambda s, w: {"data": []}, ext="json", sleep=0)
    collect(src, [("BROKEN", "all")], lambda s, w: 1 / 0, ext="json", sleep=0)
    print(f"  skip logic      : {already_done(src, 'FAKE', 'all')}")
    print(f"  complete(ok)    : {complete(src, [('FAKE', 'all'), ('FAKEDF', 'w')])}")
    print(f"  complete(err)   : {complete(src, [('FAKE', 'all'), ('BROKEN', 'all')])} (must be False)")

    with _ledger() as conn:
        conn.execute("DELETE FROM fetches WHERE source=?", (src,))
    shutil.rmtree(DATA_RAW / src, ignore_errors=True)
    print("\n  cleaned up. spine OK.")
