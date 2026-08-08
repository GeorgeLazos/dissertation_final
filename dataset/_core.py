"""
dataset/_core.py — the shared plumbing of the build layer.

Every dataset file computes a different table but stores and reports it
IDENTICALLY through this module: one Parquet file per table in
data/processed/, written atomically, read back through load_table. A dataset
file declares its NAME and one build() that returns the finished DataFrame;
the CLI and the writing live here once.

Tables are written index-free — keys (date, ticker) are ordinary columns.

Run `python -m dataset._core` for an offline round-trip test.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from collectors._core import console_utf8

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
LEDGER = ROOT / "ledger.db"

# Where a processed table lives: data/processed/{name}.parquet.
def table_path(name: str) -> Path:
    return PROCESSED / f"{name}.parquet"

# Write one table atomically and report its shape. A crash mid-write leaves
# the previous file intact, never a half-written one.
def write_table(name: str, df: pd.DataFrame) -> Path:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = table_path(name)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, engine="pyarrow", index=False, compression="snappy")
    os.replace(tmp, path)
    print(f"{name}: {len(df):,} rows x {len(df.columns)} cols  -> {path.relative_to(ROOT)}")
    return path

# Read one table back. Fails with the rebuild command if it was never built,
# and warns when the table predates the raw data (ledger.db) it came from.
def load_table(name: str) -> pd.DataFrame:
    path = table_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} is not built - run: python -m dataset.{name}")
    if LEDGER.exists() and path.stat().st_mtime < LEDGER.stat().st_mtime:
        print(f"  ! {path.name} is older than the raw data - "
              f"rebuild: python -m dataset.{name}", file=sys.stderr)
    return pd.read_parquet(path)

# The one command-line entry point, shared by every dataset file: running it
# builds and writes the table; --show prints the stored one without rebuilding.
# Exits 0 only if the table exists afterwards.
def cli(name: str, build_fn, description: str) -> None:
    console_utf8()
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--show", action="store_true",
                    help="print the stored table, no rebuild")
    args = ap.parse_args()
    if args.show:
        df = load_table(name)
        print(df.dtypes.to_string())
        print()
        print(df)
    else:
        write_table(name, build_fn())
    raise SystemExit(0 if table_path(name).exists() else 1)


# ── offline round-trip test: python -m dataset._core ─────────────────────
if __name__ == "__main__":
    console_utf8()
    print("build-layer plumbing test (no network)\n")

    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
                       "ticker": ["AAPL", "AAPL"], "x": [1.0, 2.0]})
    write_table("_selftest", df)
    back = load_table("_selftest")
    print(f"  round-trip identical : {back.equals(df)}")
    print(f"  dtypes preserved     : {dict(back.dtypes.astype(str))}")

    table_path("_selftest").unlink()
    print("\n  cleaned up. plumbing OK.")
