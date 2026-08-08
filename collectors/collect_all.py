"""
collectors/collect_all.py — run every CORE source in sequence, then digest.

Each adapter is idempotent (skips jobs already ok, retries errors), so this
fills gaps and retries failures in one command. A failing source is reported
and the rest still run — but the EXIT CODE is completeness: 0 only if every
job of every source has an 'ok' ledger row, so a run that did not finish
cannot look finished.

    python -m collectors.collect_all            # collect anything missing
    python -m collectors.collect_all --force    # re-fetch everything
"""

from __future__ import annotations
import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collectors import core_modules
from collectors._core import complete, console_utf8
from collectors.digest import digest

# Function to run all core collectors in sequence, then print the digest and completeness report.
def run_all(force: bool = False) -> int:
    modules = core_modules()
    for m in modules:
        print(f"\n{'=' * 64}\n>>> {m.SOURCE}\n{'=' * 64}")
        try:
            m.collect(force=force)
        except Exception:
            print(f"!!! {m.SOURCE} FAILED — continuing with the rest:")
            traceback.print_exc()

    print(f"\n{'=' * 64}\n>>> DIGEST\n{'=' * 64}")
    digest()

    short = [m.SOURCE for m in modules if not complete(m.SOURCE, m.JOBS)]
    print(f"\ncompleteness: " + ("every source complete"
          if not short else f"INCOMPLETE — {short}"))
    return 1 if short else 0

# Runs the run_all function when the script is executed directly
if __name__ == "__main__":
    console_utf8()
    ap = argparse.ArgumentParser(description="Run all collectors, then print the digest.")
    ap.add_argument("--force", action="store_true", help="re-fetch every job")
    args = ap.parse_args()
    raise SystemExit(run_all(force=args.force))
