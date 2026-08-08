"""
dataset/build_all.py — rebuild every processed table, in dependency order.

Runs TABLES from dataset/__init__.py: calendar first, then the three tables
that align to it. The EXIT CODE is completeness: 0 only if every table exists
on disk afterwards, so a run that died halfway cannot look finished.

    python -m dataset.build_all
"""

from __future__ import annotations
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collectors._core import console_utf8
from dataset import table_modules
from dataset._core import table_path, write_table


# Build and write each table in order; a failure is reported and the rest
# still run. Returns 1 if any table is missing at the end.
def run_all() -> int:
    modules = table_modules()
    for m in modules:
        print(f"\n{'=' * 64}\n>>> {m.NAME}\n{'=' * 64}")
        try:
            write_table(m.NAME, m.build())
        except Exception:
            print(f"!!! {m.NAME} FAILED - continuing with the rest:")
            traceback.print_exc()

    missing = [m.NAME for m in modules if not table_path(m.NAME).exists()]
    print(f"\ncompleteness: " + ("every table built"
          if not missing else f"MISSING - {missing}"))
    return 1 if missing else 0


if __name__ == "__main__":
    console_utf8()
    raise SystemExit(run_all())
