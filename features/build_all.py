"""
features/build_all.py — rebuild both feature tables.

The two tables are independent; the order is convention, not a dependency.
Exit code: 0 only if every build ran AND every table exists — a stale file
surviving a failed run cannot look finished.

    python -m features.build_all
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collectors._core import console_utf8
from dataset._core import run_all
from features import table_modules

if __name__ == "__main__":
    console_utf8()
    raise SystemExit(run_all(table_modules()))
