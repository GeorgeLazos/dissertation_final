"""
dataset/build_all.py — rebuild every processed table, in dependency order.

Runs TABLES from dataset/__init__.py: calendar first, then the three tables
that align to it. Exit code: 0 only if every build ran AND every table
exists — a stale file surviving a failed run cannot look finished.

    python -m dataset.build_all
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collectors._core import console_utf8
from dataset import table_modules
from dataset._core import run_all

if __name__ == "__main__":
    console_utf8()
    raise SystemExit(run_all(table_modules()))
