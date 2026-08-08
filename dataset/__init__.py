"""
dataset — layer 1: build point-in-time tables from the raw collections.

Each table is one module with build() and load(); the processed files live in
data/processed/. Downstream code reads ONLY through dataset.loader.

TABLES is the single, ORDERED list of what exists: the calendar must build
first because every other table aligns to it. build_all runs this list; there
is no second list to drift out of sync.
"""

TABLES = (
    "calendar",                 # the trading-day spine        (5,658 x 1)
    "price_dataset",            # daily panel, 123 tickers     (695,934 x 12)
    "macro_dataset",            # PIT macro, lags baked in     (5,658 x 8)
    "fundamentals_dataset",     # quarterly filings, wide      (~12.6k x 77)
)


# Return the table modules, imported, in build order.
def table_modules():
    import importlib
    return [importlib.import_module(f"dataset.{name}") for name in TABLES]
