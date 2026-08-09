"""
features — layer 2: computed signals from the layer-1 tables.

Two tables, one builder module each; the registry declares every column and
the builders implement exactly that list. Downstream code reads ONLY through
features.loader. Inputs come ONLY through dataset.loader — no feature touches
raw files or Parquet directly.

The two tables are independent — neither reads the other — so the order in
TABLES is convention, not a dependency.
"""

TABLES = (
    "market_features",          # per-day state              (5,658 x 19)
    "asset_features",           # per-(date,ticker) signals  (695,934 x 59)
)


# Return the table modules, imported, in build order.
def table_modules():
    import importlib
    return [importlib.import_module(f"features.{name}") for name in TABLES]
