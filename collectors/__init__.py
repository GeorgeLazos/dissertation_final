"""
collectors — layer 0: fetch raw data and store it faithfully.

CORE is the single list of sources. collect_all runs it; the build layer reads
the same modules through their load() functions. There is no second list to
drift out of sync.
"""

CORE = (
    "sharadar_prices",          # daily OHLCV, equities + REITs       -> parquet
    "sharadar_actions",         # typed dividends/splits/spin-offs    -> json
    "sharadar_fundamentals",    # SF1 quarterly, ~112 fields          -> json
    "yfinance_prices",          # ETFs primary + equity cross-check   -> parquet
    "fred_macro",               # 7 series, vintages where revised    -> json
    "edgar_fundamentals",       # raw XBRL, the fundamentals check    -> json
)

# Return the list of core modules, imported.
def core_modules():
    import importlib
    return [importlib.import_module(f"collectors.{name}") for name in CORE]
