"""
macroeconomic series (FRED / ALFRED).

Single source of truth for the macro indicators the model uses as context.

These are NOT tradeable assets. They are economic context fed to the agents as raw external inputs.

The four REVISED series are collected with their full vintage history (ALFRED
ALL_RELEASES), so the model can use the value PUBLICLY KNOWN on each date rather
than a figure revised years later. The three daily market series are never
revised and are collected as a single series each — asking for vintages on a
daily series exceeds FRED's 2,000-vintage cap and would add nothing.
"""

# ---------------------------------------------------------------------------
# THE SERIES
# ---------------------------------------------------------------------------
# id -> human description. 
# The collector fetches each at its NATIVE frequency and as LEVELS 
# Collect raw, transform downstream.

FRED_SERIES = {
    # --- daily, market-determined, effectively never revised ---------------
    "DTB3":     "3-month Treasury bill, secondary market rate (daily, %)",
    "DFF":      "Federal funds effective rate (daily, %)",
    "VIXCLS":   "CBOE Volatility Index, VIX close (daily) — also via yfinance ^VIX",
    "DGS10":    "10-year Treasury constant maturity yield (daily, %) — the long "
                "leg of the term spread",
    "DBAA":     "Moody's seasoned Baa corporate bond yield (daily, %) — the "
                "risky leg of the default spread",
    "DAAA":     "Moody's seasoned Aaa corporate bond yield (daily, %) — the "
                "safe leg of the default spread",

    # --- monthly, revised ---------------------------------------------------
    "CPIAUCSL": "CPI, all urban consumers, all items, seasonally adj. (monthly, index)",
    "UNRATE":   "Unemployment rate (monthly, %)",

    # --- quarterly, revised --------------------------------------
    "GDP":      "Gross domestic product, NOMINAL (quarterly, $bn, SAAR)",
    "GDPC1":    "Real gross domestic product (quarterly, chained $bn, SAAR)",
}

# ---------------------------------------------------------------------------
# WHICH SERIES NEED ALFRED VINTAGES
# ---------------------------------------------------------------------------
# Only revised series need the full vintage history
REVISED_SERIES = {"CPIAUCSL", "UNRATE", "GDP", "GDPC1"}

# ---------------------------------------------------------------------------
# NOTES ON THE CHOICES
# ---------------------------------------------------------------------------
# NOMINAL + REAL GDP: both kept. Real (GDPC1) strips out inflation and is the
#   better pure-growth signal; nominal (GDP) is the brief's default. Keeping both
#   is nearly free and lets features use either — and nominal/real yields the GDP
#   deflator, a second inflation read alongside CPI.
#
# VIX TWICE: VIXCLS here duplicates ^VIX from yfinance on purpose. Different
#   sources, different folders (data/raw/fred_macro/VIXCLS vs
#   data/raw/yfinance_prices/_VIX), so no collision — a free cross-source sanity
#   check. yfinance gives OHLC; FRED gives the close only.
#
# ADDING A SERIES: append its FRED id here (find it on fred.stlouisfed.org) and
#   the collector picks it up on the next run.

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

# Flat list of FRED series ids to fetch — mirrors tickers.all_tickers().
def all_series():
    return list(FRED_SERIES.keys())
