"""
MIDAS — asset universe.

Single source of truth for what gets fetched, trained on and allocated across.
Every collector and every downstream stage imports from here.

FOUR CLASSES: equities, bond ETFs, commodity ETFs, REITs.

SOURCE BOUNDARY: Sharadar covers equities and individual REITs only — it sells
no fund data. Funds and the ^VIX index come from yfinance.

    sharadar_tickers()   108  equities + REITs   (Sharadar: prices, fundamentals, actions)
    fund_tickers()        15  ETFs + benchmarks  (yfinance)
    ^VIX                   1  index              (yfinance -> context table, never the panel)

NOTE ON SURVIVORSHIP BIAS: EQUITIES below is the S&P 100 as constituted in 2025.
Using today's membership across all history is a known bias — these are the
firms that survived and got large. See UNIVERSE_NOTE at the bottom.
"""

# ---------------------------------------------------------------------------
# CLASS 1 — EQUITIES (S&P 100, 98 tickers after exclusions below)
# ---------------------------------------------------------------------------
# Sector tags feed diagnosis, reporting and sector-level features.
#
# THE MAP IS THE CURRENT (2025) GICS APPLIED HISTORICALLY. Two cases carry
# FUTURE information rather than merely a stale label — declare both:
#   * V, MA, PYPL are tagged Financials, effective 2023-03-17. Before that
#     they were Information Technology, so the tag is wrong across all of
#     train, all of val and the first ~14 months of test.
#   * Communication Services did not exist until 2018-09-24. Seven of our
#     eight names (CMCSA, DIS, GOOGL, META, NFLX, T, TMUS, VZ) sat in Consumer
#     Discretionary or Information Technology for essentially all of train.
# ~10 of 98 names are affected — the same class of error as survivorship bias.
# Known remedy if sector ever becomes load-bearing: point-in-time SIC codes
# from EDGAR filings.
#
# Sector membership is THIN at the tails: Materials has exactly one member
# (LIN), Energy and Utilities three each. Peer-relative features must leave
# LIN blank rather than invent a peer group; any model-side sector embedding
# must merge thin sectors into one bucket. The merge rule: sectors below
# SECTOR_MIN_MEMBERS share one "other" bucket, and the result must come to
# exactly SECTOR_BUCKETS — the one-hot width models are trained against.
SECTOR_MIN_MEMBERS = 8
SECTOR_BUCKETS = 8

EQUITIES = {
    "AAPL":  "Information Technology",
    "ABBV":  "Health Care",
    "ABT":   "Health Care",
    "ACN":   "Information Technology",
    "ADBE":  "Information Technology",
    "AIG":   "Financials",
    "AMD":   "Information Technology",
    "AMGN":  "Health Care",
    "AMZN":  "Consumer Discretionary",
    "AVGO":  "Information Technology",
    "AXP":   "Financials",
    "BA":    "Industrials",
    "BAC":   "Financials",
    "BNY":   "Financials",
    "BKNG":  "Consumer Discretionary",
    "BLK":   "Financials",
    "BMY":   "Health Care",
    "BRK-B": "Financials",           # yfinance spelling; Sharadar uses BRK.B
    "C":     "Financials",
    "CAT":   "Industrials",
    "CL":    "Consumer Staples",
    "CMCSA": "Communication Services",
    "COF":   "Financials",
    "COP":   "Energy",
    "COST":  "Consumer Staples",
    "CRM":   "Information Technology",
    "CSCO":  "Information Technology",
    "CVS":   "Health Care",
    "CVX":   "Energy",
    "DE":    "Industrials",
    "DHR":   "Health Care",
    "DIS":   "Communication Services",
    "DUK":   "Utilities",
    "EMR":   "Industrials",
    "FDX":   "Industrials",
    "GD":    "Industrials",
    "GE":    "Industrials",
    "GILD":  "Health Care",
    "GM":    "Consumer Discretionary",
    "GOOGL": "Communication Services",   # Class A — GOOG excluded, see below
    "GS":    "Financials",
    "HD":    "Consumer Discretionary",
    "HON":   "Industrials",
    "IBM":   "Information Technology",
    "INTC":  "Information Technology",
    "INTU":  "Information Technology",
    "ISRG":  "Health Care",
    "JNJ":   "Health Care",
    "JPM":   "Financials",
    "KO":    "Consumer Staples",
    "LIN":   "Materials",
    "LLY":   "Health Care",
    "LMT":   "Industrials",
    "LOW":   "Consumer Discretionary",
    "MA":    "Financials",
    "MCD":   "Consumer Discretionary",
    "MDLZ":  "Consumer Staples",
    "MDT":   "Health Care",
    "MET":   "Financials",
    "META":  "Communication Services",
    "MMM":   "Industrials",
    "MO":    "Consumer Staples",
    "MRK":   "Health Care",
    "MS":    "Financials",
    "MSFT":  "Information Technology",
    "NEE":   "Utilities",
    "NFLX":  "Communication Services",
    "NKE":   "Consumer Discretionary",
    "NOW":   "Information Technology",
    "NVDA":  "Information Technology",
    "ORCL":  "Information Technology",
    "PEP":   "Consumer Staples",
    "PFE":   "Health Care",
    "PG":    "Consumer Staples",
    "PLTR":  "Information Technology",   # IPO 2020
    "PM":    "Consumer Staples",         # IPO 2008
    "PYPL":  "Financials",               # IPO 2015
    "QCOM":  "Information Technology",
    "RTX":   "Industrials",
    "SBUX":  "Consumer Discretionary",
    "SCHW":  "Financials",
    "SO":    "Utilities",
    "T":     "Communication Services",
    "TGT":   "Consumer Discretionary",
    "TMO":   "Health Care",
    "TMUS":  "Communication Services",
    "TSLA":  "Consumer Discretionary",   # IPO 2010
    "TXN":   "Information Technology",
    "UBER":  "Industrials",              # IPO 2019
    "UNH":   "Health Care",
    "UNP":   "Industrials",
    "UPS":   "Industrials",
    "USB":   "Financials",
    "V":     "Financials",               # IPO 2008
    "VZ":    "Communication Services",
    "WFC":   "Financials",
    "WMT":   "Consumer Staples",
    "XOM":   "Energy",
}

# UNIVERSE_DECISIONS — 101 S&P 100 constituents -> 98 equities:
#   GOOG  Alphabet Class C: near-identical series to GOOGL; holding both
#         doubles Alphabet's weight. GOOGL kept.
#   AMT   American Tower — a REIT, moved to REITS.
#   SPG   Simon Property — a REIT, moved to REITS.
#
# SHORT_HISTORY (no data before IPO — handled by the tradeable mask, never filled):
#   PLTR 2020, UBER 2019, PYPL 2015, NOW 2012, META 2012, TSLA 2010,
#   GM 2010 re-IPO, AVGO 2009, V 2008, PM 2008, TMUS 2007.


# ---------------------------------------------------------------------------
# CLASS 2 — BOND ETFs (yfinance)
# ---------------------------------------------------------------------------
# Spans the two axes bonds vary along, with little overlap: maturity
# (SHY short -> IEF medium -> TLT long) and credit (treasuries -> LQD
# investment-grade -> HYG high-yield), plus TIP for real rates.

BOND_ETFS = {
    "TLT": "Long Treasury (20+ yr) — high duration, main rates hedge",
    "IEF": "Intermediate Treasury (7-10 yr)",
    "SHY": "Short Treasury (1-3 yr) — near cash, low volatility",
    "LQD": "Investment-grade corporate — credit spread exposure",
    "HYG": "High-yield corporate — equity-like risk, weak diversifier",
    "TIP": "Inflation-protected (TIPS) — real rates",
}

# ---------------------------------------------------------------------------
# CLASS 3 — COMMODITY ETFs (yfinance)
# ---------------------------------------------------------------------------
# ROLL DECAY: USO holds futures and must roll contracts monthly; in contango
# this bleeds value regardless of spot price, so its long-run return badly
# lags oil. A real economic effect, not a data error — the agent faces it.

COMMODITY_ETFS = {
    "GLD": "Gold — the main commodity diversifier, history from 2004",
    "SLV": "Silver — higher volatility than gold, from 2006",
    "DBC": "Broad commodity basket — energy-heavy, from 2006",
    "USO": "Crude oil — WTI futures; severe roll decay, see note",
    "DBA": "Agriculture basket, from 2007",
}

# ---------------------------------------------------------------------------
# CLASS 4 — REITs (Sharadar — individual companies, not funds)
# ---------------------------------------------------------------------------
# Spanning property sectors so the class has genuine internal variation.

REITS = {
    "AMT":  "American Tower — cell towers (also in S&P 100)",
    "PLD":  "Prologis — industrial / logistics warehouses",
    "EQIX": "Equinix — data centres",
    "SPG":  "Simon Property — malls / retail (also in S&P 100)",
    "PSA":  "Public Storage — self storage",
    "O":    "Realty Income — net lease retail, monthly dividend",
    "WELL": "Welltower — healthcare / senior housing",
    "AVB":  "AvalonBay — residential apartments",
    "DLR":  "Digital Realty — data centres, overlaps EQIX",
    "VTR":  "Ventas — healthcare, overlaps WELL",
}

REIT_BENCHMARK = {"VNQ": "Vanguard Real Estate ETF — class benchmark, not a holding"}

# ---------------------------------------------------------------------------
# BENCHMARKS AND REFERENCE SERIES (fetched, never allocated to)
# ---------------------------------------------------------------------------

REFERENCE = {
    "SPY":  "S&P 500 ETF — market benchmark, beta reference",
    "OEF":  "S&P 100 ETF — matches the equity universe exactly",
    "^VIX": "Volatility index — context series, not a purchasable asset",
    "BIL":  "1-3 month T-bill ETF — risk-free proxy for Sharpe ratios",
}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

# The 98 equity tickers. Exclusions are applied at source — see UNIVERSE_DECISIONS.
def equities():
    return list(EQUITIES.keys())


# {class_name: [tickers]} for the four allocatable classes — the structure the
# per-class agents mirror.
def all_classes():
    return {
        "equities":    equities(),
        "bonds":       list(BOND_ETFS.keys()),
        "commodities": list(COMMODITY_ETFS.keys()),
        "reits":       list(REITS.keys()),
    }


# Flat list of all 124 symbols to fetch, including ^VIX.
def all_tickers(include_reference=True):
    out = [t for tickers in all_classes().values() for t in tickers]
    if include_reference:
        out += list(REFERENCE.keys())
        out += list(REIT_BENCHMARK.keys())
    return sorted(set(out))


# The 108 instruments Sharadar covers: equities + individual REITs. Their
# prices, fundamentals and corporate actions all come from there.
def sharadar_tickers():
    c = all_classes()
    return sorted(set(c["equities"] + c["reits"]))


# The 15 ETFs and benchmark funds. yfinance only — Sharadar sells no fund data.
def fund_tickers():
    return sorted({*BOND_ETFS, *COMMODITY_ETFS, *REIT_BENCHMARK,
                   *(t for t in REFERENCE if not t.startswith("^"))})


# ---------------------------------------------------------------------------
UNIVERSE_NOTE = """
SURVIVORSHIP BIAS — address this in the dissertation.

EQUITIES is the S&P 100 as of 2025. Applying it to a training window starting
in 2005 means the universe consists entirely of firms that survived and stayed
large — the failures are absent. Returns are therefore biased upward for EVERY
strategy tested, including the benchmarks. Because it inflates the RL agent and
the 1/N and Markowitz benchmarks alike, the comparison stays broadly fair even
though absolute returns are overstated. State it in the Data chapter and
revisit under threats to validity.
"""
