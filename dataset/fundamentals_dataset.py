"""
dataset/fundamentals_dataset.py — quarterly fundamentals, one row per filing.

Sharadar SF1 as-reported quarterly, WIDE: every standardized field as a
column, every company on the same schema. A restated quarter keeps BOTH rows
(same quarter, different published dates) — the as-reported history is the
point-in-time record, and the loader picks what was known at a given day.

published is the PIT key: the day the row became available. A row is
admissible from the first trading day AFTER it, since filings can land after
the close.

EDGAR is not read here — it is the cross-check source (dataset/checks.py).

INPUT
    sharadar_fundamentals.load(t)   list of ~129 dicts per 108 tickers,
                                    112 keys each: 7 metadata + ~105
                                    financial fields, numerics sometimes
                                    arriving as strings. Vendor spelling
                                    (BRK.B) — rewritten to ours (BRK-B).

OUTPUT  data/processed/fundamentals_dataset.parquet   (12,568 x 77)

  KEYS (5)
    ticker        str          OUR spelling, matching price_dataset
    quarter       datetime64   calendardate — the normalized quarter end
    period_end    datetime64   reportperiod — the fiscal period's real last day
    published     datetime64   when it became knowable — THE PIT KEY
    fiscal        str          the company's own label, e.g. '2026-Q3'

  VALUES (72, all float64) — raw reported facts only; every vendor-computed
  ratio is dropped (see DROP) and layer 2 builds its own.
    balance sheet (26)   assets assetsc assetsnc cashneq investments
                         investmentsc investmentsnc receivables inventory
                         ppnenet intangibles tangibles taxassets deposits
                         liabilities liabilitiesc liabilitiesnc payables
                         debt debtc debtnc deferredrev taxliabilities
                         equity retearn accoci
    income (20)          revenue cor gp opex sgna rnd sbcomp depamor opinc
                         ebit ebitda intexp ebt taxexp consolinc netinc
                         netinccmn netincdis netincnci prefdivis
    cash flow (12)       ncfo ncfi ncff ncf ncfbus ncfinv ncfdebt
                         ncfcommon ncfdiv ncfx capex fcf
    shares (10)          sharesbas shareswa shareswadil eps epsdil dps
                         bvps tbvps sps fcfps
    market (2)           marketcap — the vendor's authoritative figure
                                     (close x sharesbas misses share-class
                                     cases, 98.2% within 1%);
                         price — the close ON THE FILING DATE (median error
                                 0.000000 vs close(published), 0.0098 vs the
                                 next day): the anchor for scaling marketcap
                                 forward
    metadata (2)         fxusd sharefactor

  TTM means the sum of four single quarters — the vendor's own ps, evebitda
  and pe1 match that construction to within 0.007%.

  COVERAGE (measured on quarters >= 2004): the current/non-current family
  (assetsc assetsnc liabilitiesc liabilitiesnc debtc debtnc investmentsc
  investmentsnc) is 84/108 — banks, insurers and REITs do not present a
  classified balance sheet, which is definitional, not missing. Everything
  else is 108/108.

    python -m dataset.fundamentals_dataset            # build
    python -m dataset.fundamentals_dataset --show     # print the stored table
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from collectors import sharadar_fundamentals
from config.tickers import sharadar_tickers
from dataset._core import cli, load_table

NAME = "fundamentals_dataset"

# Sharadar metadata keys -> our names; 'dimension' (constant ARQ) and
# 'lastupdated' (vendor bookkeeping) are dropped. The availability date is
# the API's 'date' field.
KEYS = {"ticker": "ticker", "calendardate": "quarter",
        "reportperiod": "period_end", "date": "published",
        "fiscalperiod": "fiscal"}
DROP = {
    "dimension", "lastupdated",
    # always empty on the ARQ dimension
    "roa", "roe", "roic", "ros",
    "assetsavg", "equityavg", "invcapavg", "assetturnover",
    # exact duplicates of their base field: every filer reports in USD
    "revenueusd", "ebitusd", "ebitdausd", "netinccmnusd",
    "equityusd", "debtusd", "cashnequsd", "epsusd",
    # vendor ratios, recomputable from kept fields. Measured median relative
    # difference of the reconstruction:
    #   ev = marketcap + debt - cashneq           0.00000
    #   workingcapital = assetsc - liabilitiesc   0.00000
    #   de = liabilities / equity                 0.00010
    #   currentratio = assetsc / liabilitiesc     0.00016
    #   ps = marketcap / revenue_ttm              0.00007
    #   evebitda = ev / ebitda_ttm                0.00002
    #   pe1 = price / eps_ttm                     0.00001
    #   pb grossmargin netmargin ebitdamargin payoutratio    all < 0.3%
    # pe, evebit, divyield and invcap follow formulas the vendor does not
    # document (pe1 matches price/eps_ttm exactly; pe is 1.3% off the same
    # formula). A ratio whose definition cannot be stated is worse than one
    # computed from the raw fields.
    "ev", "pe", "pe1", "pb", "ps", "ps1", "evebit", "evebitda", "de",
    "currentratio", "divyield", "payoutratio", "grossmargin", "netmargin",
    "ebitdamargin", "workingcapital", "invcap",
}


# All quarters for all companies as one wide frame: metadata renamed and
# dated, every financial field coerced to float. The ticker column is OUR
# spelling, not the vendor's — Sharadar answers for BRK.B, but every other
# table says BRK-B, and a join on ticker must not drop Berkshire.
def build() -> pd.DataFrame:
    frames = []
    for t in sharadar_tickers():
        rows = sharadar_fundamentals.load(t)
        if rows:
            df = pd.DataFrame(rows)
            df["ticker"] = t
            frames.append(df)
    f = pd.concat(frames, ignore_index=True)
    f = f.drop(columns=[c for c in DROP if c in f.columns]).rename(columns=KEYS)

    for c in ("quarter", "period_end", "published"):
        f[c] = pd.to_datetime(f[c]).astype("datetime64[us]")
    value_cols = [c for c in f.columns if c not in ("ticker", "fiscal", "quarter",
                                                    "period_end", "published")]
    f[value_cols] = (f[value_cols].apply(pd.to_numeric, errors="coerce")
                     .astype("float64"))

    key_order = ["ticker", "quarter", "period_end", "published", "fiscal"]
    f = f[key_order + value_cols]
    return f.sort_values(["ticker", "quarter", "published"]).reset_index(drop=True)


# The stored fundamentals: one row per filing, wide.
def load() -> pd.DataFrame:
    return load_table(NAME)


if __name__ == "__main__":
    cli(NAME, build, "Build the quarterly fundamentals table.")
