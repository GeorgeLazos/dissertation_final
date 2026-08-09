"""
features/selftest.py — prove the truncation gate can FAIL.

A green --pit run means nothing unless the gate demonstrably catches leaks,
so this injects one of each known leak shape and asserts detection:

  peek       a column reading tomorrow — at the cutoff the truncated build
             has no tomorrow, so the value becomes NaN against a number.
             max(0.0, nan) is 0.0 in Python: only the NaN-PATTERN branch of
             the comparison can see this, which is why that branch exists.
  centred    a window centred on t (reads both sides) — values near the
             cutoff change when the future is deleted.
  zscore     a full-sample normalization — every value changes.

Shapes are tested against the comparator, then one live end-to-end run
doctors the market builder with a peek and drives the real gate plumbing.
Everything is restored afterwards; nothing on disk changes.

    python -m features.selftest
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from collectors._core import console_utf8
from dataset import loader as dloader
from features import asset_features, market_features
from features.checks import TRUNCATION_DATE, compare_frames

RESULTS = []


def _case(label: str, caught: bool, must_catch: bool = True):
    ok = caught == must_catch
    RESULTS.append(ok)
    verdict = "CAUGHT" if caught else ("clean" if not must_catch else "MISSED")
    print(f"  [{verdict:6s}] {label}")


def comparator_cases():
    n = 120
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)
    x = pd.Series(rng.normal(size=n), index=dates)
    cut = 100

    def frames(full_col, trunc_col):
        full = pd.DataFrame({"date": dates, "f": full_col.values})
        trunc = pd.DataFrame({"date": dates[:cut], "f": trunc_col.values[:cut]})
        return compare_frames(full, trunc, ["date"])

    honest = x.rolling(5).mean()
    _case("honest trailing window", bool(frames(honest, honest)),
          must_catch=False)

    peek = x.shift(-1)
    _case("one-day peek (NaN-pattern branch only)",
          bool(frames(peek, x[:cut].shift(-1).reindex(dates))))

    centred_full = x.rolling(5, center=True).mean()
    centred_trunc = x[:cut].rolling(5, center=True).mean().reindex(dates)
    _case("centred window", bool(frames(centred_full, centred_trunc)))

    z_full = (x - x.mean()) / x.std()
    z_trunc = ((x[:cut] - x[:cut].mean()) / x[:cut].std()).reindex(dates)
    _case("full-sample z-score", bool(frames(z_full, z_trunc)))


def live_case():
    # doctor ONE market column into a peek, run the real gate plumbing
    T = pd.Timestamp(TRUNCATION_DATE)
    real_build = market_features.build

    def doctored():
        out = real_build()
        out["vix_chg_21"] = out["vix_chg_21"].shift(-1)
        return out

    full = doctored()
    real = {"prices": dloader.prices, "macro": dloader.macro,
            "calendar": dloader.calendar}
    P, M, CAL = dloader.prices(), dloader.macro(), dloader.calendar()
    try:
        dloader.prices = lambda split=None: P[P["date"] <= T].reset_index(drop=True)
        dloader.macro = lambda split=None: M[M["date"] <= T].reset_index(drop=True)
        dloader.calendar = lambda: CAL[CAL <= T]
        trunc = doctored()
    finally:
        for k, v in real.items():
            setattr(dloader, k, v)
    verdict = compare_frames(full, trunc, ["date"])
    _case("live gate vs a doctored market builder",
          "vix_chg_21" in verdict)
    _case("live gate stays quiet on the honest columns",
          any(c != "vix_chg_21" for c in verdict), must_catch=False)


# The asset channel, on a 5-instrument subset so both builds run in seconds:
# a peek doctored into a FUNDAMENTALS-driven column must be caught, which
# exercises the fundamentals truncation path the market case never touches.
def live_asset_case():
    T = pd.Timestamp(TRUNCATION_DATE)
    keep = {"AAPL", "MSFT", "KO", "TLT", "SPY"}
    real = {"prices": dloader.prices, "macro": dloader.macro,
            "fundamentals": dloader.fundamentals, "calendar": dloader.calendar}
    P, M, F, CAL = (dloader.prices(), dloader.macro(),
                    dloader.fundamentals(), dloader.calendar())
    P5 = P[P["ticker"].isin(keep)].reset_index(drop=True)
    F5 = F[F["ticker"].isin(keep)].reset_index(drop=True)

    def doctored():
        out = asset_features.build()
        out["ep_ttm"] = out.groupby("ticker")["ep_ttm"].shift(-1)
        return out

    try:
        dloader.prices = lambda split=None: P5
        dloader.fundamentals = lambda split=None: F5
        full = doctored()
        dloader.prices = lambda split=None: P5[P5["date"] <= T].reset_index(drop=True)
        dloader.macro = lambda split=None: M[M["date"] <= T].reset_index(drop=True)
        dloader.fundamentals = lambda split=None: (
            F5[F5["published"] <= T].reset_index(drop=True))
        dloader.calendar = lambda: CAL[CAL <= T]
        trunc = doctored()
    finally:
        for k, v in real.items():
            setattr(dloader, k, v)
    verdict = compare_frames(full, trunc, ["date", "ticker"])
    _case("live gate vs a doctored ASSET builder (fundamentals channel)",
          "ep_ttm" in verdict)
    _case("asset gate quiet on the honest columns",
          any(c != "ep_ttm" for c in verdict), must_catch=False)


if __name__ == "__main__":
    console_utf8()
    print("COMPARATOR")
    comparator_cases()
    print("LIVE")
    live_case()
    live_asset_case()
    n_ok = sum(RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} passed")
    raise SystemExit(0 if n_ok == len(RESULTS) else 1)
