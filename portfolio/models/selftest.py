"""
portfolio/models/selftest.py — prove the environment checks can fail.

Each case injects one fault into a live copy of the machinery, asserts the
relevant check family reports it (a crash counts: the CLI exits non-zero
either way), then restores everything and confirms a clean control. A
green checks run means nothing unless this is green beside it.

    python -m portfolio.models.selftest
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from portfolio import engine as eng
from portfolio.models import checks, environment, scaling


def _caught(fn) -> bool:
    try:
        return bool(fn())
    except Exception:
        return True


# A wrong differential-Sharpe formula (the -0.5*A*dB term dropped).
def inject_bad_dsr() -> bool:
    orig = environment.dsr_step

    def bad(A, B, r, eta):
        dA, dB = r - A, r * r - B
        var = B - A * A
        D = (B * dA) / var ** 1.5 if var > 1e-12 else 0.0
        return A + eta * dA, B + eta * dB, D

    environment.dsr_step = bad
    try:
        return _caught(checks.check_reward)
    finally:
        environment.dsr_step = orig


# The turnover penalty leaking through warm-up (withholding only D).
def inject_penalty_leak() -> bool:
    orig = environment.Environment._trade

    def leaky(self, target):
        d = self.dates[self.i]
        if self.holdings is None:
            cost = float(np.abs(target) @ self.rates) * self.value
            turnover = 0.5 * float(np.abs(target).sum())
            net = self.value - cost
            self.holdings = target * net
            rec = {"value": net, "cost": cost, "turnover": turnover}
        else:
            self.holdings, rec = eng.step(
                self.holdings, np.zeros(len(self.columns)), target,
                self.rates, self.band)
        self.value = rec["value"]
        self.path["value"][d] = rec["value"]
        self.path["cost"][d] = rec["cost"]
        self.path["turnover"][d] = rec["turnover"]
        self.path["weights"][d] = self.holdings / self.value
        r = self.value / self.prev_value - 1.0
        self.path["ret"][d] = r
        self.prev_value = self.value
        return self._reward_increment(r) - self.lam * rec["turnover"]

    environment.Environment._trade = leaky
    try:
        return _caught(checks.check_reward)
    finally:
        environment.Environment._trade = orig


# The tradeable mask deleted from action interpretation.
def inject_no_mask() -> bool:
    orig = environment.Environment._interpret

    def unmasked(self, action):
        a = np.asarray(action, dtype=float)
        return a / a.sum()

    environment.Environment._interpret = unmasked
    try:
        return _caught(checks.check_interpret)
    finally:
        environment.Environment._interpret = orig


# Clip-and-renormalise passed off as the cap projection.
def inject_clip_cap() -> bool:
    orig = environment.project_cap

    def clip(w, cap):
        w = np.minimum(w, cap)
        return w / w.sum()

    environment.project_cap = clip
    checks.project_cap = clip
    try:
        return _caught(checks.check_interpret)
    finally:
        environment.project_cap = orig
        checks.project_cap = orig


# The scaler fitted through validation as well as train.
def inject_scaler_leak() -> bool:
    orig = scaling.fit

    def leaky(name, features, assets):
        stats = orig(name, features, assets)
        stats["window"] = ["2005-01-01", "2021-12-31"]
        return stats

    scaling.fit = leaky
    try:
        return _caught(checks.check_scaling)
    finally:
        scaling.fit = orig


# The finished-episode guard removed: post-done steps re-trade silently.
def inject_no_lifecycle() -> bool:
    orig = environment.Environment.step

    def unguarded(self, action):
        self._live = True
        return orig(self, action)

    environment.Environment.step = unguarded
    try:
        return _caught(checks.check_episode)
    finally:
        environment.Environment.step = orig


CASES = {
    "wrong DSR formula": inject_bad_dsr,
    "penalty leaks through warm-up": inject_penalty_leak,
    "tradeable mask deleted": inject_no_mask,
    "clip instead of projection": inject_clip_cap,
    "scaler window into validation": inject_scaler_leak,
    "no finished-episode guard": inject_no_lifecycle,
}

if __name__ == "__main__":
    from collectors._core import console_utf8
    console_utf8()
    missed = []
    for label, case in CASES.items():
        ok = case()
        print(f"  [{'CAUGHT' if ok else 'MISSED'}] {label}")
        if not ok:
            missed.append(label)
    clean = sum(len(f()) for f in checks.FAMILIES.values())
    print(f"  [{'clean' if clean == 0 else 'DIRTY'} ] control after restores")
    n = len(CASES)
    print(f"\n{n - len(missed)}/{n} caught" + ("" if clean == 0 else
          "; CONTROL FAILED"))
    raise SystemExit(1 if missed or clean else 0)
