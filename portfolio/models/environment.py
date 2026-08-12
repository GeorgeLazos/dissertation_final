"""
portfolio/models/environment.py — the world an agent lives in.

One class, instantiated per agent: asset list, feature list, clock and
cash flag differ; the loop, cost model and reward rule are identical. All
money arithmetic comes from engine.step() — the baselines' own physics —
except the one step the engine cannot express: a cash-less sleeve's first
buy-in (no cash slot to sell), computed here with the engine's formulas
at value 1.0 and recorded at turnover 0.5.

    reset()              -> obs, info         a random training episode
    step(action)         -> obs, reward, done, info
    evaluate(policy, ..) -> engine.run()-shaped result + the decision frame

The observation is [scaled asset feats, scaled market feats, their
was-blank flags, drifted weights, episode progress]; obs_size and
action_size are properties. Every done is a time-limit truncation
(info["truncated"]) — the trainer bootstraps the terminal value — and a
finished episode refuses further step() calls.

A step is one DECISION: the trade at that session's close, then a drift
advance to the next decision date (daily = the next session; monthly =
the next month start; a decision falling on the window's final session
is never offered — nothing follows it to hold). Arrivals call
step(target=None) and the trade calls step(zeros, target); x*(1+0) == x
exactly, so a cash environment's recorded run replays through
engine.run() bit-for-bit, band included — the frame holds
post-interpretation, PRE-band targets. A cash-less sleeve is verified by
the decomposed engine.step replay instead: run() cannot start uninvested.

Actions are weights over the agent's columns: weights on non-tradeable
assets are zeroed and the rest renormalised; NaN, negatives or a bad sum
raise. A set asset_cap is enforced by capped-simplex projection over the
live assets only, CASH exempt. Reward is the differential Sharpe
(Moody & Saffell) of per-session returns minus TURNOVER_LAMBDA times
executed turnover, withheld in full for the first REWARD_WARMUP sessions
of an episode.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from config import portfolio as cfg
from config.splits import get_split
from features import loader as fl, registry
from portfolio import engine
from portfolio.models import scaling


# One differential-Sharpe update. Returns the new running moments and the
# reward increment for this return. predict_proba-style smoothing is never
# involved: A and B carry only past returns.
def dsr_step(A: float, B: float, r: float, eta: float) -> tuple:
    dA = r - A
    dB = r * r - B
    var = B - A * A
    if var > 1e-12:
        D = (B * dA - 0.5 * A * dB) / var ** 1.5
    else:
        D = 0.0
    return A + eta * dA, B + eta * dB, D


# Project onto the capped simplex: every weight <= cap, sum preserved at 1.
# Breachers are pinned at the cap and the remainder renormalised, repeated
# until nothing breaches — renormalising THROUGH a cap would breach it.
def project_cap(w: np.ndarray, cap: float) -> np.ndarray:
    if cap * len(w) < 1.0 - 1e-9:
        raise ValueError(f"cap {cap} infeasible for {len(w)} assets")
    w = w.copy()
    fixed = np.zeros(len(w), dtype=bool)
    for _ in range(len(w)):
        over = ~fixed & (w > cap + 1e-15)
        if not over.any():
            break
        w[over] = cap
        fixed |= over
        free = ~fixed
        remainder = 1.0 - cap * fixed.sum()
        s = w[free].sum()
        w[free] = remainder / free.sum() if s <= 0 else w[free] * remainder / s
    return w

# One instance per agent: its own assets, features, clock and knobs.
class Environment:
    def __init__(self, assets: list, features: list, clock: str = "daily",
                 cash: bool = False, limits: dict | None = None,
                 band: float = cfg.BAND, scaling_name: str | None = None,
                 window: str | tuple = "train", seed: int = 0,
                 eta: float = cfg.REWARD_ETA,
                 lam: float = cfg.TURNOVER_LAMBDA,
                 warmup: int = cfg.REWARD_WARMUP,
                 episode_len: int = cfg.EPISODE_LEN,
                 bundle: dict | None = None):

        if clock not in ("daily", "monthly"):
            raise ValueError(f"unknown clock {clock!r}")
        unknown = set(limits or {}) - {"asset_cap"}
        if unknown:
            raise ValueError(f"unknown limits {sorted(unknown)}")

        self.assets = list(assets)
        self.columns = self.assets + ([engine.CASH] if cash else [])
        self.cash = cash
        self.clock = clock
        self.band = band
        self.cap = (limits or {}).get("asset_cap", cfg.ASSET_CAP)
        self.eta, self.lam = eta, lam
        self.warmup, self.episode_len = warmup, episode_len
        self.rng = np.random.default_rng(seed)

        if bundle is None:
            from portfolio.run import load_bundle
            bundle = load_bundle()

        self.dates = bundle["dates"]
        ret = bundle["ret"][self.assets].copy()
        if cash:
            ret[engine.CASH] = bundle["cash"].reindex(self.dates).values
        self.R = ret.to_numpy(dtype=float)
        self.tradeable = bundle["tradeable"][self.assets].to_numpy(dtype=bool)
        self.rates = cfg.cost_rates(self.columns).to_numpy(dtype=float)

        # Determine the window of interest and the training start index. 
        start, end = get_split(window) if isinstance(window, str) else window
        inside = (self.dates >= start) & (self.dates <= end)
        if not inside.any():
            raise ValueError(f"window {start}..{end} does not intersect " f"the calendar")
        self.i_lo = int(np.argmax(inside))
        self.i_hi = int(len(self.dates) - np.argmax(inside[::-1]) - 1)

        # where training starts: the first session with at least two tradeable assets
        floor = self.dates.searchsorted(pd.Timestamp(cfg.AGENT_TRAIN_START))
        need = min(2, len(self.assets))
        multi = np.nonzero(self.tradeable.sum(axis=1) >= need)[0]
        if not len(multi):
            raise ValueError("no session offers a decision in this sleeve")
        self.i_train_lo = max(self.i_lo, int(floor), int(multi[0]))

        # Load the features and, if requested, the scaling stats.
        self._load_features(features)
        if scaling_name is not None:
            stats = scaling.load(scaling_name)

            if stats["features"] != list(features):
                raise ValueError("scaling file was fitted on a different feature list")
            if stats["assets"] != list(self.assets):
                raise ValueError("scaling file was fitted on different assets")

            self.mu_a, self.sd_a = (scaling.vectors(stats, self.a_names)
                                    if self.a_names else (None, None))
            self.mu_m, self.sd_m = (scaling.vectors(stats, self.m_names)
                                    if self.m_names else (None, None))
        else:
            z = lambda n: (np.zeros(n), np.ones(n))
            self.mu_a, self.sd_a = z(len(self.a_names))
            self.mu_m, self.sd_m = z(len(self.m_names))

        self._reset_state(self.i_train_lo, self.i_hi)

    # Pivot each feature into a (dates x assets) grid once; observations are
    # then O(1) slices. Market features are one row per date.
    def _load_features(self, features: list) -> None:
        self.a_names = [n for n in features
                        if registry.spec(n)["grain"] == "asset"]
        self.m_names = [n for n in features
                        if registry.spec(n)["grain"] == "market"]
        
        if self.a_names:
            af = fl.features_asset(None, self.a_names)
            af = af[af["ticker"].isin(self.assets)]
            grids = [af.pivot(index="date", columns="ticker", values=n)
                       .reindex(index=self.dates, columns=self.assets)
                       .to_numpy(dtype=float)
                     for n in self.a_names]
            self.F_asset = np.stack(grids, axis=-1)   # dates x assets x feats
        else:
            self.F_asset = np.zeros((len(self.dates), len(self.assets), 0))

        if self.m_names:
            mf = fl.features_market(None, self.m_names).set_index("date")
            self.F_market = (mf.reindex(self.dates)[self.m_names].to_numpy(dtype=float))
        else:
            self.F_market = np.zeros((len(self.dates), 0))

    # Reset the environment's state so no trace of the previous episode remains.
    def _reset_state(self, i: int, i_end: int) -> None:
        self.i = i
        self.i_end = i_end

        if self.clock == "monthly":
            sub = self.dates[i:i_end + 1]
            self._starts = np.array(
                [self.dates.get_loc(d) for d in cfg.month_starts(sub)])
        else:
            self._starts = None

        k = len(self.columns)
        if self.cash:
            self.holdings = np.zeros(k)
            self.holdings[-1] = 1.0
            self.value = 1.0
        else:
            self.holdings = None          # uninvested until the first action
            self.value = 1.0

        self.A = 0.0
        self.B = 0.0
        self.steps_seen = 0
        self.record: dict = {}
        self.path: dict = {"value": {}, "ret": {}, "cost": {}, "turnover": {}, "weights": {}}
        self.prev_value = 1.0
        self._ep_sessions = i_end - i + 1
        self._live = True

    # The network constructor reads these; no RNG is consumed.
    @property
    def obs_size(self) -> int:
        return (2 * (len(self.assets) * len(self.a_names) + len(self.m_names))
                + len(self.columns) + 1)

    @property
    def action_size(self) -> int:
        return len(self.columns)

    # ── the public loop ─────────────────────────────────────────────────

    # Start a training episode: a random ~episode_len window inside the
    # environment's span, no earlier than AGENT_TRAIN_START. Returns the
    # first observation and its info, so the mask is in hand for the very
    # first decision.
    def reset(self) -> tuple:
        hi = self.i_hi - self.episode_len
        if hi < self.i_train_lo:
            raise ValueError("window shorter than one episode")
        start = int(self.rng.integers(self.i_train_lo, hi + 1))
        self._reset_state(start, start + self.episode_len)
        return self._observe(), self._info()

    # One decision: interpret the action, trade at this session's close,
    # advance to the next decision date, return what the agent sees there.
    def step(self, action) -> tuple:
        if not self._live:
            raise RuntimeError("episode finished — call reset() or "
                               "evaluate()")
        target = self._interpret(action)
        self.record[self.dates[self.i]] = target

        reward = self._trade(target)
        j = self._next_decision()
        done = False
        while True:
            self.i += 1
            if self.i > self.i_end:
                done = True
                self.i = self.i_end
                break
            if self.i == j == self.i_end:
                # Terminal landing: no decision follows, so the session's
                # value is final — pay it here and end the walk.
                reward += self._advance_day(pay=True)
                done = True
                break
            if self.i == j:
                self._advance_day(pay=False)
                break
            reward += self._advance_day(pay=True)
        if done:
            self._live = False
        return (self._observe(), reward, done, self._info())

    # ── internals ───────────────────────────────────────────────────────

    # Takes an action: checks for errors, renormalises if needed then apply caps if needed
    def _interpret(self, action) -> np.ndarray:
        a = np.asarray(action, dtype=float).copy()

        if a.shape != (len(self.columns),):
            raise ValueError(f"action must have {len(self.columns)} weights")
        if np.isnan(a).any() or (a < -1e-9).any():
            raise ValueError("action contains NaN or negative weights")
        if abs(a.sum() - 1.0) > 1e-6:
            raise ValueError(f"action sums to {a.sum():.6f}, not 1")

        # Apply the tradeable mask and renormalise. The cash slot is never masked
        live = self.tradeable[self.i]
        dead = a[:len(self.assets)][~live]
        if (dead != 0.0).any():
            # Renormalise only when masking removed weight
            a[:len(self.assets)][~live] = 0.0
            s = a.sum()
            if s <= 0:
                raise ValueError("no weight on any tradeable asset")
            a = a / s

        # Apply the asset_cap if requested. The cash slot is never capped.
        if self.cap is not None:
            live_ix = np.nonzero(live)[0]
            w_cash = a[-1] if self.cash else 0.0
            mass = 1.0 - w_cash
            if mass > 1e-12:
                if self.cap * len(live_ix) < mass - 1e-9:
                    raise ValueError(f"cap {self.cap} infeasible for "
                                     f"{len(live_ix)} live assets holding "
                                     f"{mass:.3f}")
                sub = a[live_ix] / mass
                a[live_ix] = project_cap(sub, self.cap / mass) * mass

        return a

    # Trade at the close of the current session. Returns the differential-
    # Sharpe increment of the session's return, minus lambda times the
    # turnover actually executed; both withheld during warm-up.
    def _trade(self, target: np.ndarray) -> float:
        d = self.dates[self.i]

        if self.holdings is None:
            # First action of a cash-less sleeve: buy in from value 1.0,
            # full costs on the whole target — the all-cash start without
            # a cash slot.
            cost = float(np.abs(target) @ self.rates) * self.value
            turnover = 0.5 * float(np.abs(target).sum())
            net = self.value - cost
            self.holdings = target * net
            rec = {"value": net, "cost": cost, "turnover": turnover}
        else:
            self.holdings, rec = engine.step(
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

        # get the reward increment for this decision's return 
        # and apply a penalty for the turnover executed, if the warmup is over
        D = self._reward_increment(r)
        pen = (self.lam * rec["turnover"]
               if self.steps_seen > self.warmup else 0.0)
        return D - pen

    # One drift session: apply the day's returns, no trade. pay=False on a
    # landing day — the next trade finalises its value and pays its return.
    def _advance_day(self, pay: bool) -> float:
        d = self.dates[self.i]

        rets = engine.police_rets(self.holdings, self.R[self.i], self.columns, d, first=False)
        self.holdings, rec = engine.step(self.holdings, rets, None, self.rates, self.band)

        self.value = rec["value"]
        self.path["value"][d] = rec["value"]
        self.path["cost"][d] = 0.0
        self.path["turnover"][d] = 0.0
        self.path["weights"][d] = self.holdings / self.value
        if not pay:
            return 0.0

        r = self.value / self.prev_value - 1.0
        self.path["ret"][d] = r
        self.prev_value = self.value
        return self._reward_increment(r)

    # Reward increment for a return: update the running moments and return
    def _reward_increment(self, r: float) -> float:
        self.A, self.B, D = dsr_step(self.A, self.B, r, self.eta)
        self.steps_seen += 1
        return D if self.steps_seen > self.warmup else 0.0

    # Determine the next decision point
    def _next_decision(self) -> int:
        if self.clock == "daily":
            return self.i + 1
        later = self._starts[self._starts > self.i]
        return int(later[0]) if len(later) else self.i_end

    # return the observation vector for the current session
    def _observe(self) -> np.ndarray:
        raw_a = self.F_asset[self.i]                      # assets x feats
        sa, fa = scaling.apply(raw_a, self.mu_a, self.sd_a) \
            if raw_a.shape[-1] else (raw_a.astype(np.float32),) * 2
        raw_m = self.F_market[self.i]
        sm, fm = scaling.apply(raw_m, self.mu_m, self.sd_m) \
            if raw_m.shape[-1] else (raw_m.astype(np.float32),) * 2
        w = (self.holdings / self.value if self.holdings is not None
             else np.zeros(len(self.columns)))
        # Episode progress: the reward's differential-Sharpe scale drifts
        # with episode age, so the value function needs the age in view.
        progress = np.float32(min(1.0, self.steps_seen / self._ep_sessions))
        return np.concatenate([sa.ravel(), sm.ravel(), fa.ravel(), fm.ravel(),
                               w.astype(np.float32), [progress]])

    # Return the info dict for the current session
    def _info(self) -> dict:
        return {"date": self.dates[self.i],
                "tradeable": self.tradeable[self.i].copy(),
                "weights": (self.holdings / self.value
                            if self.holdings is not None
                            else np.zeros(len(self.columns))),
                "value": self.value,
                # Episodes only ever end by running out of window: a done
                # is a truncation, so the trainer bootstraps V(final obs).
                "truncated": not self._live}

    # ── deterministic evaluation ────────────────────────────────────────

    # One pass of a frozen policy over a whole window. Returns an
    # engine.run()-shaped result plus the decision frame.
    def evaluate(self, policy, window: str | tuple | None = None) -> dict:

        if window is not None:
            start, end = (get_split(window) if isinstance(window, str)
                          else window)
            inside = (self.dates >= start) & (self.dates <= end)
            if not inside.any():
                raise ValueError(f"window {start}..{end} does not intersect "
                                 f"the calendar")
            i0 = int(np.argmax(inside))
            i1 = int(len(self.dates) - np.argmax(inside[::-1]) - 1)
        else:
            i0, i1 = self.i_lo, self.i_hi

        self._reset_state(i0, i1)
        saved_warmup, self.warmup = self.warmup, -1   # withhold nothing here

        try:
            obs, info = self._observe(), self._info()
            done = i0 >= i1
            while not done:
                action = policy(obs, info)
                obs, _, done, info = self.step(action)
        finally:
            self.warmup = saved_warmup

        idx = pd.DatetimeIndex(sorted(self.path["value"]))
        out = {k: pd.Series([self.path[k][d] for d in idx], index=idx, name=k)
               for k in ("value", "ret", "cost", "turnover")}
        out["weights"] = pd.DataFrame(
            [self.path["weights"][d] for d in idx], index=idx,
            columns=self.columns)
        out["decisions"] = pd.DataFrame(
            [self.record[d] for d in sorted(self.record)],
            index=pd.DatetimeIndex(sorted(self.record)), columns=self.columns)
        return out
