"""config/agents/flat.py — everything the flat universe agent trains with.

Self-contained: edit here, train with
    python -m portfolio.models.train --sleeve flat

The monolithic baseline: one network over all 119 investable assets plus
CASH, observing the full asset-feature union (features outside an
asset's class ride as flagged blanks). NOT PROMOTED AND NOT DEPLOYED —
this file records the best walk-forward iteration; the agent's role is a
comparison arm against the sleeve hierarchy, judged at the final test
pass.
"""

from config import portfolio as cfg

SLEEVE = "flat"
ASSETS = "investable"

NETWORK = {
    "hidden": 512,
    "layers": 2,
    "dropout": 0.1,      # value-net dropout; policy stays dry
}

PPO = {
    "lr": 1e-4,
    "clip": 0.2,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "epochs": 10,
    "minibatches": 4,
    "entropy_coef": 1e-3,
    "value_coef": 0.5,
    "grad_clip": 0.5,
    "weight_decay": 1e-4,  # L2 on weight matrices only
}

ENV = {
    "clock": "monthly",    # decisions at month starts; drift between
    "cash": True,          # the flat agent manages its own cash weight
    "band": cfg.BAND,      # shared protocol; a literal here = deviation
    "eta": cfg.REWARD_ETA,
    "lam": 0.05,           # turnover penalty — tuned per agent
    "warmup": cfg.REWARD_WARMUP,
    "episode_len": cfg.EPISODE_LEN,
}

TRAIN = {
    "updates": 300,
    "episodes_per_update": 16,   # ~12 decisions each on the monthly clock
    "eval_every": 30,
    "seed": 0,
}

FEATURES = None
