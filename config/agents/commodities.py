"""config/agents/commodities.py — everything the commodity agent trains
with.

Self-contained: edit here, train with
    python -m portfolio.models.train --sleeve commodities
Five funds, and episodes never start before a second one trades
(2006-02-06) — the environment enforces that floor itself.
"""

from config import portfolio as cfg

SLEEVE = "commodities"

NETWORK = {
    "hidden": 64,
    "layers": 2,
    "dropout": 0.0,      # value-net dropout; policy stays dry
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
    "weight_decay": 0.0,   # L2 on weight matrices only
}

ENV = {
    "clock": "monthly",    # decisions at month starts; drift between
    "cash": False,         # the sleeve is always fully invested
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
