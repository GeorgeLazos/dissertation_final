"""config/agents/equities.py — everything the equity agent trains with.

Self-contained: edit here, train with
    python -m portfolio.models.train --sleeve equities
The 98-asset sleeve carries the widest observation (~11k inputs) and the
richest decision, so it gets the largest network — and it is the one
sleeve where regularisation helps: value-net dropout with decoupled
weight decay adds fold-mean Sharpe at this width where both reduce it
on the small sleeves.
"""

from config import portfolio as cfg

SLEEVE = "equities"

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
