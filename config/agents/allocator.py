"""config/agents/allocator.py — everything the allocation agent trains
with.

The top of the hierarchy: class shares plus CASH over the four frozen
sleeve agents, decided monthly from the market feature block. Trains
through
    python -m portfolio.models.allocate
(the synthetic sleeve-return world lives there; train() refuses this
file by design). KAPPA blends the executed shares toward equal class
shares — 1.0 is fully direct; the walk-forward arms select it.

DEPLOYED pins the frozen sleeve checkpoint the allocator's world is
built on, per class; DEPLOYED_RUN is the allocator's own promoted
checkpoint. allocate.check() replays both against their promotion
records.
"""

from config import portfolio as cfg

SLEEVE = "allocator"
ASSETS = "classes"
KAPPA = 1.0

DEPLOYED = {
    "bonds": "2026-08-12_171642_seed0",
    "commodities": "2026-08-12_183530_seed0",
    "reits": "2026-08-12_212516_seed0",
    "equities": "2026-08-13_025743_seed0",
}

DEPLOYED_RUN = "2026-08-13_190107_seed0"

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
    "cash": True,          # the cash weight is the allocator's to manage
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

# The market block by name: synthetic assets carry no asset-grain
# features, so the allocator observes market state alone.
FEATURES = [
    "vix", "vix_chg_21", "vrp_21", "dtb3", "dtb3_chg_63", "dff",
    "term_spread", "def_spread", "cpi_yoy", "unrate", "unrate_chg_12m",
    "gdpc1_yoy", "cred_ig_21", "term_ret_21", "sb_corr_63", "cs_disp_21",
    "spy_mom_252", "spy_bear_504",
]
