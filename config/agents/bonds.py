"""config/agents/bonds.py — everything the bond agent trains with.

Self-contained: edit here, train with
    python -m portfolio.models.train --sleeve bonds
Nothing else needs touching. NETWORK sizes the policy/value nets, PPO
holds the optimiser's hyperparameters, ENV overrides the environment's
knobs for this agent, TRAIN is the run itself. FEATURES = None derives the observation
list from the registry (class-scoped + market); a list here overrides it.

This sleeve trades MONTHLY on a curated observation: six bond ETFs carry
~0.34% daily cross-sectional dispersion, and the walk-forward arms showed
the exploitable structure is macro-frequency duration/credit tilting —
visible only with daily churn removed and the observation cut to the
features such tilts act on.
"""

from config import portfolio as cfg

SLEEVE = "bonds"

NETWORK = {
    "hidden": 64,          # width of each hidden layer
    "layers": 2,           # hidden layers in policy and value nets
    "dropout": 0.0,        # value-net dropout; policy stays dry
}

PPO = {
    "lr": 1e-4,            # Adam learning rate
    "clip": 0.2,           # PPO clip range
    "gamma": 0.99,         # discount
    "gae_lambda": 0.95,    # advantage smoothing
    "epochs": 10,          # passes over each rollout
    "minibatches": 4,
    "entropy_coef": 1e-3,  # exploration bonus
    "value_coef": 0.5,     # value-loss weight
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
    "eval_every": 30,      # validation Sharpe cadence (best ckpt kept)
    "seed": 0,
}

# The sleeve's own trend/risk state plus the rate, credit and macro block —
# what a duration or credit tilt can act on.
FEATURES = [
    "mom_21", "mom_63", "mom_126", "mom_252", "rv_21", "rv_63",
    "ewma_vol", "dd_252", "beta_252", "corr_252", "dp_ttm", "mom_rank",
    "vix", "dtb3", "dtb3_chg_63", "dff", "term_spread", "def_spread",
    "cpi_yoy", "unrate_chg_12m", "cred_ig_21", "term_ret_21",
    "sb_corr_63", "cs_disp_21",
]
