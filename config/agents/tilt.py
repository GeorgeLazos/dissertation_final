"""config/agents/tilt.py — the anchored-tilt agent.

Trades the flat investable universe, but its executed weights are a
bounded pivot around the Markowitz anchor's live row:
w = normalize(anchor * exp(TAU * tanh(a))). The policy head starts at
zero, so the untrained agent IS the anchor and training can only move
it away where the reward earns it. TAU bounds the worst case.

The anchor artifact builds itself on first use:
    python -m portfolio.models.train --sleeve tilt
"""

from config import portfolio as cfg

SLEEVE = "tilt"
ASSETS = "investable"
ANCHOR = "markowitz"
TAU = 0.25

NETWORK = {
    "hidden": 64,
    "layers": 2,
    "dropout": 0.0,      
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
    "weight_decay": 0.0,  
}

ENV = {
    "clock": "monthly",    
    "cash": True,          
    "band": cfg.BAND,      
    "eta": cfg.REWARD_ETA,
    "lam": 0.05,           
    "warmup": cfg.REWARD_WARMUP,
    "episode_len": cfg.EPISODE_LEN,
}

TRAIN = {
    "updates": 300,
    "episodes_per_update": 16,   
    "eval_every": 30,
    "seed": 0,
}

FEATURES = None
