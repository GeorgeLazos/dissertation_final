"""
portfolio/models/networks.py — the policy and value networks.

The policy is a Gaussian in unconstrained space: the network outputs a mean
per column, a learned state-independent log-std supplies the spread, and
the sampled vector becomes portfolio weights through a masked softmax —
non-tradeable columns get -1e9 before the softmax, so their weight is
exactly zero and the environment's own masking never has to repair the
action. PPO's log-probabilities live on the Gaussian sample, not the
weights: the softmax is part of the world, not the distribution.

Deterministic evaluation uses the mean — no sampling, so a checkpoint
reproduces its backtest exactly.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

# Returns the defined network architecture
def _mlp(n_in: int, hidden: int, n_out: int, layers: int, dropout: float = 0.0) -> nn.Sequential:
    seq, last = [], n_in
    for _ in range(layers):
        seq += [nn.Linear(last, hidden), nn.Tanh()]
        if dropout > 0:
            seq.append(nn.Dropout(dropout))
        last = hidden
    return nn.Sequential(*seq, nn.Linear(last, n_out))

class PolicyValue(nn.Module):
    # Dropout regularises the VALUE net only: dropout inside the policy
    # would desynchronise PPO's ratio (the action sampled under one unit
    # mask, scored under another).
    def __init__(self, obs_size: int, action_size: int, hidden: int = 64, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.pi = _mlp(obs_size, hidden, action_size, layers)
        self.v = _mlp(obs_size, hidden, 1, layers, dropout)
        # Initial exploration std exp(-1); learned, and inert under the
        # deterministic evaluation policy.
        self.log_std = nn.Parameter(torch.full((action_size,), -1.0))

    # One forward pass returns the mean vector and the value scalar.
    def forward(self, obs: torch.Tensor) -> tuple:
        return self.pi(obs), self.v(obs).squeeze(-1)

    # Returns the action distribution for a given mean vector.
    def dist(self, mu: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(mu, self.log_std.exp())


# Log-probability over live columns only: a dead column cannot affect
# the world, so it carries no probability and sends no gradient.
def masked_logp(d: torch.distributions.Normal, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (d.log_prob(z) * mask).sum(-1)


# Scores become weights: dead columns forced to -1e9, then a float64
# softmax — masked columns come out exactly zero, sum exactly 1.
def masked_weights(z: torch.Tensor, mask: np.ndarray) -> np.ndarray:
    m = torch.as_tensor(mask, dtype=torch.bool)
    z = torch.where(m, z.double(), torch.tensor(-1e9, dtype=torch.float64))
    w = torch.softmax(z, dim=-1)
    return w.detach().numpy()


# The tradeable mask, plus an always-live CASH slot when the environment
# carries one; a mask that does not fill the action row raises.
def action_mask(info: dict, action_size: int) -> np.ndarray:
    live = info["tradeable"]
    cash = info["cash"]
    if len(live) + int(cash) != action_size:
        raise ValueError(f"{len(live)} assets with cash={cash} cannot fill "
                         f"{action_size} action columns")
    if cash:
        return np.concatenate([live, [True]])
    return live.copy()
