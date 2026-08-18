"""
portfolio/models/train.py — the PPO trainer for one agent.

Standard PPO with GAE, at the constants the agent's config file declares.
Episodes end by time limit, so every rollout tail bootstraps its final
observation's value (zeroing it would bias episode ends). Seeded end to
end, so a run is exactly reproducible; the scaler is fitted on the train
slice before the first environment is built.

Checkpoints: best-validation and final, selected by validation Sharpe
every eval_every updates. Each run owns a folder under
agent_runs/{sleeve}/{run_id}/: both checkpoints, the report, the figures.

    python -m portfolio.models.train --sleeve bonds --seed 1
    python -m portfolio.models.train --sleeve bonds --smoke
    # smoke = 3 updates, evaluated every update: the whole loop in miniature
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
from config import feature_registry as registry
from config import portfolio as cfg
from config.splits import get_split
from config.tickers import all_classes
from portfolio import metrics
from portfolio.models import scaling
from portfolio.models.environment import Environment
from portfolio.models.networks import (PolicyValue, action_mask, masked_logp, masked_weights)

AGENT_RUNS = Path(__file__).resolve().parents[2] / "agent_runs"

# Every run's settings come from its agent file (config/agents/{sleeve}.py);
# these are the legal keys per section, so a typo cannot silently vanish.
KEYS = {
    "NETWORK": {"hidden", "layers", "dropout"},
    "PPO": {"lr", "clip", "gamma", "gae_lambda", "epochs", "minibatches",
            "entropy_coef", "value_coef", "grad_clip", "weight_decay"},
    "ENV": {"clock", "cash", "band", "eta", "lam", "warmup", "episode_len"},
    "TRAIN": {"updates", "episodes_per_update", "eval_every", "seed"},
}
_SECTIONS = set(KEYS) | {"SLEEVE", "FEATURES", "ASSETS", "KAPPA",
                         "ANCHOR", "TAU", "DEPLOYED", "DEPLOYED_RUN"}

# Load and validate one agent's configuration module: exactly the legal
# sections, exactly the legal keys per section, and a FEATURES that is
# None or a non-empty list of registry names.
def agent_config(sleeve: str):
    import importlib

    # Load the agent file
    mod = importlib.import_module(f"config.agents.{sleeve}")
    if mod.SLEEVE != sleeve:
        raise ValueError(f"config/agents/{sleeve}.py declares SLEEVE = "f"{mod.SLEEVE!r}")
    stray = {k for k in vars(mod) if k.isupper()} - _SECTIONS

    if stray:
        raise ValueError(f"{sleeve} config: unknown names {sorted(stray)}")

    for section, legal in KEYS.items():
        got = set(getattr(mod, section))
        if got != legal:
            raise ValueError(f"{sleeve} config {section}: unknown "f"{sorted(got - legal)}, missing "f"{sorted(legal - got)}")

    if not hasattr(mod, "FEATURES"):
        raise ValueError(f"{sleeve} config: FEATURES missing "
                         f"(None derives the registry list)")

    f = mod.FEATURES
    if f is not None:
        if not isinstance(f, (list, tuple)) or not f:
            raise ValueError(f"{sleeve} config: FEATURES must be None or a "f"non-empty list")
        
        unknown = [n for n in f if n not in registry.REGISTRY]
        if unknown:
            raise ValueError(f"{sleeve} config FEATURES: not in the "f"registry {unknown[:5]}")

    if getattr(mod, "ASSETS", None) not in (None, "investable", "classes"):
        raise ValueError(f"{sleeve} config: ASSETS must be absent, None, "f"'investable' or 'classes'")

    kappa = getattr(mod, "KAPPA", None)
    if kappa is not None and not 0.0 < kappa <= 1.0:
        raise ValueError(f"{sleeve} config: KAPPA must be in (0, 1]")

    anchor = getattr(mod, "ANCHOR", None)
    if anchor is not None:
        from portfolio.models import anchors
        if anchor not in anchors.ANCHORS:
            raise ValueError(f"{sleeve} config: ANCHOR must be one of "
                             f"{anchors.ANCHORS}")
        if kappa is not None:
            raise ValueError(f"{sleeve} config: ANCHOR and KAPPA are "
                             f"mutually exclusive")
        tau = getattr(mod, "TAU", None)
        if tau is None or not 0.0 < tau <= 1.0:
            raise ValueError(f"{sleeve} config: an anchored agent needs "
                             f"TAU in (0, 1]")
    elif getattr(mod, "TAU", None) is not None:
        raise ValueError(f"{sleeve} config: TAU without ANCHOR")

    dep = getattr(mod, "DEPLOYED", None)
    if dep is not None and (
            not isinstance(dep, dict) or not dep
            or not all(isinstance(k, str) and isinstance(v, str)
                       for k, v in dep.items())):
        raise ValueError(f"{sleeve} config: DEPLOYED must be a dict of "f"sleeve -> run-id strings")

    run = getattr(mod, "DEPLOYED_RUN", None)
    if run is not None and not isinstance(run, str):
        raise ValueError(f"{sleeve} config: DEPLOYED_RUN must be a run-id "f"string")
    return mod


# The agent's asset universe: its own class by default; "investable"
# spans every class (the flat agent's world); "classes" is the four
# class names themselves — the allocator's synthetic assets.
def agent_assets(acfg) -> list:
    scope = getattr(acfg, "ASSETS", None)

    if scope == "investable":
        return [t for members in all_classes().values() for t in members]
    
    if scope == "classes":
        return list(all_classes())
    
    return all_classes()[acfg.SLEEVE]

# One walk-forward fold: expanding train window through the fold year,
# judged on the following year. Both must stay inside the train split —
# reaching past it would tune on validation.
def fold_windows(y: int) -> tuple:
    t0, t1 = get_split("train")
    train_w = (t0, f"{y}-12-31")
    val_w = (f"{y + 1}-01-01", f"{y + 1}-12-31")
    if val_w[1] > t1:
        raise ValueError(f"fold year {y + 1} reaches past the train split")
    return train_w, val_w

# Every asset feature scoped to this class (the registry scopes by class
# name or "all") plus the whole market block; "investable" takes the
# union, where out-of-class features ride as flagged blanks.
def feature_list(sleeve: str) -> list:
    if sleeve == "investable":
        names = [n for n in registry.REGISTRY
                 if registry.spec(n)["grain"] == "asset"]
    else:
        names = [n for n in registry.REGISTRY
                 if registry.spec(n)["grain"] == "asset"
                 and {sleeve, "all"} & set(registry.spec(n)["classes"])]
    if not names:
        raise ValueError(f"no asset features scoped to {sleeve!r}")
    names += [n for n in registry.REGISTRY
              if registry.spec(n)["grain"] == "market"]
    return names

# The observation an agent file resolves to, honouring ASSETS scope.
def _feature_scope(acfg) -> str:
    return ("investable" if getattr(acfg, "ASSETS", None) == "investable"
            else acfg.SLEEVE)

# The observation list an agent file resolves to: its FEATURES override
# when given, else every feature scoped to its sleeve. Trainer and tuner
# both resolve observations here, so the two can never differ.
def sleeve_features(acfg) -> list:
    if acfg.FEATURES is not None:
        return list(acfg.FEATURES)
    return feature_list(_feature_scope(acfg))

# The 8 merged sector buckets as one-hot static covariates: sectors below
# the member floor share one bucket. The map is 2025 GICS applied
# historically — config/tickers.py declares which names that affects.
def sector_static(assets: list) -> np.ndarray:
    from collections import Counter
    from config.tickers import EQUITIES, SECTOR_BUCKETS, SECTOR_MIN_MEMBERS
    counts = Counter(EQUITIES.values())
    big = sorted(s for s, n in counts.items() if n >= SECTOR_MIN_MEMBERS)
    buckets = big + ["other"]
    if len(buckets) != SECTOR_BUCKETS:
        raise ValueError(f"sector merge yields {len(buckets)} buckets, "
                         f"not {SECTOR_BUCKETS}")
    m = np.zeros((len(assets), len(buckets)), dtype=np.float32)
    for i, t in enumerate(assets):
        s = EQUITIES.get(t)
        if s is None:
            raise ValueError(f"no sector tag for {t!r}")
        m[i, buckets.index(s if s in big else "other")] = 1.0
    return m

# Generalised advantage estimation over one episode. Episodes end by
# truncation, so the tail bootstraps from last_value.
def gae(rewards: np.ndarray, values: np.ndarray, last_value: float, gamma: float, lam: float) -> tuple:
    T = len(rewards)
    adv = np.empty(T)
    running = 0.0
    nxt = last_value
    for t in range(T - 1, -1, -1):
        delta = rewards[t] + gamma * nxt - values[t]
        running = delta + gamma * lam * running
        adv[t] = running
        nxt = values[t]
    return adv, adv + values

# Thread count changes float reduction order
NUM_THREADS = cfg.NUM_THREADS


# Every RNG the run touches, plus the thread pin the seed only means
# something at.
def seed_everything(seed: int) -> None:
    torch.set_num_threads(NUM_THREADS)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Adam with decoupled weight decay on the weight MATRICES only. Decaying
# log_std drags exploration toward exp(0) = 1 — an arbitrary attractor,
# not a regulariser — and decayed biases shift outputs, not capacity.
def make_optimizer(net: PolicyValue, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, keep = [], []
    for name, p in net.named_parameters():
        (decay if name.endswith("weight") else keep).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": keep, "weight_decay": 0.0}], lr=lr)

# One rollout: episodes_per_update episodes under the current policy, the
# net in eval mode so GAE's value estimates are deterministic and no RNG
# is consumed. act_map transforms the EXECUTED weights only — a world
# transformation, so log-probabilities and the PPO ratio are untouched.
def collect(env: Environment, net: PolicyValue, h: dict, act_map=None) -> dict:
    net.eval()
    obs_l, mask_l, z_l, logp_l = [], [], [], []
    adv_l, ret_l, ep_rewards = [], [], []

    for _ in range(h["episodes_per_update"]):
        obs, info = env.reset()
        rewards, values = [], []
        done = False

        # One episode/one trading session per loop
        while not done:
            mask = action_mask(info, env.action_size)
            with torch.no_grad():
                t_obs = torch.as_tensor(obs, dtype=torch.float32)
                mu, v = net(t_obs)
                d = net.dist(mu)
                z = d.sample()
                logp = masked_logp(
                    d, z, torch.as_tensor(mask, dtype=torch.float32))

            # Calculate the executed weights
            w = masked_weights(z, mask)
            if act_map is not None:
                w = act_map(w, info)

            # Record the step's data for the rollout
            obs_l.append(obs)
            mask_l.append(mask)
            z_l.append(z.numpy())
            logp_l.append(float(logp))
            values.append(float(v))
            obs, r, done, info = env.step(w)
            rewards.append(r)

        # At the episode's end hold the last observation's value for GAE bootstrapping.
        with torch.no_grad():
            _, last_v = net(torch.as_tensor(obs, dtype=torch.float32))

        #Compute the advantages and returns for the episode, and record the total reward.
        adv, ret = gae(np.array(rewards), np.array(values), float(last_v), h["gamma"], h["gae_lambda"])
        adv_l.append(adv)
        ret_l.append(ret)
        ep_rewards.append(float(np.sum(rewards)))

    # Concatenate the episode advantages and normalise them.
    adv = np.concatenate(adv_l)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # Return the rollout data as a dictionary of NumPy arrays, ready for the PPO update.
    return {"obs": np.array(obs_l, dtype=np.float32),
            "mask": np.array(mask_l),
            "z": np.array(z_l, dtype=np.float32),
            "logp": np.array(logp_l, dtype=np.float32),
            "adv": adv.astype(np.float32),
            "ret": np.concatenate(ret_l).astype(np.float32),
            "ep_reward": float(np.mean(ep_rewards))}

# One PPO update over a rollout; returns the gauges. The gradient epochs
# are the ONLY train-mode region — elsewhere the net stays in eval mode so
# dropout never perturbs targets or consumes RNG.
def update(net: PolicyValue, opt: torch.optim.Optimizer, roll: dict, h: dict) -> dict:

    #Unpack batch
    obs = torch.as_tensor(roll["obs"])
    z = torch.as_tensor(roll["z"])
    old_logp = torch.as_tensor(roll["logp"])
    adv = torch.as_tensor(roll["adv"])
    ret = torch.as_tensor(roll["ret"])
    m = torch.as_tensor(roll["mask"], dtype=torch.float32)

    n = len(obs)
    kls, clipfracs = [], []
    net.train()

    for _ in range(h["epochs"]):
        for idx in torch.randperm(n).split(max(1, n // h["minibatches"])):

            # Compute the policy and value outputs
            mu, v = net(obs[idx])
            d = net.dist(mu)
            logp = masked_logp(d, z[idx], m[idx])
            ratio = (logp - old_logp[idx]).exp()
            s1 = ratio * adv[idx]
            s2 = ratio.clamp(1 - h["clip"], 1 + h["clip"]) * adv[idx]
            pi_loss = -torch.min(s1, s2).mean()
            v_loss = (v - ret[idx]).pow(2).mean()
            ent = d.entropy().sum(-1).mean()
            loss = (pi_loss + h["value_coef"] * v_loss - h["entropy_coef"] * ent)

            # Backpropagate and update the network parameters
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), h["grad_clip"])
            opt.step()

            # Compute the gauges for this minibatch
            with torch.no_grad():
                kls.append(float((old_logp[idx] - logp).mean()))
                clipfracs.append(float(((ratio - 1).abs() > h["clip"]).float().mean()))

    # Compute the explained variance of the value function on the entire rollout
    net.eval()
    with torch.no_grad():
        _, v_all = net(obs)
        var_ret = ret.var()
        ev = float(1 - (ret - v_all).var() / var_ret) if var_ret > 0 else 0.0

    return {"kl": float(np.mean(kls)), "clipfrac": float(np.mean(clipfracs)),
            "explained_var": ev, "pi_loss": float(pi_loss.detach()),
            "v_loss": float(v_loss.detach())}

# A network as a deterministic policy: masked mean actions. 
def det_policy(net: PolicyValue, action_size: int):
    net.eval()
    def policy(obs, info):
        mask = action_mask(info, action_size)
        with torch.no_grad():
            mu, _ = net(torch.as_tensor(obs, dtype=torch.float32))
        return masked_weights(mu, mask)
    return policy

# Equal weight over the session's tradeable assets, nothing on cash — the
# in-environment 1/N every agent is graded against.
def naive_policy(action_size: int):
    def policy(obs, info):
        live = info["tradeable"]
        a = np.zeros(action_size)
        a[:len(live)][live] = 1.0 / live.sum()
        return a
    return policy

# Executed weights = (1-kappa) * equal-weight prior + kappa * the
# policy's weights. The prior spans the live assets with nothing on
# CASH, and is a fixed point of the blend at every kappa.
def blend(w: np.ndarray, info: dict, kappa: float) -> np.ndarray:
    live = info["tradeable"]
    prior = np.zeros(len(w))
    prior[: len(live)][live] = 1.0 / live.sum()
    return (1.0 - kappa) * prior + kappa * w

# Wraps a policy so every weight vector it returns leaves through blend.
def blended(policy, kappa: float):
    def p(obs, info):
        return blend(policy(obs, info), info, kappa)
    return p

# Executed weights = normalize(anchor * exp(tau * tanh(a)))
# Used by the markowitz-anchored tilt agent
def tilt(w: np.ndarray, info: dict, A, tau: float) -> np.ndarray:
    i = A.index.searchsorted(info["date"], side="right") - 1
    if i < 0:
        raise ValueError(f"no anchor row at or before {info['date']}")
    anchor = A.iloc[i].to_numpy(dtype=float)
    live = np.ones(len(w), dtype=bool)
    live[: len(info["tradeable"])] = info["tradeable"]
    a = np.where(live, np.log(np.maximum(w, 1e-12)), 0.0)
    a = np.where(live, a - a[live].mean(), 0.0)
    out = np.where(live, anchor * np.exp(tau * np.tanh(a)), 0.0)
    s = out.sum()
    return out / s if s > 0 else anchor

# Wraps a policy so every weight vector it returns leaves through tilt.
def tilted(policy, A, tau: float):
    def p(obs, info):
        return tilt(policy(obs, info), info, A, tau)
    return p

# One frozen policy over one window: the summary every strategy is graded
# by, plus the full evaluate() output for equity paths.
def score(env: Environment, policy, bundle: dict, window: str | tuple = "val") -> tuple:
    out = env.evaluate(policy, window)
    rf = bundle["cash"].reindex(out["value"].index)
    return metrics.summary(out, rf, start_value=1.0), out

# Deterministic validation pass of a network, scored by the same metrics
# as every baseline.
def evaluate_policy(env: Environment, net: PolicyValue, bundle: dict, window: str = "val") -> dict:
    return score(env, det_policy(net, env.action_size), bundle, window)[0]

# A checkpoint: the weights, the optimiser state, and the run's meta.
def save(path: Path, net: PolicyValue, opt, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(), "meta": meta}, path)

# Rebuild a checkpoint's network from the shape its own meta records,
# ready to evaluate.
def load(path: Path) -> tuple:
    ck = torch.load(path, weights_only=False)
    meta = ck["meta"]
    nw = meta["network"]        # the checkpoint's sole architecture record
    net = PolicyValue(meta["obs_size"], meta["action_size"], nw["hidden"], nw["layers"], nw["dropout"])
    net.load_state_dict(ck["model"])
    net.eval()
    return net, meta

# Trains one agent end to end, returning its best validation Sharpe and
# run folder. The keyword arguments override the agent file for a single
# run; the run's meta carries the EFFECTIVE values. bundle injects a
# pre-built world (the allocator's synthetic assets); None loads the
# real one. extra_meta lands in the run's meta verbatim.
def train(sleeve: str, seed: int | None = None, updates: int | None = None,
          eval_every: int | None = None, clock: str | None = None,
          episodes_per_update: int | None = None, hidden: int | None = None,
          lam: float | None = None, static: str | None = None,
          dropout: float | None = None,
          weight_decay: float | None = None,
          bundle: dict | None = None,
          extra_meta: dict | None = None) -> dict:
    from datetime import datetime

    #Load agent file
    acfg = agent_config(sleeve)
    if getattr(acfg, "ASSETS", None) == "classes" and bundle is None:
        raise ValueError("ASSETS='classes' trades synthetic assets that "
                         "must be built first — pass bundle= (see "
                         "portfolio.models.allocate)")

    # The config's kappa blends executed weights toward the equal-weight
    # prior; 1.0 is the identity and leaves the action untouched.
    kappa = getattr(acfg, "KAPPA", None) or 1.0
    act_map = None if kappa == 1.0 else (
        lambda w, info: blend(w, info, kappa))

    # Extract the effective hyperparameters for this run, overriding the agent file when given.
    seed = acfg.TRAIN["seed"] if seed is None else seed
    updates = acfg.TRAIN["updates"] if updates is None else updates
    clock = acfg.ENV["clock"] if clock is None else clock
    hidden = acfg.NETWORK["hidden"] if hidden is None else hidden
    lam = acfg.ENV["lam"] if lam is None else lam
    dropout = acfg.NETWORK["dropout"] if dropout is None else dropout
    weight_decay = (acfg.PPO["weight_decay"] if weight_decay is None else weight_decay)

    # Set up the run folder
    run_id = f"{datetime.now():%Y-%m-%d_%H%M%S}_seed{seed}"
    run_dir = AGENT_RUNS / sleeve / run_id

    # if timestamp already exists, add a numeric suffix to avoid overwriting
    n = 1
    while run_dir.exists(): 
        n += 1
        run_dir = AGENT_RUNS / sleeve / f"{run_id}_{n}"
    run_dir.mkdir(parents=True)

    # Assembe th hyperparameter dict
    hypers = {**acfg.PPO, "weight_decay": weight_decay,
              "episodes_per_update": (
                  acfg.TRAIN["episodes_per_update"]
                  if episodes_per_update is None else episodes_per_update),
              "eval_every": (acfg.TRAIN["eval_every"] if eval_every is None
                             else eval_every)}

    # Set Seed
    seed_everything(seed)

    # Load the bundle (unless one was injected), the assets and features
    if bundle is None:
        from portfolio.run import load_bundle
        bundle = load_bundle()
    assets = agent_assets(acfg)
    feats = sleeve_features(acfg)

    # Fit the scaler on the train slice
    scaling.fit(sleeve, feats, assets)
    if static not in (None, "sectors"):
        raise ValueError(f"unknown static block {static!r}")

    # Build sector dataset
    st = sector_static(assets) if static == "sectors" else None

    # Initialise the environment
    env = Environment(assets, feats, clock=clock,
                      cash=acfg.ENV["cash"],
                      scaling_name=sleeve, window="train", seed=seed,
                      bundle=bundle, band=acfg.ENV["band"],
                      eta=acfg.ENV["eta"], lam=lam,
                      warmup=acfg.ENV["warmup"],
                      episode_len=acfg.ENV["episode_len"], static=st)

    # Build the network and optimizer, and record the run's meta
    net = PolicyValue(env.obs_size, env.action_size,
                      hidden, acfg.NETWORK["layers"], dropout)

    anchor_name = getattr(acfg, "ANCHOR", None)
    tau = getattr(acfg, "TAU", None)
    A = None
    if anchor_name is not None:
        from portfolio.models import anchors
        A = anchors.load(anchor_name, env.columns)
        act_map = lambda w, info: tilt(w, info, A, tau)
        with torch.no_grad():
            net.pi[-1].weight.zero_()
            net.pi[-1].bias.zero_()

    # Apply optimiser to the networks
    opt = make_optimizer(net, hypers["lr"], hypers["weight_decay"])

    # Run metadata
    meta = {"sleeve": sleeve, "seed": seed, "hypers": hypers,
            "network": {**dict(acfg.NETWORK), "hidden": hidden, "dropout": dropout},
            "env": {**dict(acfg.ENV), "clock": clock, "lam": lam},
            "static": static, "kappa": kappa,
            "anchor": anchor_name, "tau": tau,
            "features": feats, "scaling": sleeve,
            "obs_size": env.obs_size, "action_size": env.action_size,
            "torch_threads": NUM_THREADS, **(extra_meta or {})}

    # Set up tracking variables
    best = -np.inf
    tag = f"{sleeve}_seed{seed}"
    history = {"update": [], "ep_reward": [], "kl": [], "clipfrac": [],
               "ev": [], "eval_update": [], "eval_sharpe": []}

    # Train Loop
    for u in range(1, updates + 1):

        # Run a batch of episodes collect outpt and update the network
        roll = collect(env, net, hypers, act_map=act_map)
        g = update(net, opt, roll, hypers)

        # Record the gauges and print the update summary
        history["update"].append(u)
        history["ep_reward"].append(roll["ep_reward"])
        history["kl"].append(g["kl"])
        history["clipfrac"].append(g["clipfrac"])
        history["ev"].append(g["explained_var"])
        print(f"[{tag}] update {u:4d}  ep_reward {roll['ep_reward']:+8.2f}  "
              f"kl {g['kl']:+.4f}  clip {g['clipfrac']:.2f}  "
              f"ev {g['explained_var']:+.2f}")

        # Evaluate the policy on the validation set every eval_every updates, and save the best checkpoint
        if u % hypers["eval_every"] == 0:
            pol = det_policy(net, env.action_size)
            if kappa != 1.0:
                pol = blended(pol, kappa)
            if A is not None:
                pol = tilted(pol, A, tau)
            s = score(env, pol, bundle)[0]
            history["eval_update"].append(u)
            history["eval_sharpe"].append(s["sharpe"])
            print(f"[{tag}]   val sharpe {s['sharpe']:+.3f}  "
                  f"ann {s['annual_return'] * 100:+.2f}%  "
                  f"mdd {s['max_drawdown'] * 100:.1f}%")

            # Save the best checkpoint if the validation Sharpe is better than the previous best
            if np.isnan(s["sharpe"]):
                print(f"[{tag}]   val sharpe is NaN — not a checkpoint candidate")
            elif s["sharpe"] > best:
                best = s["sharpe"]
                save(run_dir / "best.pt", net, opt, {**meta, "update": u, "val_sharpe": s["sharpe"]})

    # Save the final checkpoint and write the run report
    save(run_dir / "final.pt", net, opt, {**meta, "update": updates})
    _write_run_report(run_dir, meta, history, env, bundle, net, kappa,
                      A, tau)
    print(f"[{tag}] run folder -> {run_dir}")

    return {"best_val_sharpe": best, "run_dir": str(run_dir)}

# The run's written record: best/final/naive on validation (train-window
# Sharpe alongside, so the overfit gap is visible), the figures, the raw
# history. The final policy is the live net; only best.pt needs a load.
def _write_run_report(run_dir: Path, meta: dict, history: dict, env: Environment, bundle: dict, net: PolicyValue, kappa: float = 1.0,
                      A=None, tau: float | None = None) -> None:
    from portfolio import report as rep

    # function to evaluate a policy on both the validation and training windows.
    def val_and_train(policy):
        s_v, out_v = score(env, policy, bundle, "val")
        s_t, _ = score(env, policy, bundle, "train")
        return s_v, s_t["sharpe"], out_v["value"]

    results, paths = {}, {}
    candidates = [("agent (final)", net.eval())]

    if (run_dir / "best.pt").exists():
        candidates.insert(0, ("agent (best)", load(run_dir / "best.pt")[0]))

    for name, n in candidates:
        pol = det_policy(n, env.action_size)
        if kappa != 1.0:
            pol = blended(pol, kappa)
        if A is not None:
            pol = tilted(pol, A, tau)
        s_v, ts, v = val_and_train(pol)
        results[name] = (s_v, ts)
        paths[name] = v

    s_v, ts, v = val_and_train(naive_policy(env.action_size))
    results["1/N in sleeve"] = (s_v, ts)
    paths["1/N in sleeve"] = v

    rep.write_all(run_dir, meta, results, history, paths)


if __name__ == "__main__":
    from collectors._core import console_utf8
    console_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleeve", required=True)
    ap.add_argument("--seed", type=int, default=None, help="overrides the agent file's TRAIN seed")
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        train(args.sleeve, seed=args.seed, updates=3, eval_every=1)
    else:
        train(args.sleeve, seed=args.seed, updates=args.updates)
