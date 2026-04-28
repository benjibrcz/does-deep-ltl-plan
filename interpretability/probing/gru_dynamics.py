#!/usr/bin/env python3
"""Linear next-state probe on the combined embedding.

For each rollout step t, capture h_t = compute_embedding(obs_t). The probe
question is: how predictable is Δh_t = h_{t+1} − h_t from h_t alone?

We compare:
  - Linear f(h_t) → Δh
  - Linear f(h_t, action_t) → Δh

We additionally break results down into:
  - Within-link steps: the LTL embedding does NOT change between t and t+1
    (the agent is still pursuing the same subgoal).
  - Goal-switch steps: the LTL embedding does change.

If next-state predictability drops sharply at goal switches, the network's
latent dynamics are essentially the "stable state machine that retargets at
switches" picture from the PDF.

Splits: episode-disjoint.
Output: results/gru_dynamics/<exp>/results.json + bar chart.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from tqdm import tqdm

sys.path.insert(0, "src")

import preprocessing
from config import model_configs
from envs import make_env
from envs.remove_trunc_wrapper import RemoveTruncWrapper
from envs.seq_wrapper import SequenceWrapper
from envs.zones.safety_gym_wrapper import SafetyGymWrapper
from gymnasium.wrappers import FlattenObservation, TimeLimit
from ltl.automata import LDBASequence
from ltl.logic import Assignment
from model.model import build_model
from safety_gymnasium.utils.registration import make as sg_make
from sequence.samplers import CurriculumSampler, curricula
from utils.model_store import ModelStore

torch.set_grad_enabled(False)


def load_model(exp: str, env_name: str = "PointLtl2-v0"):
    sampler = CurriculumSampler.partial(curricula[env_name])
    env = make_env(env_name, sampler, sequence=True)
    cfg = model_configs[env_name]
    store = ModelStore(env_name, exp, seed=0)
    status = store.load_training_status(map_location="cpu")
    store.load_vocab()
    model = build_model(env, status, cfg)
    model.eval()
    propositions = list(env.get_propositions())
    return model, propositions, env


COLOURS = ["blue", "green", "yellow", "magenta"]
COLOUR_PAIRS = [(a, b) for a in COLOURS for b in COLOURS if a != b]


def make_two_step_env(int_colour: str, goal_colour: str, propositions, seed: int):
    """Build PointLtl2-v0 with task F int_colour THEN F goal_colour."""
    base = sg_make("PointLtl2-v0")
    env = SafetyGymWrapper(base)
    env = FlattenObservation(env)
    int_reach = frozenset([Assignment.single_proposition(int_colour, propositions).to_frozen()])
    goal_reach = frozenset([Assignment.single_proposition(goal_colour, propositions).to_frozen()])
    task = LDBASequence([(int_reach, frozenset()), (goal_reach, frozenset())])

    def sampler(_props):
        return lambda: task

    env = SequenceWrapper(env, sampler(propositions), step_penalty=0.0)
    env = TimeLimit(env, max_episode_steps=400)
    env = RemoveTruncWrapper(env)
    return env


def collect(model, propositions, _env_unused, n_episodes: int, max_steps: int):
    """Roll out 2-step tasks. Cycle through coloured pairs to vary the task."""
    rows = defaultdict(list)
    for ep in tqdm(range(n_episodes), desc="rollouts"):
        int_c, goal_c = COLOUR_PAIRS[ep % len(COLOUR_PAIRS)]
        env = make_two_step_env(int_c, goal_c, propositions, seed=42 + ep)
        out = env.reset(seed=42 + ep)
        obs = out[0] if isinstance(out, tuple) else out

        prev_h = None
        prev_ltl = None
        prev_action = None

        for step in range(max_steps):
            preprocessed = preprocessing.preprocess_obss([obs], set(propositions))
            env_emb = model.compute_env_embedding(preprocessed)
            ltl_emb = model.ltl_net(preprocessed.seq)
            h = torch.cat([env_emb, ltl_emb], dim=1)

            h_np = h.detach().numpy().flatten()
            ltl_np = ltl_emb.detach().numpy().flatten()

            if prev_h is not None:
                rows["h_t"].append(prev_h)
                rows["h_tp1"].append(h_np)
                rows["delta_h"].append(h_np - prev_h)
                rows["delta_ltl"].append(ltl_np - prev_ltl)
                rows["action_t"].append(prev_action)
                # switch = LTL embedding moved between t and t+1
                rows["is_switch"].append(int(not np.allclose(prev_ltl, ltl_np, atol=1e-6)))
                rows["episode"].append(ep)

            dist = model.actor(h)
            dist.set_epsilon_mask(preprocessed.epsilon_mask)
            action = dist.mode.numpy().flatten()

            prev_h = h_np
            prev_ltl = ltl_np
            prev_action = action.copy()

            obs, _r, done, _info = env.step(action)
            if done:
                break
        env.close()

    return {k: np.array(v) for k, v in rows.items()}


def episode_disjoint_mask(episodes: np.ndarray, frac: float = 0.2, seed: int = 42):
    rng = np.random.RandomState(seed)
    unique = np.unique(episodes)
    rng.shuffle(unique)
    n_test = max(1, int(frac * len(unique)))
    test = set(unique[:n_test].tolist())
    return np.array([e in test for e in episodes])


def fit_ridge(X: np.ndarray, Y: np.ndarray, mask: np.ndarray):
    m = Ridge(alpha=1.0).fit(X[~mask], Y[~mask])
    pred = m.predict(X[mask])
    # Multivariate R^2: pooled across output dims (variance-weighted)
    return float(r2_score(Y[mask], pred, multioutput="variance_weighted"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="fresh_baseline")
    p.add_argument("--n_episodes", type=int, default=80)
    p.add_argument("--max_steps", type=int, default=120)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir or f"interpretability/results/gru_dynamics/{args.exp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.exp}")
    model, propositions, env = load_model(args.exp)

    print("\n[1/3] Rolling out episodes...")
    data = collect(model, propositions, env, args.n_episodes, args.max_steps)
    n_states = len(data["h_t"])
    n_switches = int(data["is_switch"].sum())
    print(f"  Collected {n_states} (h_t, h_tp1) pairs, {n_switches} switches")

    h_t = data["h_t"]
    delta_h = data["delta_h"]
    delta_ltl = data["delta_ltl"]
    a_t = data["action_t"]
    is_switch = data["is_switch"].astype(bool)
    eps = data["episode"]
    mask = episode_disjoint_mask(eps)

    print("\n[2/3] Fitting Ridge probes...")

    def r2_for(X, Y, sub_mask=None):
        if sub_mask is None:
            return fit_ridge(X, Y, mask)
        # Restrict both train and test to rows where sub_mask is True
        idx = np.where(sub_mask)[0]
        X_sub = X[idx]
        Y_sub = Y[idx]
        m_sub = mask[idx]
        if (~m_sub).sum() < 50 or m_sub.sum() < 5:
            return float("nan")
        return fit_ridge(X_sub, Y_sub, m_sub)

    Xh = h_t
    Xha = np.concatenate([h_t, a_t], axis=1)

    # We track two targets:
    #   - Δh:   change in the full combined embedding (env + LTL).
    #   - Δltl: change in the LTL embedding alone (this is the "GRU latent
    #           state" of the planner). It is constant during pursuit of a
    #           subgoal and jumps at goal switches.
    def block(target):
        return {
            "overall": {
                "f(h)":         r2_for(Xh, target),
                "f(h, action)": r2_for(Xha, target),
            },
            "during_link": {
                "f(h)":         r2_for(Xh, target, sub_mask=~is_switch),
                "f(h, action)": r2_for(Xha, target, sub_mask=~is_switch),
            },
            "at_switch": {
                "f(h)":         r2_for(Xh, target, sub_mask=is_switch),
                "f(h, action)": r2_for(Xha, target, sub_mask=is_switch),
            },
        }

    results = {
        "delta_h":   block(delta_h),
        "delta_ltl": block(delta_ltl),
    }

    # Baselines for Δltl (zero target)
    r2_zero = float(r2_score(delta_ltl[mask], np.zeros_like(delta_ltl[mask]),
                             multioutput="variance_weighted"))
    Ymean = np.tile(delta_ltl[~mask].mean(axis=0), (mask.sum(), 1))
    r2_mean = float(r2_score(delta_ltl[mask], Ymean, multioutput="variance_weighted"))
    results["baselines_ltl"] = {
        "Δltl ≡ 0":            r2_zero,
        "Δltl ≡ mean(train)":  r2_mean,
    }

    print("\nResults:")
    for target, blocks in results.items():
        print(f"\n  [{target}]")
        if isinstance(blocks, dict) and any(isinstance(v, dict) for v in blocks.values()):
            for cond, scores in blocks.items():
                print(f"    {cond}:")
                for name, score in scores.items():
                    print(f"      {name:<20}  R²={score:.3f}")
        else:
            for name, score in blocks.items():
                print(f"    {name:<20}  R²={score:.3f}")

    print(f"\n  Switch fraction: {n_switches}/{n_states} = {n_switches/n_states:.3f}")

    out = {
        "exp": args.exp,
        "args": vars(args),
        "n_states": int(n_states),
        "n_switches": int(n_switches),
        "results": results,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(out, f, indent=2)

    # Plot Δltl (the cleaner test of "stable state machine")
    print("\n[3/3] Plotting...")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    conds = ["during_link", "at_switch"]
    pretty = {"during_link": "Within-link\n(Δltl ≈ 0)", "at_switch": "At goal switch"}
    x = np.arange(len(conds))
    width = 0.4
    target_blocks = results["delta_ltl"]
    h_only = [target_blocks[c]["f(h)"] for c in conds]
    h_act = [target_blocks[c]["f(h, action)"] for c in conds]
    ax.bar(x - width / 2, h_only, width, label="f(h)",
           color="#2980b9", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, h_act, width, label="f(h, action)",
           color="#27ae60", edgecolor="black", linewidth=0.5)
    for i, (a, b) in enumerate(zip(h_only, h_act)):
        ya = a if np.isfinite(a) else 0
        yb = b if np.isfinite(b) else 0
        ax.text(i - width / 2, ya + 0.02, f"{a:.2f}" if np.isfinite(a) else "—",
                ha="center", fontsize=9)
        ax.text(i + width / 2, yb + 0.02, f"{b:.2f}" if np.isfinite(b) else "—",
                ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[c] for c in conds])
    ax.set_ylabel(r"Predicting Δltl_emb — variance-weighted R²")
    ax.set_ylim(-0.2, 1.05)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("LTL-embedding dynamics: stable during pursuit (Δltl ≈ 0),\n"
                 "harder to predict at goal switches",
                 fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "gru_dynamics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
