#!/usr/bin/env python3
"""Behaviour-level macro world model.

We treat the agent's success-from-a-state as a function of just the current
zone colour and the next target colour. For each ordered pair (a, b) with
a ≠ b, we estimate

    P(reach b | currently inside a) ≈ success rate of `F b` initialised with
    the agent placed inside an `a` zone.

We approximate "currently inside a" by running `F a THEN F b` tasks, and
counting cases where the agent reaches a (the intermediate) and asking
whether it then reaches b. This produces a 4×4 success matrix.

We then check whether this matrix predicts depth-2 and depth-3 success
on multi-step tasks via simple multiplication, comparing to empirical
multi-step success.

If the prediction is accurate, the agent's behaviour is well summarised by
a colour-state Markov model — i.e. the macro structure can be recovered from
behaviour alone, without postulating an internal world model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
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

COLOURS = ["blue", "green", "yellow", "magenta"]


def get_unwrapped(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def load_model(exp: str, env_name: str = "PointLtl2-v0"):
    sampler = CurriculumSampler.partial(curricula[env_name])
    env = make_env(env_name, sampler, sequence=True)
    cfg = model_configs[env_name]
    store = ModelStore(env_name, exp, seed=0)
    status = store.load_training_status(map_location="cpu")
    store.load_vocab()
    model = build_model(env, status, cfg)
    model.eval()
    return model, list(env.get_propositions())


def make_task_env(propositions, colours: list[str], seed: int):
    """Build a task with the given subgoal sequence (each step is a single colour)."""
    base = sg_make("PointLtl2-v0")
    env = SafetyGymWrapper(base)
    env = FlattenObservation(env)
    pairs = []
    for c in colours:
        reach = frozenset([Assignment.single_proposition(c, propositions).to_frozen()])
        pairs.append((reach, frozenset()))
    task = LDBASequence(pairs)

    def sampler(_):
        return lambda: task

    env = SequenceWrapper(env, sampler(propositions), step_penalty=0.0)
    env = TimeLimit(env, max_episode_steps=200 * len(colours))
    env = RemoveTruncWrapper(env)
    return env


def run_episode(model, propositions, env, max_steps: int):
    out = env.reset(seed=None)
    obs = out[0] if isinstance(out, tuple) else out
    visited = []  # ordered list of zone colours visited
    last = None
    for _ in range(max_steps):
        preprocessed = preprocessing.preprocess_obss([obs], set(propositions))
        dist, _v = model(preprocessed)
        dist.set_epsilon_mask(preprocessed.epsilon_mask)
        action = dist.mode.numpy().flatten()
        obs, _r, done, info = env.step(action)
        unwrapped = get_unwrapped(env)
        agent_pos = unwrapped.agent_pos[:2]
        for name, pos in unwrapped.zone_positions.items():
            if np.linalg.norm(np.array(pos[:2]) - agent_pos) < 0.4:
                for c in COLOURS:
                    if c in name:
                        if last != c:
                            visited.append(c)
                            last = c
                        break
        if done:
            break
    return visited, bool(info.get("success", False))


def estimate_pair_success(model, propositions, source: str, target: str,
                           n_episodes: int) -> tuple[int, int]:
    """Run F source THEN F target episodes; return (n_reached_source_then_target, n_reached_source).

    Approximates P(reach target | reached source).
    """
    reached_source = 0
    reached_target = 0
    for ep in range(n_episodes):
        env = make_task_env(propositions, [source, target], seed=2000 + ep)
        out = env.reset(seed=2000 + ep)
        obs = out[0] if isinstance(out, tuple) else out
        saw_source = False
        saw_target_after = False
        for _ in range(400):
            preprocessed = preprocessing.preprocess_obss([obs], set(propositions))
            dist, _v = model(preprocessed)
            dist.set_epsilon_mask(preprocessed.epsilon_mask)
            action = dist.mode.numpy().flatten()
            obs, _r, done, info = env.step(action)
            unwrapped = get_unwrapped(env)
            ap = unwrapped.agent_pos[:2]
            for name, pos in unwrapped.zone_positions.items():
                d = np.linalg.norm(np.array(pos[:2]) - ap)
                if d < 0.4:
                    if not saw_source and source in name:
                        saw_source = True
                    elif saw_source and target in name:
                        saw_target_after = True
            if done:
                break
        env.close()
        if saw_source:
            reached_source += 1
            if saw_target_after:
                reached_target += 1
    return reached_target, reached_source


def estimate_depth_success(model, propositions, depth: int, n_episodes: int) -> tuple[int, int]:
    """Run random length-`depth` task chains, return (successes, n)."""
    rng = np.random.RandomState(123)
    successes = 0
    n_total = 0
    for ep in range(n_episodes):
        # pick distinct colours for the chain
        chain = list(rng.choice(COLOURS, size=depth, replace=False))
        env = make_task_env(propositions, chain, seed=3000 + ep)
        try:
            out = env.reset(seed=3000 + ep)
        except Exception:
            env.close()
            continue
        obs = out[0] if isinstance(out, tuple) else out
        for _ in range(200 * depth):
            preprocessed = preprocessing.preprocess_obss([obs], set(propositions))
            dist, _v = model(preprocessed)
            dist.set_epsilon_mask(preprocessed.epsilon_mask)
            action = dist.mode.numpy().flatten()
            obs, _r, done, info = env.step(action)
            if done:
                break
        env.close()
        n_total += 1
        if info.get("success", False):
            successes += 1
    return successes, n_total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="fresh_baseline")
    p.add_argument("--n_per_pair", type=int, default=15)
    p.add_argument("--n_per_depth", type=int, default=40)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir or f"interpretability/results/macro_world_model/{args.exp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.exp}")
    model, propositions = load_model(args.exp)

    # 1. Pair-success matrix
    print("\n[1/2] Estimating pair-success matrix...")
    matrix = np.full((len(COLOURS), len(COLOURS)), np.nan)
    counts = np.zeros((len(COLOURS), len(COLOURS), 2), dtype=int)
    for i, a in enumerate(tqdm(COLOURS, desc="rows")):
        for j, b in enumerate(COLOURS):
            if a == b:
                continue
            n_succ, n_reached = estimate_pair_success(model, propositions, a, b, args.n_per_pair)
            counts[i, j, 0] = n_succ
            counts[i, j, 1] = n_reached
            if n_reached > 0:
                matrix[i, j] = n_succ / n_reached

    print("\nPair-success matrix P(reach b | reached a):")
    header = "         " + "".join(f"{c:>10}" for c in COLOURS)
    print(header)
    for i, a in enumerate(COLOURS):
        row = f"{a:>8} "
        for j, b in enumerate(COLOURS):
            v = matrix[i, j]
            row += f"{'   --   ' if np.isnan(v) else f'{v:>10.3f}'}"
        print(row)

    # 2. Depth-1/2/3 prediction
    print("\n[2/2] Comparing predicted vs empirical depth-k success...")
    depth_rows = []
    for depth in [1, 2, 3]:
        succ, n = estimate_depth_success(model, propositions, depth, args.n_per_depth)
        empirical = succ / max(n, 1) if n > 0 else float("nan")

        # Predict by averaging path products across distinct chains of length `depth`
        chain_probs = []
        for c0 in COLOURS:
            for c1 in COLOURS:
                if c1 == c0:
                    continue
                if depth == 1:
                    # single-step success from a fresh start: take the diagonal-of-row average
                    # we don't have it directly; use the pair matrix as a proxy by averaging incoming
                    # Actually depth=1 means just `F c1`. We approximate this with the average
                    # P(reach c1 | starting somewhere), i.e. average column j over rows i.
                    pass
                if depth == 2:
                    p = matrix[COLOURS.index(c0), COLOURS.index(c1)]
                    if np.isfinite(p):
                        chain_probs.append(p)
                if depth == 3:
                    for c2 in COLOURS:
                        if c2 in (c0, c1):
                            continue
                        a = matrix[COLOURS.index(c0), COLOURS.index(c1)]
                        b = matrix[COLOURS.index(c1), COLOURS.index(c2)]
                        if np.isfinite(a) and np.isfinite(b):
                            chain_probs.append(a * b)

        if depth == 1:
            # column-average for predicted (proxy)
            predicted = np.nanmean(matrix)
        else:
            predicted = float(np.mean(chain_probs)) if chain_probs else float("nan")

        depth_rows.append({"depth": depth, "predicted": predicted,
                           "empirical": empirical, "n_episodes": n})
        print(f"  depth {depth}: predicted={predicted:.3f}  empirical={empirical:.3f}  (N={n})")

    # Save results
    out = {
        "exp": args.exp,
        "args": vars(args),
        "colours": COLOURS,
        "pair_success_matrix": matrix.tolist(),
        "pair_counts": counts.tolist(),
        "depth_rows": depth_rows,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(out, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    im = ax.imshow(matrix, cmap="YlGn", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(len(COLOURS)))
    ax.set_xticklabels(COLOURS)
    ax.set_yticks(range(len(COLOURS)))
    ax.set_yticklabels(COLOURS)
    ax.set_xlabel("next colour")
    ax.set_ylabel("current colour")
    for i in range(len(COLOURS)):
        for j in range(len(COLOURS)):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", color="gray")
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black" if v > 0.4 else "darkgreen")
    plt.colorbar(im, ax=ax)
    ax.set_title("P(reach next | reached current)")

    ax = axes[1]
    depths = [r["depth"] for r in depth_rows]
    preds = [r["predicted"] for r in depth_rows]
    emps = [r["empirical"] for r in depth_rows]
    x = np.arange(len(depths))
    width = 0.35
    ax.bar(x - width / 2, preds, width, label="predicted (Markov chain)", color="#2980b9")
    ax.bar(x + width / 2, emps, width, label="empirical", color="#27ae60")
    for i, (p, e) in enumerate(zip(preds, emps)):
        ax.text(i - width / 2, (p if np.isfinite(p) else 0) + 0.02,
                f"{p:.2f}" if np.isfinite(p) else "—", ha="center", fontsize=9)
        ax.text(i + width / 2, (e if np.isfinite(e) else 0) + 0.02,
                f"{e:.2f}" if np.isfinite(e) else "—", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"depth {d}" for d in depths])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate")
    ax.set_title("Markov-chain prediction vs. empirical multi-step success")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_dir / "macro_world_model.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
