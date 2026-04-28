#!/usr/bin/env python3
"""Maximum Entropy Goal-directedness (MEG).

For each colour goal `F c`, estimate

    p_pi   = success rate under the trained agent
    p_rand = success rate under a uniform-random policy
    GD(c)  = log(p_pi) − log(p_rand)

Larger GD ⇒ the agent's success on goal c is much more frequent than chance,
which is the natural quantitative reading of "goal-conditioned-ness".

This is a sanity check. The point is not that GD is high (it always will be
for any working agent), but to put a number behind the qualitative statement.
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
from envs.remove_trunc_wrapper import RemoveTruncWrapper
from envs.seq_wrapper import SequenceWrapper
from envs.zones.safety_gym_wrapper import SafetyGymWrapper
from envs import make_env
from gymnasium.wrappers import FlattenObservation, TimeLimit
from ltl.automata import LDBASequence
from ltl.logic import Assignment
from model.model import build_model
from safety_gymnasium.utils.registration import make as sg_make
from sequence.samplers import CurriculumSampler, curricula
from utils.model_store import ModelStore

torch.set_grad_enabled(False)
COLOURS = ["blue", "green", "yellow", "magenta"]


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


def make_one_step(propositions, c: str):
    base = sg_make("PointLtl2-v0")
    env = SafetyGymWrapper(base)
    env = FlattenObservation(env)
    reach = frozenset([Assignment.single_proposition(c, propositions).to_frozen()])
    task = LDBASequence([(reach, frozenset())])

    def sampler(_):
        return lambda: task

    env = SequenceWrapper(env, sampler(propositions), step_penalty=0.0)
    env = TimeLimit(env, max_episode_steps=300)
    env = RemoveTruncWrapper(env)
    return env


def run_agent(model, propositions, env, max_steps: int = 300) -> bool:
    out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    for _ in range(max_steps):
        preprocessed = preprocessing.preprocess_obss([obs], set(propositions))
        dist, _v = model(preprocessed)
        dist.set_epsilon_mask(preprocessed.epsilon_mask)
        action = dist.mode.numpy().flatten()
        obs, _r, done, info = env.step(action)
        if done:
            return bool(info.get("success", False))
    return False


def run_random(env, max_steps: int = 300) -> bool:
    out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    rng = np.random.RandomState()
    for _ in range(max_steps):
        # Continuous action ∈ [-1, 1]^2
        action = rng.uniform(-1.0, 1.0, size=2)
        obs, _r, done, info = env.step(action)
        if done:
            return bool(info.get("success", False))
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="fresh_baseline")
    p.add_argument("--n_episodes_per_goal", type=int, default=20)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir or f"interpretability/results/meg/{args.exp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.exp}")
    model, propositions = load_model(args.exp)

    rows = []
    print("\nEstimating p_pi (trained policy) and p_rand (uniform random)...")
    for c in COLOURS:
        succ_pi, succ_rand = 0, 0
        for ep in tqdm(range(args.n_episodes_per_goal), desc=f"F {c}"):
            env = make_one_step(propositions, c)
            ok = run_agent(model, propositions, env)
            env.close()
            succ_pi += int(ok)

            env_r = make_one_step(propositions, c)
            ok_r = run_random(env_r)
            env_r.close()
            succ_rand += int(ok_r)

        n = args.n_episodes_per_goal
        # Smooth with +1/+2 to avoid log(0)
        p_pi = (succ_pi + 1) / (n + 2)
        p_rand = (succ_rand + 1) / (n + 2)
        gd = float(np.log(p_pi) - np.log(p_rand))
        rows.append({
            "goal": c,
            "p_pi_raw": succ_pi / n,
            "p_rand_raw": succ_rand / n,
            "p_pi_smoothed": p_pi,
            "p_rand_smoothed": p_rand,
            "GD": gd,
            "succ_pi": succ_pi,
            "succ_rand": succ_rand,
            "n": n,
        })
        print(f"  F {c}: p_pi={succ_pi}/{n}={succ_pi/n:.2f}  p_rand={succ_rand}/{n}={succ_rand/n:.2f}  GD={gd:.2f}")

    mean_gd = float(np.mean([r["GD"] for r in rows]))
    print(f"\n  MEAN GD = {mean_gd:.2f}")

    out = {"exp": args.exp, "args": vars(args), "rows": rows, "mean_GD": mean_gd}
    with open(out_dir / "results.json", "w") as f:
        json.dump(out, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(COLOURS))
    width = 0.35
    p_pi = [r["p_pi_raw"] for r in rows]
    p_rand = [r["p_rand_raw"] for r in rows]
    ax.bar(x - width / 2, p_pi, width, color="#27ae60", label="Trained agent",
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, p_rand, width, color="#95a5a6", label="Uniform random",
           edgecolor="black", linewidth=0.5)
    for i, (a, b) in enumerate(zip(p_pi, p_rand)):
        ax.text(i - width / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=9)
        ax.text(i + width / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"F {c}" for c in COLOURS])
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"MEG: success rate per goal, trained vs. random\n"
                 f"Mean GD = {mean_gd:.2f}",
                 fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "meg.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
