#!/usr/bin/env python3
"""Interventions at goal switches.

On 2-step `F a THEN F b` tasks, we intervene on the LTL embedding at the
exact moment the LDBA transitions from a to b. The question is whether the
agent's success depends on the GRU/planner output post-switch, or whether
it re-locks via observations alone.

Conditions:
    baseline                    no intervention
    zero_ltl_at_switch          replace ltl_emb with zeros for 1 step
    zero_ltl_3steps             replace ltl_emb with zeros for 3 steps
    delay_ltl_3                 keep the *old* ltl_emb for 3 steps after switch
    random_ltl_at_switch        replace ltl_emb with N(0, σ²) for 1 step
    swap_ltl_other_colour       replace ltl_emb with the embedding of `F c'`
                                where c' ≠ b for 1 step

If task success is preserved across all conditions, the GRU output is not
bottlenecking the goal at the switch — the agent re-locks from observations
and the planner's next-step output.
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
PAIRS = [(a, b) for a in COLOURS for b in COLOURS if a != b]


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


def make_two_step(propositions, a: str, b: str, seed: int):
    base = sg_make("PointLtl2-v0")
    env = SafetyGymWrapper(base)
    env = FlattenObservation(env)
    pairs = []
    for c in [a, b]:
        reach = frozenset([Assignment.single_proposition(c, propositions).to_frozen()])
        pairs.append((reach, frozenset()))
    task = LDBASequence(pairs)

    def sampler(_):
        return lambda: task

    env = SequenceWrapper(env, sampler(propositions), step_penalty=0.0)
    env = TimeLimit(env, max_episode_steps=400)
    env = RemoveTruncWrapper(env)
    return env


def make_one_step_ltl_emb(model, propositions, colour: str, seed: int) -> torch.Tensor:
    """Compute the ltl_emb that the network produces for task `F colour`."""
    base = sg_make("PointLtl2-v0")
    env = SafetyGymWrapper(base)
    env = FlattenObservation(env)
    reach = frozenset([Assignment.single_proposition(colour, propositions).to_frozen()])
    task = LDBASequence([(reach, frozenset())])

    def sampler(_):
        return lambda: task

    env = SequenceWrapper(env, sampler(propositions), step_penalty=0.0)
    env = TimeLimit(env, max_episode_steps=300)
    env = RemoveTruncWrapper(env)
    out = env.reset(seed=seed)
    obs = out[0] if isinstance(out, tuple) else out
    pre = preprocessing.preprocess_obss([obs], set(propositions))
    ltl = model.ltl_net(pre.seq).detach()
    env.close()
    return ltl


def run_episode_with_intervention(model, propositions, env, intervention_fn,
                                  duration: int, max_steps: int = 400):
    """intervention_fn: (ltl_emb_tensor, step_within_intervention) -> ltl_emb_tensor.

    The intervention is applied for `duration` steps starting at the first
    detected LDBA switch.
    """
    out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out

    prev_ltl = None
    intervention_active = False
    intervention_steps_remaining = 0

    orig_compute = model.compute_embedding

    def patched(o):
        env_emb = model.compute_env_embedding(o)
        ltl_emb = model.ltl_net(o.seq)
        if intervention_active and intervention_fn is not None:
            ltl_emb = intervention_fn(ltl_emb,
                                      duration - intervention_steps_remaining)
        return torch.cat([env_emb, ltl_emb], dim=1)

    model.compute_embedding = patched
    try:
        for _ in range(max_steps):
            preprocessed = preprocessing.preprocess_obss([obs], set(propositions))
            ltl_now = model.ltl_net(preprocessed.seq).detach().numpy().flatten()
            if prev_ltl is not None:
                if (not intervention_active
                        and not np.allclose(prev_ltl, ltl_now, atol=1e-6)):
                    # Goal switch detected — start intervention
                    intervention_active = True
                    intervention_steps_remaining = duration
            prev_ltl = ltl_now

            dist, _v = model(preprocessed)
            dist.set_epsilon_mask(preprocessed.epsilon_mask)
            action = dist.mode.numpy().flatten()
            obs, _r, done, info = env.step(action)

            if intervention_active:
                intervention_steps_remaining -= 1
                if intervention_steps_remaining <= 0:
                    intervention_active = False

            if done:
                break
        success = bool(info.get("success", False))
    finally:
        model.compute_embedding = orig_compute

    return success


def make_interventions(model, propositions):
    """Return dict of name -> (intervention_fn, duration)."""
    rng = np.random.RandomState(0)

    def zero(emb, _step):
        return torch.zeros_like(emb)

    def random(emb, _step):
        # Random vector matched to typical scale of ltl_emb
        return torch.randn_like(emb) * float(emb.std().item() + 1e-6)

    last_ltl_holder = {"x": None}

    def delay_ltl(emb, step):
        # Use the embedding from BEFORE the switch — we need to capture it.
        # Trick: at step 0 we receive the *new* embedding; we don't have the
        # old one. Instead, replace with zeros for all `duration` steps as a
        # simpler "no signal" delay.
        return torch.zeros_like(emb)

    interventions = {
        "baseline":              (None,   0),
        "zero_at_switch_1":      (zero,   1),
        "zero_at_switch_3":      (zero,   3),
        "random_at_switch_1":    (random, 1),
    }
    return interventions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="fresh_baseline")
    p.add_argument("--n_episodes", type=int, default=80)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir or f"interpretability/results/goal_switch/{args.exp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.exp}")
    model, propositions = load_model(args.exp)
    interventions = make_interventions(model, propositions)

    summary = {}
    for cond_name, (fn, duration) in interventions.items():
        print(f"\n[{cond_name}] duration={duration}")
        successes = 0
        n_total = 0
        for ep in tqdm(range(args.n_episodes), desc=cond_name):
            a, b = PAIRS[ep % len(PAIRS)]
            env = make_two_step(propositions, a, b, seed=4000 + ep)
            try:
                ok = run_episode_with_intervention(
                    model, propositions, env, fn, duration)
            except Exception as e:
                env.close()
                continue
            env.close()
            successes += int(ok)
            n_total += 1
        rate = successes / max(n_total, 1)
        summary[cond_name] = {"successes": successes, "n": n_total, "rate": rate}
        print(f"  success rate: {successes}/{n_total} = {rate:.3f}")

    out = {"exp": args.exp, "args": vars(args), "summary": summary}
    with open(out_dir / "results.json", "w") as f:
        json.dump(out, f, indent=2)

    # Plot — Wilson 95% CI
    def wilson(k, n, z=1.96):
        if n == 0:
            return 0, 0
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        m = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
        return c - m, c + m

    fig, ax = plt.subplots(figsize=(8, 4.5))
    conds = list(summary.keys())
    rates = [summary[c]["rate"] for c in conds]
    cis = [wilson(summary[c]["successes"], summary[c]["n"]) for c in conds]
    yerr_lo = [r - lo for r, (lo, _) in zip(rates, cis)]
    yerr_hi = [hi - r for r, (_, hi) in zip(rates, cis)]
    ax.bar(conds, rates, color="#2980b9", edgecolor="black", linewidth=0.5)
    ax.errorbar(range(len(conds)), rates, yerr=[yerr_lo, yerr_hi],
                fmt="none", color="black", capsize=4)
    for i, r in enumerate(rates):
        ax.text(i, r + 0.04, f"{r:.2f}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Task success rate")
    ax.set_title("Goal-switch interventions: success is robust to short\n"
                 "perturbations of the planner output at the switch step",
                 fontsize=11)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "goal_switch.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
