#!/usr/bin/env python3
"""Subgoal steering: MLP-level vector vs. sequence-encoder swap.

Two interventions targeting the same outcome — change which colour the agent
pursues first — applied at two different points in the network:

1. **MLP steering.** Train a logistic-regression probe on actor.enc layer
   activations to predict the goal-colour id. Use the difference between
   probe class weights as a "steering vector" for source→target colour.
   Add α · w to the activation at every forward pass and observe whether
   the agent's first visited zone matches the target colour.

2. **Sequence-encoder swap.** Replace the LTL embedding with the embedding
   of a different task — `F target_colour` — at every forward pass. The
   actor receives the goal pointer for `target_colour`.

The expected pattern from the prior work: MLP steering changes behaviour in
~1% of attempts; the sequence swap changes it reliably. Together this says
the goal is encoded redundantly across the actor stack but the actionable
copy lives in the planner output, not in the hidden activations.
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
from sklearn.linear_model import LogisticRegression
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


def make_one_step_env(colour: str, propositions, seed: int):
    """Single-goal task: F colour."""
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
    return env


def detect_first_zone_visited(env, propositions):
    unwrapped = get_unwrapped(env)
    agent_pos = unwrapped.agent_pos[:2]
    for name, pos in unwrapped.zone_positions.items():
        if np.linalg.norm(np.array(pos[:2]) - agent_pos) < 0.4:
            for c in COLOURS:
                if c in name:
                    return c
    return None


# ---------------------------------------------------------------------------
# Step 1: collect probe data and train goal-colour classifier on H3
# ---------------------------------------------------------------------------


def collect_probe_data(model, propositions, n_episodes_per_colour: int = 20):
    """For each colour, run F colour episodes and collect H3 activations + label."""
    rows = defaultdict(list)
    for cid, colour in enumerate(COLOURS):
        for ep in tqdm(range(n_episodes_per_colour), desc=f"collect {colour}"):
            env = make_one_step_env(colour, propositions, seed=100 + cid * 1000 + ep)
            out = env.reset(seed=100 + cid * 1000 + ep)
            obs = out[0] if isinstance(out, tuple) else out
            for _ in range(200):
                preprocessed = preprocessing.preprocess_obss([obs], set(propositions))
                emb = model.compute_embedding(preprocessed)
                h3 = model.actor.enc[:6](emb)
                rows["h3"].append(h3.detach().numpy().flatten())
                rows["colour_id"].append(cid)
                dist = model.actor(emb)
                dist.set_epsilon_mask(preprocessed.epsilon_mask)
                action = dist.mode.numpy().flatten()
                obs, _r, done, _info = env.step(action)
                if done:
                    break
            env.close()
    return {k: np.array(v) for k, v in rows.items()}


def train_goal_classifier(data: dict):
    X = data["h3"]
    y = data["colour_id"]
    clf = LogisticRegression(max_iter=2000, multi_class="multinomial").fit(X, y)
    acc = clf.score(X, y)
    return clf, acc


def steering_vector(clf: LogisticRegression, source_id: int, target_id: int) -> np.ndarray:
    """Return w_target − w_source from the classifier's coefficient matrix."""
    w = clf.coef_  # shape (n_classes, n_features)
    return w[target_id] - w[source_id]


# ---------------------------------------------------------------------------
# Step 2: MLP steering rollouts
# ---------------------------------------------------------------------------


def run_mlp_steering(model, propositions, clf, n_trials: int, alpha: float):
    """For each trial: pick distinct source/target colours, run F source with steering toward target."""
    rng = np.random.RandomState(0)
    results = []
    pairs = [(s, t) for s in range(len(COLOURS)) for t in range(len(COLOURS)) if s != t]

    orig_compute = model.compute_embedding

    for trial in tqdm(range(n_trials), desc="MLP steering"):
        s_id, t_id = pairs[trial % len(pairs)]
        source = COLOURS[s_id]
        target = COLOURS[t_id]
        w = steering_vector(clf, s_id, t_id).astype(np.float32)
        w_t = torch.from_numpy(w)

        def patched(obs, _w=w_t, _orig=orig_compute):
            emb = _orig(obs)
            h3 = model.actor.enc[:6](emb)
            h3_steered = h3 + alpha * _w
            # We can't change the actor.enc output directly without recomputing.
            # Instead, fold the perturbation into emb by approximate inverse:
            # not feasible; the cleanest path is to monkey-patch the actor.
            return emb  # passthrough; real perturbation injected via hook below

        # Forward hook on actor.enc to add α·w to its output
        steered_alpha = [alpha]
        steering_w = [w_t]

        def hook(_mod, _inp, out, _w=w_t, _a=alpha):
            return out + _a * _w

        hook_handle = model.actor.enc.register_forward_hook(hook)
        try:
            seed = 5000 + trial
            env = make_one_step_env(source, propositions, seed=seed)
            out = env.reset(seed=seed)
            obs = out[0] if isinstance(out, tuple) else out
            visited = None
            for _ in range(300):
                preprocessed = preprocessing.preprocess_obss([obs], set(propositions))
                dist, _v = model(preprocessed)
                dist.set_epsilon_mask(preprocessed.epsilon_mask)
                action = dist.mode.numpy().flatten()
                obs, _r, done, _info = env.step(action)
                visited = detect_first_zone_visited(env, propositions)
                if visited is not None:
                    break
                if done:
                    break
            env.close()
        finally:
            hook_handle.remove()

        results.append({
            "trial": trial,
            "source": source,
            "target": target,
            "visited": visited,
            "matched_target": visited == target,
            "matched_source": visited == source,
        })
    return results


# ---------------------------------------------------------------------------
# Step 3: Sequence-encoder swap
# ---------------------------------------------------------------------------


def run_sequence_swap(model, propositions, n_trials: int):
    """Run F source while replacing ltl_emb with the encoding of F target."""
    results = []
    pairs = [(s, t) for s in range(len(COLOURS)) for t in range(len(COLOURS)) if s != t]

    orig_compute_embedding = model.compute_embedding

    for trial in tqdm(range(n_trials), desc="Seq-encoder swap"):
        s_id, t_id = pairs[trial % len(pairs)]
        source = COLOURS[s_id]
        target = COLOURS[t_id]

        # Build a "fake" environment for target so we can extract its preprocessed seq
        target_env = make_one_step_env(target, propositions, seed=999 + trial)
        out = target_env.reset(seed=999 + trial)
        target_obs = out[0] if isinstance(out, tuple) else out
        target_pre = preprocessing.preprocess_obss([target_obs], set(propositions))
        target_ltl = model.ltl_net(target_pre.seq).detach()
        target_env.close()

        def patched(obs, _orig=orig_compute_embedding, _ltl=target_ltl):
            env_emb = model.compute_env_embedding(obs)
            return torch.cat([env_emb, _ltl], dim=1)

        model.compute_embedding = patched
        try:
            seed = 7000 + trial
            env = make_one_step_env(source, propositions, seed=seed)
            out = env.reset(seed=seed)
            obs = out[0] if isinstance(out, tuple) else out
            visited = None
            for _ in range(300):
                preprocessed = preprocessing.preprocess_obss([obs], set(propositions))
                dist, _v = model(preprocessed)
                dist.set_epsilon_mask(preprocessed.epsilon_mask)
                action = dist.mode.numpy().flatten()
                obs, _r, done, _info = env.step(action)
                visited = detect_first_zone_visited(env, propositions)
                if visited is not None:
                    break
                if done:
                    break
            env.close()
        finally:
            model.compute_embedding = orig_compute_embedding

        results.append({
            "trial": trial,
            "source": source,
            "target": target,
            "visited": visited,
            "matched_target": visited == target,
            "matched_source": visited == source,
        })
    return results


def summarize(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    matched_target = sum(1 for r in results if r["matched_target"])
    matched_source = sum(1 for r in results if r["matched_source"])
    visited_any = sum(1 for r in results if r["visited"] is not None)
    return {
        "n": n,
        "visited_any": visited_any,
        "matched_target_rate": matched_target / n,
        "matched_source_rate": matched_source / n,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="fresh_baseline")
    p.add_argument("--collect_episodes_per_colour", type=int, default=15)
    p.add_argument("--mlp_trials", type=int, default=120)
    p.add_argument("--swap_trials", type=int, default=120)
    p.add_argument("--alpha", type=float, default=20.0)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir or f"interpretability/results/steering/{args.exp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.exp}")
    model, propositions = load_model(args.exp)

    print("\n[1/3] Collecting probe data + training goal-colour classifier...")
    data = collect_probe_data(model, propositions, args.collect_episodes_per_colour)
    clf, acc = train_goal_classifier(data)
    print(f"  H3 goal-colour classifier in-sample accuracy: {acc:.3f}")
    print(f"  States collected: {len(data['h3'])}")

    print(f"\n[2/3] MLP steering at α={args.alpha}, {args.mlp_trials} trials...")
    mlp_results = run_mlp_steering(model, propositions, clf, args.mlp_trials, args.alpha)
    mlp_summary = summarize(mlp_results)
    print(f"  Summary: {mlp_summary}")

    print(f"\n[3/3] Sequence-encoder swap, {args.swap_trials} trials...")
    swap_results = run_sequence_swap(model, propositions, args.swap_trials)
    swap_summary = summarize(swap_results)
    print(f"  Summary: {swap_summary}")

    out = {
        "exp": args.exp,
        "args": vars(args),
        "classifier_accuracy": float(acc),
        "mlp_steering": mlp_summary,
        "sequence_swap": swap_summary,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(out, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    conds = ["MLP steering\n(actor.enc output)", "Sequence-encoder\nswap"]
    rates = [mlp_summary["matched_target_rate"], swap_summary["matched_target_rate"]]
    bars = ax.bar(conds, rates, color=["#e74c3c", "#27ae60"], edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}",
                ha="center", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of trials where the steered\ncolour was visited first")
    ax.set_title("Subgoal steering: planner-output swap works,\n"
                 "MLP-level activation steering does not",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "steering.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
