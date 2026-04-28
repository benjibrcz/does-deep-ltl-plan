#!/usr/bin/env python3
"""
Causal mediation: does the chained-distance representation actually
drive the agent's zone choice?

Pipeline:
  1. Collect hidden states + planning labels from optvar rollouts.
  2. Train linear probes for two features:
       - chained_distance (d_int_to_goal): the planning-relevant feature.
       - agent_to_zone (d_agent_to_int): the proximity feature, used as
         a positive control. Removing this should break behaviour.
  3. Get the probe direction w for each feature (Ridge weight, unit-normed).
  4. Run optvar behavioural tests under three conditions:
       - baseline  (no intervention)
       - ablate chained-distance direction
       - ablate agent-to-zone direction
     plus a sufficiency sweep that adds alpha * w_chained for various alpha.
  5. Report: optimal-choice rate, task success, mean probe readout under
     each condition.

The intervention is applied at the combined_embedding level (just before
the actor/critic heads), via a monkey-patch of model.compute_embedding.

Output: interpretability/results/causal/<exp>/results.json + figure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, "src")

import preprocessing
from config import model_configs
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
COLORS = ["blue", "green", "yellow", "magenta"]


def get_unwrapped(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def make_optvar_env(int_color: str, goal_color: str, layout_seed: int, propositions):
    config = {
        "agent_name": "Point",
        "intermediate_color": int_color,
        "goal_color": goal_color,
        "layout_seed": layout_seed,
    }
    env = sg_make("PointLtl2-v0.optvar", config=config)
    env = SafetyGymWrapper(env)
    env = FlattenObservation(env)

    int_reach = frozenset([Assignment.single_proposition(int_color, propositions).to_frozen()])
    goal_reach = frozenset([Assignment.single_proposition(goal_color, propositions).to_frozen()])
    task = LDBASequence([(int_reach, frozenset()), (goal_reach, frozenset())])

    def sampler(_props):
        return lambda: task

    env = SequenceWrapper(env, sampler(propositions), step_penalty=0.0)
    env = TimeLimit(env, max_episode_steps=300)
    env = RemoveTruncWrapper(env)
    return env


def load_model(exp: str, env_name: str = "PointLtl2-v0"):
    from envs import make_env

    temp_sampler = CurriculumSampler.partial(curricula[env_name])
    temp_env = make_env(env_name, temp_sampler, sequence=True)
    config = model_configs[env_name]
    store = ModelStore(env_name, exp, seed=0)
    status = store.load_training_status(map_location="cpu")
    store.load_vocab()
    model = build_model(temp_env, status, config)
    model.eval()
    propositions = list(temp_env.get_propositions())
    temp_env.close()
    return model, propositions


# ---------------------------------------------------------------------------
# Step 1: collect probing data
# ---------------------------------------------------------------------------


def collect_probe_data(model, propositions, n_episodes: int, max_steps: int):
    """Roll out optvar episodes and record combined_embedding + labels.

    Also records the episode index for each state, so probes can be evaluated
    with episode-disjoint train/test splits (within-episode leakage would
    inflate R^2 for any feature that is constant across an episode, e.g. the
    fixed zone-to-goal distance).
    """
    rows = defaultdict(list)
    color_pairs = [(a, b) for a in COLORS for b in COLORS if a != b]

    for ep in tqdm(range(n_episodes), desc="collect"):
        int_color, goal_color = color_pairs[ep % len(color_pairs)]
        layout_seed = 1000 + ep * 7
        try:
            env = make_optvar_env(int_color, goal_color, layout_seed, propositions)
        except Exception:
            continue

        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        unwrapped = get_unwrapped(env)
        zone_positions = {k: np.array(v[:2]) for k, v in unwrapped.zone_positions.items()}

        int_zones = [(k, v) for k, v in zone_positions.items() if int_color in k]
        goal_zones = [(k, v) for k, v in zone_positions.items() if goal_color in k]
        if len(int_zones) < 2 or len(goal_zones) < 1:
            env.close()
            continue
        goal_pos = goal_zones[0][1]

        for _step in range(max_steps):
            agent_pos = unwrapped.agent_pos[:2].copy()

            preprocessed = preprocessing.preprocess_obss([obs], set(propositions))
            embedding = model.compute_embedding(preprocessed)  # [1, D]
            value = model.critic(embedding).squeeze(1)
            dist = model.actor(embedding)
            dist.set_epsilon_mask(preprocessed.epsilon_mask)
            action = dist.mode.numpy().flatten()

            d_to_int = [np.linalg.norm(agent_pos - p) for _, p in int_zones]
            d_int_to_goal = [np.linalg.norm(p - goal_pos) for _, p in int_zones]
            min_d_agent = float(min(d_to_int))
            min_d_chain = float(min(d_int_to_goal))

            rows["embedding"].append(embedding.detach().numpy().flatten())
            rows["d_agent"].append(min_d_agent)
            rows["d_chain"].append(min_d_chain)
            rows["value"].append(float(value.detach().numpy().flatten()[0]))
            rows["episode"].append(ep)

            obs, _r, done, _info = env.step(action)
            if done:
                break
        env.close()

    return {k: np.array(v) for k, v in rows.items()}


# ---------------------------------------------------------------------------
# Step 2: train probes and extract directions
# ---------------------------------------------------------------------------


def train_probe(X: np.ndarray, y: np.ndarray, episodes: np.ndarray | None = None):
    """Ridge probe with episode-disjoint train/test split when episodes given.

    Per-episode features (e.g. fixed zone positions) are constant within an
    episode, so a state-level random split lets the probe memorise episode
    identity. Splitting by episode avoids that leak.
    """
    if episodes is not None:
        unique_eps = np.unique(episodes)
        rng = np.random.RandomState(42)
        rng.shuffle(unique_eps)
        n_test = max(1, int(0.2 * len(unique_eps)))
        test_eps = set(unique_eps[:n_test].tolist())
        test_mask = np.array([e in test_eps for e in episodes])
        X_tr, X_te = X[~test_mask], X[test_mask]
        y_tr, y_te = y[~test_mask], y[test_mask]
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    r2 = r2_score(y_te, pred)
    w = model.coef_.astype(np.float32)
    w_norm = float(np.linalg.norm(w))
    return {
        "r2": float(r2),
        "w": w,
        "w_unit": w / (w_norm + 1e-12),
        "w_norm": w_norm,
        "intercept": float(model.intercept_),
    }


# ---------------------------------------------------------------------------
# Step 3: behavioural test under intervention
# ---------------------------------------------------------------------------


def run_behavioural(
    model,
    propositions,
    n_episodes: int,
    seed: int,
    intervention=None,  # callable: embedding -> embedding (or None for baseline)
    max_steps: int = 500,
):
    """Run optvar trials, return per-episode dicts with optimality/success."""
    import random as _random

    _random.seed(seed)
    rng = np.random.RandomState(seed)
    color_pairs = [(a, b) for a in COLORS for b in COLORS if a != b]

    # Patch compute_embedding to apply intervention
    orig_compute = model.compute_embedding
    if intervention is not None:
        def patched(obs):
            h = orig_compute(obs)
            return intervention(h)
        model.compute_embedding = patched

    results = []
    try:
        for ep in tqdm(range(n_episodes), desc=f"behaviour seed={seed}"):
            int_color, goal_color = _random.choice(color_pairs)
            layout_seed = seed + ep * 7
            try:
                env = make_optvar_env(int_color, goal_color, layout_seed, propositions)
            except Exception:
                continue
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            unwrapped = get_unwrapped(env)
            start_pos = unwrapped.agent_pos[:2].copy()
            zone_positions = {k: np.array(v[:2]) for k, v in unwrapped.zone_positions.items()}
            int_zones = [(k, v) for k, v in zone_positions.items() if int_color in k]
            goal_zones = [(k, v) for k, v in zone_positions.items() if goal_color in k]
            if len(int_zones) < 2 or len(goal_zones) < 1:
                env.close()
                continue
            goal_pos = goal_zones[0][1]

            int_analysis = []
            for name, pos in int_zones:
                d_a = np.linalg.norm(pos - start_pos)
                d_g = np.linalg.norm(pos - goal_pos)
                int_analysis.append({"name": name, "pos": pos, "d_a": d_a, "total": d_a + d_g})
            myopic = min(int_analysis, key=lambda x: x["d_a"])["name"]
            optimal = min(int_analysis, key=lambda x: x["total"])["name"]
            contested = myopic != optimal

            choice = None
            success = False
            for _step in range(max_steps):
                preprocessed = preprocessing.preprocess_obss([obs], set(propositions))
                embedding = model.compute_embedding(preprocessed)
                dist = model.actor(embedding)
                dist.set_epsilon_mask(preprocessed.epsilon_mask)
                action = dist.mode.numpy().flatten()
                obs, _r, done, info = env.step(action)
                pos = unwrapped.agent_pos[:2].copy()
                if choice is None:
                    for name, zpos in int_zones:
                        if np.linalg.norm(pos - zpos) < 0.4:
                            choice = name
                            break
                if done:
                    success = bool(info.get("success", False))
                    break
            env.close()
            results.append({
                "ep": ep,
                "contested": contested,
                "choice": choice,
                "is_optimal": choice == optimal,
                "is_myopic": choice == myopic,
                "success": success,
                "int_color": int_color,
                "goal_color": goal_color,
            })
    finally:
        # Restore the original compute_embedding (avoid leaking patches between
        # interventions when this function is called repeatedly on the same model)
        model.compute_embedding = orig_compute

    return results


# ---------------------------------------------------------------------------
# Step 4: orchestration
# ---------------------------------------------------------------------------


def make_project_out(w_unit: np.ndarray):
    w_t = torch.from_numpy(w_unit).float()
    def fn(h: torch.Tensor) -> torch.Tensor:
        # h: [B, D]; remove the component along w_unit
        proj = (h @ w_t).unsqueeze(-1) * w_t
        return h - proj
    return fn


def make_add_direction(w_unit: np.ndarray, alpha: float):
    w_t = torch.from_numpy(w_unit).float()
    def fn(h: torch.Tensor) -> torch.Tensor:
        return h + alpha * w_t
    return fn


def summarize(results: list[dict]) -> dict:
    contested = [r for r in results if r["contested"] and r["choice"] is not None]
    n = len(contested)
    if n == 0:
        return {"n_contested": 0}
    optimal = sum(1 for r in contested if r["is_optimal"])
    success_total = sum(1 for r in results if r["success"]) / max(len(results), 1)
    return {
        "n_total": len(results),
        "n_contested": n,
        "optimal_rate": optimal / n,
        "myopic_rate": sum(1 for r in contested if r["is_myopic"]) / n,
        "success_rate_overall": success_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="fresh_baseline")
    parser.add_argument("--collect_episodes", type=int, default=80)
    parser.add_argument("--collect_max_steps", type=int, default=120)
    parser.add_argument("--behaviour_episodes", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alphas", default="-3,-1,0,1,3")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir or f"interpretability/results/causal/{args.exp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.exp}")
    model, propositions = load_model(args.exp)

    # ----- collect probe data -----
    print("\n[1/4] Collecting probe data...")
    data = collect_probe_data(
        model, propositions,
        n_episodes=args.collect_episodes,
        max_steps=args.collect_max_steps,
    )
    print(f"  Collected {len(data['embedding'])} states, embedding dim={data['embedding'].shape[1]}")
    np.savez(out_dir / "probe_data.npz", **data)

    # ----- train probes -----
    print("\n[2/4] Training probes...")
    X = data["embedding"]
    eps = data.get("episode", None)
    probe_chain = train_probe(X, data["d_chain"], episodes=eps)
    probe_agent = train_probe(X, data["d_agent"], episodes=eps)
    print(f"  d_chain   R^2 = {probe_chain['r2']:.3f}  ||w|| = {probe_chain['w_norm']:.3f}")
    print(f"  d_agent   R^2 = {probe_agent['r2']:.3f}  ||w|| = {probe_agent['w_norm']:.3f}")

    # ----- behavioural sweep -----
    print("\n[3/4] Behaviour under intervention...")
    conditions = {
        "baseline": None,
        "ablate_chain": make_project_out(probe_chain["w_unit"]),
        "ablate_agent": make_project_out(probe_agent["w_unit"]),  # positive control
    }
    alphas = [float(a) for a in args.alphas.split(",")]
    for a in alphas:
        if a == 0.0:
            continue  # baseline already covers it
        conditions[f"add_chain_a{a:+g}"] = make_add_direction(probe_chain["w_unit"], a)

    summaries = {}
    raw = {}
    for name, fn in conditions.items():
        print(f"\n  -- {name}")
        res = run_behavioural(
            model, propositions,
            n_episodes=args.behaviour_episodes,
            seed=args.seed,
            intervention=fn,
        )
        summaries[name] = summarize(res)
        raw[name] = res
        print(f"     summary: {summaries[name]}")

    # ----- save -----
    print("\n[4/4] Saving results...")
    out = {
        "exp": args.exp,
        "args": vars(args),
        "probes": {
            "d_chain": {"r2": probe_chain["r2"], "w_norm": probe_chain["w_norm"]},
            "d_agent": {"r2": probe_agent["r2"], "w_norm": probe_agent["w_norm"]},
        },
        "summaries": summaries,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    np.savez(
        out_dir / "raw_results.npz",
        **{k: np.array(v, dtype=object) for k, v in raw.items()},
    )
    print(f"\nWrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
