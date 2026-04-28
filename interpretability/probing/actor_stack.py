#!/usr/bin/env python3
"""Linear probes through the actor stack.

For each rollout step, we capture activations at five depths:

    IN  = combined_embedding         (96-d, fusion of env + LTL)
    H1  = actor.enc[:2](IN)          (64-d, after first Linear+ReLU)
    H2  = actor.enc[:4](IN)          (64-d, after second)
    H3  = actor.enc[:6](IN)          (64-d, full enc output)
    OUT = actor.mu(H3)               ( 2-d, policy mean)

We train Ridge / LogisticRegression probes on each layer for ~10 targets that
cover the gradient from raw sensors → affordances → task signals →
policy-aligned features. The resulting heatmap shows how representations
shift from environment-relevant to action-relevant down the stack.

Splits: episode-disjoint (16 of 80 layouts held out).
Output: results/actor_stack/<exp>/results.json + heatmap.png
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from tqdm import tqdm

sys.path.insert(0, "src")

import preprocessing
from config import model_configs
from envs import make_env
from model.model import build_model
from sequence.samplers import CurriculumSampler, curricula
from utils.model_store import ModelStore

torch.set_grad_enabled(False)


# ---------------------------------------------------------------------------
# Model loading + activation capture
# ---------------------------------------------------------------------------

LAYERS = ["IN", "H1", "H2", "H3", "OUT"]


def load_model(exp: str, env_name: str = "PointLtl2-v0"):
    sampler = CurriculumSampler.partial(curricula[env_name])
    temp_env = make_env(env_name, sampler, sequence=True)
    cfg = model_configs[env_name]
    store = ModelStore(env_name, exp, seed=0)
    status = store.load_training_status(map_location="cpu")
    store.load_vocab()
    model = build_model(temp_env, status, cfg)
    model.eval()
    propositions = list(temp_env.get_propositions())
    return model, propositions, temp_env


def capture_activations(model, preprocessed):
    """Returns dict of layer_name -> 1-d numpy array."""
    emb = model.compute_embedding(preprocessed)
    enc = model.actor.enc
    # Sequential indexing slices the underlying ModuleList
    h1 = enc[:2](emb)
    h2 = enc[:4](emb)
    h3 = enc[:6](emb)
    mu = model.actor.mu(h3)
    return {
        "IN": emb.detach().numpy().flatten(),
        "H1": h1.detach().numpy().flatten(),
        "H2": h2.detach().numpy().flatten(),
        "H3": h3.detach().numpy().flatten(),
        "OUT": mu.detach().numpy().flatten(),
    }, mu


# ---------------------------------------------------------------------------
# Targets — 10 that span the sensor → action gradient
# ---------------------------------------------------------------------------

COLOURS = ["blue", "green", "yellow", "magenta"]


def get_unwrapped(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def angle_class_8(angle_rad: float) -> int:
    """Bin a signed angle in [-pi, pi] into 8 classes (0..7)."""
    a = (angle_rad + np.pi) % (2 * np.pi)
    return int((a / (2 * np.pi)) * 8) % 8


def compute_targets(env, obs, mu_np: np.ndarray, current_goal_colour: str | None,
                    last_pos: np.ndarray | None) -> dict:
    """Compute target dict for a single step.

    Targets cover:
      sensors: agent_speed (raw)
      affordance: free_space_ahead (proxy: min wall lidar in forward sector)
      task: goal_colour_id, bearing_to_goal_cls8, dist_to_nearest_goal_zone
      policy: policy_turn_sign, policy_angle_cls8
      geometry (negative controls): agent_x, delta_x
    """
    unwrapped = get_unwrapped(env)
    agent_pos = unwrapped.agent_pos[:2].astype(np.float64)
    zone_positions = {k: np.array(v[:2], dtype=np.float64)
                      for k, v in unwrapped.zone_positions.items()}

    # Agent velocity (linear, x/y in body frame would need transform; use mag)
    qvel = unwrapped.task.data.qvel[:2]
    agent_speed = float(np.linalg.norm(qvel))

    # Free space ahead: min of wall lidar bins in forward sector
    feats = obs["features"]
    # Last 4 dims = wall_sensor (added by SafetyGymWrapper). Lidar dims are 16
    # bins per colour × 4 colours = 64, then 12 IMU dims, then 4 wall = 80.
    # Conservative proxy: smallest of all lidar values across colours in front
    # sector — but the IMU/lidar layout depends on dict ordering. Use overall
    # nearest-anything as a proxy.
    # Take the largest of the colour lidar values as a "near zone signal" - free space is its complement
    # We instead use: mean of last 4 (wall_sensor 4-bin: front, back, left, right)
    wall_sensor = feats[-4:]
    free_space_ahead = float(1.0 - wall_sensor[0])  # 1 - front-wall proximity

    # Goal colour id (current LTL subgoal colour). NaN encodes "unknown".
    if current_goal_colour is None:
        goal_id = np.nan
    else:
        goal_id = float(COLOURS.index(current_goal_colour))

    # Bearing + distance to nearest current-goal zone
    if current_goal_colour is not None:
        gz = [p for k, p in zone_positions.items() if current_goal_colour in k]
        if gz:
            d_vec = np.array([np.linalg.norm(p - agent_pos) for p in gz])
            i = int(np.argmin(d_vec))
            target_pos = gz[i]
            d_goal = float(d_vec[i])
            heading = float(unwrapped.task.data.qpos[2])  # qpos[2] = yaw for Point
            delta = target_pos - agent_pos
            bearing_world = float(np.arctan2(delta[1], delta[0]))
            bearing_local = (bearing_world - heading + np.pi) % (2 * np.pi) - np.pi
            bearing_cls = float(angle_class_8(bearing_local))
        else:
            d_goal = np.nan
            bearing_cls = np.nan
    else:
        d_goal = np.nan
        bearing_cls = np.nan

    # Policy outputs. Binary turn sign: 0 = left or zero, 1 = right.
    policy_turn_sign = float(1 if mu_np[1] > 0 else 0)
    policy_angle = float(np.arctan2(mu_np[1], mu_np[0]))
    policy_angle_cls = float(angle_class_8(policy_angle))

    # Negative controls
    agent_x = float(agent_pos[0])
    if last_pos is None:
        delta_x = 0.0
    else:
        delta_x = float(agent_pos[0] - last_pos[0])

    return {
        "agent_speed":          ("reg", agent_speed),
        "goal_colour_id":       ("cls", goal_id),
        "bearing_to_goal_cls8": ("cls", bearing_cls),
        "dist_to_goal":         ("reg", d_goal),
        "policy_turn_sign":     ("cls", policy_turn_sign),
        "policy_angle_cls8":    ("cls", policy_angle_cls),
        "policy_mu_norm":       ("reg", float(np.linalg.norm(mu_np))),
        "agent_x":              ("reg", agent_x),
        "delta_x":              ("reg", delta_x),
    }


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------


def get_current_goal_colour(propositions, obs):
    """Read the LTL planner's next-goal colour from the obs structure."""
    # SequenceWrapper stores the candidate plan in obs['goal'] as a sequence
    # of proposition sets. The first set is the current sub-goal. We approximate
    # by picking the first colour mentioned.
    seq = obs.get("goal", None)
    if seq is None:
        return None
    # seq is a list of (reach_set, avoid_set) pairs; reach_set is a frozenset of
    # frozenset (assignment) - too detailed. Read out from obs['propositions']
    # is unreliable. Easier: introspect the planner via `obs['initial_goal']`.
    # Actually, the wrapper exposes the chosen plan; pull from there.
    try:
        first_pair = seq[0]
        reach = first_pair[0]
        # reach is a frozenset of frozenset. Each inner is an Assignment frozen.
        # Pick the first one's positive proposition.
        first = next(iter(reach))
        # first is a frozenset of (prop, sign) pairs. Get the positive prop.
        for prop, sign in first:
            if sign and prop in COLOURS:
                return prop
    except (StopIteration, AttributeError, KeyError, IndexError, TypeError):
        return None
    return None


def collect(model, propositions, env, n_episodes: int, max_steps: int):
    rows = defaultdict(list)
    for ep in tqdm(range(n_episodes), desc="rollouts"):
        out = env.reset(seed=42 + ep)
        obs = out[0] if isinstance(out, tuple) else out
        last_pos = None
        for step in range(max_steps):
            preprocessed = preprocessing.preprocess_obss([obs], set(propositions))
            acts, mu = capture_activations(model, preprocessed)
            mu_np = mu.detach().numpy().flatten()

            current_goal = get_current_goal_colour(propositions, obs)
            tgt = compute_targets(env, obs, mu_np, current_goal, last_pos)

            for layer in LAYERS:
                rows[f"act_{layer}"].append(acts[layer])
            for name, (kind, val) in tgt.items():
                rows[f"tgt_{name}"].append(val)
                rows[f"kind_{name}"] = kind
            rows["episode"].append(ep)
            rows["step"].append(step)

            unwrapped = get_unwrapped(env)
            last_pos = unwrapped.agent_pos[:2].copy()

            dist = model.actor(model.compute_embedding(preprocessed))
            dist.set_epsilon_mask(preprocessed.epsilon_mask)
            action = dist.mode.numpy().flatten()
            obs, _r, done, _info = env.step(action)
            if done:
                break

    # convert
    out = {}
    for k, v in rows.items():
        if k.startswith("kind_"):
            out[k] = v
        else:
            out[k] = np.array(v)
    return out


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------


def episode_disjoint_mask(episodes: np.ndarray, frac: float = 0.2, seed: int = 42):
    rng = np.random.RandomState(seed)
    unique = np.unique(episodes)
    rng.shuffle(unique)
    n_test = max(1, int(frac * len(unique)))
    test = set(unique[:n_test].tolist())
    return np.array([e in test for e in episodes])


def train_probe(X: np.ndarray, y: np.ndarray, kind: str, episodes: np.ndarray):
    test_mask = episode_disjoint_mask(episodes)
    Xtr, Xte = X[~test_mask], X[test_mask]
    ytr, yte = y[~test_mask], y[test_mask]
    if kind == "reg":
        good = np.isfinite(ytr)
        good_te = np.isfinite(yte)
        if good.sum() < 10 or good_te.sum() < 5:
            return float("nan"), "R^2"
        m = Ridge(alpha=1.0).fit(Xtr[good], ytr[good])
        return float(r2_score(yte[good_te], m.predict(Xte[good_te]))), "R^2"
    else:
        good = np.isfinite(ytr)
        good_te = np.isfinite(yte)
        if good.sum() < 10 or good_te.sum() < 5 or len(np.unique(ytr[good])) < 2:
            return float("nan"), "Acc"
        m = LogisticRegression(max_iter=1000).fit(Xtr[good], ytr[good].astype(int))
        return float(accuracy_score(yte[good_te].astype(int), m.predict(Xte[good_te]))), "Acc"


def run_probes(data: dict, target_order: list[str]):
    episodes = data["episode"]
    results = {}
    for layer in LAYERS:
        X = data[f"act_{layer}"]
        results[layer] = {}
        for tgt in target_order:
            y = data[f"tgt_{tgt}"]
            kind = data[f"kind_{tgt}"]
            score, metric = train_probe(X, y, kind, episodes)
            results[layer][tgt] = {"score": score, "metric": metric}
            print(f"  {layer:>3}  {tgt:<22}  {metric}={score:.3f}")
        print()
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_heatmap(results: dict, target_order: list[str], target_groups: dict, out_path: Path):
    n_layers = len(LAYERS)
    n_targets = len(target_order)
    matrix = np.full((n_targets, n_layers), np.nan)
    for j, layer in enumerate(LAYERS):
        for i, tgt in enumerate(target_order):
            matrix[i, j] = results[layer][tgt]["score"]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * n_targets)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.2, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(LAYERS)
    ax.set_yticks(range(n_targets))
    ax.set_yticklabels([target_groups.get(t, t) for t in target_order])

    for i in range(n_targets):
        for j in range(n_layers):
            v = matrix[i, j]
            if np.isnan(v):
                txt = "—"
            else:
                txt = f"{v:.2f}"
            color = "white" if (not np.isnan(v) and (v < 0.3 or v > 0.85)) else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label="Probe score (R² or accuracy)")
    ax.set_title("Probing the actor stack: representations shift from\n"
                 "environment-relevant (top) to action-relevant (bottom)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Order targets by where we expect them to peak: sensors at IN, affordances IN/H1,
# task signals through middle, policy-aligned at the end. Negative controls last.
TARGET_ORDER = [
    "agent_speed",
    "goal_colour_id",
    "bearing_to_goal_cls8",
    "policy_turn_sign",
    "policy_angle_cls8",
    "agent_x",
    "delta_x",
]

PRETTY = {
    "agent_speed":          "agent speed (sensor)",
    "goal_colour_id":       "goal colour id (task)",
    "bearing_to_goal_cls8": "bearing to goal cls8 (task)",
    "policy_turn_sign":     "policy turn sign (action)",
    "policy_angle_cls8":    "policy angle cls8 (action)",
    "agent_x":              "agent x position (geometry, neg. control)",
    "delta_x":              "Δx step (geometry, neg. control)",
}


def make_rollout_env(propositions, env_name="PointLtl2-v0"):
    """Build a PointLtl2-v0 environment using the standard make_env path."""
    sampler = CurriculumSampler.partial(curricula[env_name])
    return make_env(env_name, sampler, sequence=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="fresh_baseline")
    p.add_argument("--n_episodes", type=int, default=80)
    p.add_argument("--max_steps", type=int, default=120)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir or f"interpretability/results/actor_stack/{args.exp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.exp}")
    model, propositions, _model_env = load_model(args.exp)

    print("\n[1/3] Rolling out episodes...")
    env = make_rollout_env(propositions)
    data = collect(model, propositions, env,
                   n_episodes=args.n_episodes, max_steps=args.max_steps)
    n_states = len(data["episode"])
    print(f"  Collected {n_states} states across {len(np.unique(data['episode']))} episodes")

    print("\n[2/3] Training probes (episode-disjoint splits)...")
    results = run_probes(data, TARGET_ORDER)

    out = {
        "exp": args.exp,
        "args": vars(args),
        "n_states": int(n_states),
        "n_episodes": int(len(np.unique(data["episode"]))),
        "results": results,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n[3/3] Plotting...")
    plot_heatmap(results, TARGET_ORDER, PRETTY, out_dir / "actor_stack_heatmap.png")
    print(f"\nWrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
