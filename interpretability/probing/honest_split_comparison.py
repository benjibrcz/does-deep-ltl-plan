#!/usr/bin/env python3
"""Compare random-split vs episode-disjoint probe R^2 for planning features.

The original probe analysis used random train/test splits across states.
Because zone positions are constant within an optvar episode, this lets
the probe memorise "which layout is this" from features that correlate
with the episode index. Splitting by episode removes that leak.

Output: figure showing the gap between the two evaluation protocols.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


GREEN = "#27ae60"
RED = "#e74c3c"
BLUE = "#2980b9"
GREY = "#95a5a6"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})


# Pretty names for the figure
PRETTY = {
    "d_agent_to_int0": "agent → int₀",
    "d_agent_to_int1": "agent → int₁",
    "d_agent_to_goal": "agent → goal",
    "d_int0_to_goal": "int₀ → goal (chained)",
    "d_int1_to_goal": "int₁ → goal (chained)",
    "total_via_int0": "total via int₀",
    "total_via_int1": "total via int₁",
    "optimality_gap": "optimality gap",
}

OBSERVABLE = ["d_agent_to_int0", "d_agent_to_int1", "d_agent_to_goal"]
PLANNING = ["d_int0_to_goal", "d_int1_to_goal", "optimality_gap"]


def episode_disjoint_mask(episodes: np.ndarray, frac: float, seed: int = 42):
    rng = np.random.RandomState(seed)
    unique = np.unique(episodes)
    rng.shuffle(unique)
    n_test = max(1, int(frac * len(unique)))
    test = set(unique[:n_test].tolist())
    return np.array([e in test for e in episodes])


def evaluate(X: np.ndarray, y: np.ndarray, episodes: np.ndarray, alpha: float = 1.0):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rmodel = Ridge(alpha=alpha).fit(Xtr, ytr)
    r2_random = r2_score(yte, rmodel.predict(Xte))

    mask = episode_disjoint_mask(episodes, 0.2)
    emodel = Ridge(alpha=alpha).fit(X[~mask], y[~mask])
    r2_ep = r2_score(y[mask], emodel.predict(X[mask]))
    return r2_random, r2_ep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="interpretability/results/optvar_probing/probe_data_fresh_baseline.npz")
    parser.add_argument("--layer", default="combined_embedding")
    parser.add_argument("--out", default="interpretability/results/optvar_probing/honest_split_comparison.png")
    args = parser.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X = d[f"act_{args.layer}"]
    episodes = d["episode"]

    labels = OBSERVABLE + PLANNING
    rows = []
    for label in labels:
        y = d[f"label_{label}"]
        if y.ndim == 0 or y.std() < 1e-6:
            continue
        r2_random, r2_ep = evaluate(X, y, episodes)
        rows.append((label, r2_random, r2_ep))
        print(f"{label:>22}  random={r2_random:.3f}  episode-disjoint={r2_ep:.3f}")

    # Figure
    fig, ax = plt.subplots(figsize=(10, 5))
    n = len(rows)
    x = np.arange(n)
    width = 0.4

    pretty_labels = [PRETTY.get(r[0], r[0]) for r in rows]
    rand_vals = [r[1] for r in rows]
    ep_vals = [r[2] for r in rows]

    bars1 = ax.bar(x - width / 2, rand_vals, width,
                   color=GREY, edgecolor="black", linewidth=0.5,
                   label="Random split (within-episode leak)")
    bars2 = ax.bar(x + width / 2, ep_vals, width,
                   color=BLUE, edgecolor="black", linewidth=0.5,
                   label="Episode-disjoint split (honest)")

    for b, v in zip(bars1, rand_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015 if v >= 0 else v - 0.04,
                f"{v:.2f}", ha="center", fontsize=9, color="#444")
    for b, v in zip(bars2, ep_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015 if v >= 0 else v - 0.04,
                f"{v:.2f}", ha="center", fontsize=9, color="black", fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvspan(-0.5, len(OBSERVABLE) - 0.5, color=GREEN, alpha=0.05)
    ax.axvspan(len(OBSERVABLE) - 0.5, n - 0.5, color=RED, alpha=0.05)

    ax.text(len(OBSERVABLE) / 2 - 0.5, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.5,
            "Self-centric (observable)", ha="center", fontsize=10, color="#2e7d32",
            fontweight="bold")
    ax.text((len(OBSERVABLE) + n) / 2 - 0.5, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.5,
            "Planning-relevant", ha="center", fontsize=10, color="#c62828",
            fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(pretty_labels, rotation=20, ha="right")
    ax.set_ylabel(r"Probe R$^2$ on `combined_embedding`")
    ax.set_title("Random splits inflate probe R² for chained distances\n"
                 "Under episode-disjoint evaluation, chained features are below the mean-only baseline")
    ax.legend(frameon=False, loc="lower left")

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
