#!/usr/bin/env python3
"""Plot causal-mediation results from project_out.py output.

Produces two figures:
  causal_ablation.png    — bar chart of optimal-choice rate and success rate
                           under baseline / ablate-chain / ablate-agent.
  causal_sufficiency.png — sufficiency sweep: optimal rate vs. alpha for the
                           chained-distance direction.
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return center - margin, center + margin


def plot_ablation(results: dict, out_path: Path):
    """Two-panel: optimal-choice rate and overall success, under each condition."""
    summaries = results["summaries"]
    conds = ["baseline", "ablate_chain", "ablate_agent"]
    labels = ["Baseline", "Ablate\nchained-distance\ndirection", "Ablate\nagent-distance\ndirection\n(positive control)"]

    optimal_rates = [summaries[c]["optimal_rate"] for c in conds]
    success_rates = [summaries[c]["success_rate_overall"] for c in conds]
    n_contested = [summaries[c]["n_contested"] for c in conds]
    n_total = [summaries[c]["n_total"] for c in conds]

    opt_cis = [wilson_ci(int(round(r * n)), n) for r, n in zip(optimal_rates, n_contested)]
    suc_cis = [wilson_ci(int(round(r * n)), n) for r, n in zip(success_rates, n_total)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    bars = ax.bar(labels, optimal_rates, color=[GREY, BLUE, RED],
                  edgecolor="black", linewidth=0.6, width=0.55)
    for b, r, (lo, hi) in zip(bars, optimal_rates, opt_cis):
        ax.errorbar(b.get_x() + b.get_width() / 2, r,
                    yerr=[[r - lo], [hi - r]], color="black", capsize=4, linewidth=1)
        ax.text(b.get_x() + b.get_width() / 2, r + 0.025,
                f"{r:.2f}", ha="center", fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Optimal-choice rate (contested cases)")
    ax.set_title("Behaviour")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1]
    bars = ax.bar(labels, success_rates, color=[GREY, BLUE, RED],
                  edgecolor="black", linewidth=0.6, width=0.55)
    for b, r, (lo, hi) in zip(bars, success_rates, suc_cis):
        ax.errorbar(b.get_x() + b.get_width() / 2, r,
                    yerr=[[r - lo], [hi - r]], color="black", capsize=4, linewidth=1)
        ax.text(b.get_x() + b.get_width() / 2, r + 0.025,
                f"{r:.2f}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Task success rate")
    ax.set_title("Task completion")

    fig.suptitle("Project-out interventions: 1D ablations of either probe direction\n"
                 "produce similar mild shifts; task success is preserved",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_sufficiency(results: dict, out_path: Path):
    """Sufficiency sweep: optimal rate vs. alpha when adding alpha * w_chain."""
    summaries = results["summaries"]
    rows = []
    for name, s in summaries.items():
        if name.startswith("add_chain_a"):
            tag = name.replace("add_chain_a", "")
            try:
                alpha = float(tag)
            except ValueError:
                continue
            rows.append((alpha, s["optimal_rate"], s["success_rate_overall"],
                         s["n_contested"], s["n_total"]))
    if "baseline" in summaries:
        s = summaries["baseline"]
        rows.append((0.0, s["optimal_rate"], s["success_rate_overall"],
                     s["n_contested"], s["n_total"]))

    rows.sort()
    alphas = [r[0] for r in rows]
    opt = [r[1] for r in rows]
    suc = [r[2] for r in rows]
    n_c = [r[3] for r in rows]
    n_t = [r[4] for r in rows]

    opt_cis = [wilson_ci(int(round(r * n)), n) for r, n in zip(opt, n_c)]
    suc_cis = [wilson_ci(int(round(r * n)), n) for r, n in zip(suc, n_t)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(alphas, opt,
                yerr=[[o - lo for o, (lo, hi) in zip(opt, opt_cis)],
                      [hi - o for o, (lo, hi) in zip(opt, opt_cis)]],
                marker="o", color=BLUE, linewidth=2, capsize=4,
                label="Optimal-choice rate")
    ax.errorbar(alphas, suc,
                yerr=[[s - lo for s, (lo, hi) in zip(suc, suc_cis)],
                      [hi - s for s, (lo, hi) in zip(suc, suc_cis)]],
                marker="s", color=GREEN, linewidth=2, capsize=4, linestyle="--",
                label="Task success rate")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$\alpha$ (added along chained-distance direction $\hat w$)")
    ax.set_ylabel("Rate")
    ax.set_title("Sufficiency sweep along the chained-distance direction")
    ax.legend(frameon=False, loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="interpretability/results/causal/fresh_baseline/results.json")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    results_path = Path(args.results)
    out_dir = Path(args.out_dir) if args.out_dir else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        results = json.load(f)

    plot_ablation(results, out_dir / "causal_ablation.png")
    plot_sufficiency(results, out_dir / "causal_sufficiency.png")


if __name__ == "__main__":
    main()
