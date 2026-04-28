#!/usr/bin/env python3
"""Pull causal results from results.json and emit ready-to-paste numbers.

Run this after project_out.py + plot_results.py complete. Prints:
- Markdown table of conditions vs. optimal-rate / success-rate.
- Sentences for the blogpost prose.
- Copies the two figures into the blogpost figures/ directory under the
  numbering used by BLOGPOST.md.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path


def fmt_pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def main():
    results_path = Path("interpretability/results/causal/fresh_baseline/results.json")
    with open(results_path) as f:
        r = json.load(f)
    s = r["summaries"]

    print("Probe R^2:")
    for k, v in r["probes"].items():
        print(f"  {k}:  R^2 = {v['r2']:.3f},  ||w|| = {v['w_norm']:.3f}")
    print()

    cond_order = ["baseline", "ablate_chain", "ablate_agent"]
    print("| Condition | Optimal-choice rate | Task success |")
    print("|---|---|---|")
    label_map = {
        "baseline": "Baseline (no intervention)",
        "ablate_chain": "Ablate chained-distance direction",
        "ablate_agent": "Ablate agent-distance direction *(positive control)*",
    }
    for c in cond_order:
        if c not in s:
            continue
        ss = s[c]
        opt = fmt_pct(ss["optimal_rate"])
        suc = fmt_pct(ss["success_rate_overall"])
        print(f"| {label_map[c]} | {opt} | {suc} |")
    print()

    print("Sufficiency sweep (optimal-choice rate):")
    rows = []
    if "baseline" in s:
        rows.append((0.0, s["baseline"]["optimal_rate"], s["baseline"]["success_rate_overall"]))
    for k, v in s.items():
        if k.startswith("add_chain_a"):
            tag = k.replace("add_chain_a", "")
            try:
                a = float(tag)
            except ValueError:
                continue
            rows.append((a, v["optimal_rate"], v["success_rate_overall"]))
    rows.sort()
    for a, opt, suc in rows:
        print(f"  alpha = {a:+.1f}  optimal = {fmt_pct(opt)}  success = {fmt_pct(suc)}")
    print()

    # Copy figures
    src_dir = results_path.parent
    dst_dir = Path("interpretability/blogpost/figures")
    moves = [
        ("causal_ablation.png", "11_causal_ablation.png"),
        ("causal_sufficiency.png", "12_causal_sufficiency.png"),
    ]
    for src, dst in moves:
        s_path = src_dir / src
        d_path = dst_dir / dst
        if s_path.exists():
            shutil.copyfile(s_path, d_path)
            print(f"Copied {s_path} -> {d_path}")
        else:
            print(f"WARNING: {s_path} not found (run plot_results.py first)")


if __name__ == "__main__":
    main()
