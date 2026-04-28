# Interpretability of DeepLTL

A small interpretability study of the DeepLTL agent (Jackermeier & Abate, ICLR 2025), focused on whether the agent does multi-step planning or relies on reactive heuristics.

This folder contains all of the interpretability contributions on top of upstream DeepLTL. The upstream code is unmodified except for a handful of small additions in `src/` to register custom evaluation environments and to support optional auxiliary losses (see *Patches to upstream* below).

## Structure

```
interpretability/
├── behavioural/        # Optimality tests on optvar / opteq / controlled-orientation
├── probing/            # Linear probes for planning-relevant features
├── causal/             # Project-out and sufficiency-sweep experiments
├── blogpost/           # Write-up + figures
└── results/            # Per-script result outputs (mostly gitignored)
```

## Quick start

```bash
# Install upstream DeepLTL dependencies (see top-level README), then:

# Behavioural baselines
python interpretability/behavioural/optvar_test.py --exp fresh_baseline
python interpretability/behavioural/opteq_test.py --exp fresh_baseline
python interpretability/behavioural/controlled_orientation_test.py --model fresh_baseline

# Linear probes
python interpretability/probing/probe_planning.py --exp fresh_baseline

# Causal mediation: does ablating / adding the chained-distance probe
# direction shift the agent's behaviour?
python interpretability/causal/project_out.py --exp fresh_baseline \
    --collect_episodes 80 --behaviour_episodes 100 --alphas=-3,-1,1,3
python interpretability/causal/plot_results.py
```

The trained `fresh_baseline` checkpoint is shipped under `experiments/ppo/PointLtl2-v0/fresh_baseline/`.

## Custom environments

The interpretability scripts use two evaluation-only environments registered in `src/envs/zones/safety-gymnasium/safety_gymnasium/tasks/ltl/`:

- **`PointLtl2-v0.optvar`** — two intermediate zones placed so that the closer one yields a *longer* total path to the goal. Tests whether the agent prefers the optimal or the myopic intermediate.
- **`PointLtl2-v0.opteq`** — same structure, but both intermediates are equidistant from the agent. Removes the proximity cue and isolates whatever heuristic remains.

Both environments accept a config dict with `intermediate_color`, `goal_color`, and `layout_seed`.

## Patches to upstream

The minimal changes to upstream `src/` are:

- `src/envs/zones/safety-gymnasium/.../tasks/ltl/ltl_optimality_varied.py` — `optvar` task.
- `src/envs/zones/safety-gymnasium/.../tasks/ltl/ltl_optimality_equidistant.py` — `opteq` task.
- `src/envs/zones/safety-gymnasium/safety_gymnasium/__init__.py` — env registration.
- Auxiliary loss support in `src/torch_ac/algos/ppo.py` and `src/model/model.py` (used by the `aux_loss_*` and `combined_*` training variants discussed in the blogpost).
- New curriculum variants in `src/sequence/samplers/curriculum.py`.

## Reading the blogpost

Start with `blogpost/BLOGPOST.md`. Figures are reproducible:

```bash
python interpretability/blogpost/scripts/generate_figures.py
```
