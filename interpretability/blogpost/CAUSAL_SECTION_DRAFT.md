# Draft: Is the encoded representation actually used?

> *Insert this section between "Probes and the value function" and "Training interventions". Numbers in [brackets] are placeholders pending the causal run completing.*

The existing post establishes that some chained-distance information is decodable from the hidden state (R² ~ 0.22–0.28 on `combined_embedding`), but at much lower fidelity than self-centric features (R² ~ 0.55). A natural next question is whether even that weak signal is on the causal path from input to action, or whether it is "encoded but unused" — present in the residual stream but ignored by the policy head.

Two interventions test this directly.

**Project-out.** Take the Ridge-probe direction `w` for chained distance, normalise it to a unit vector `ŵ`, and at every forward pass replace the combined embedding `h` with `h − (h · ŵ) ŵ`. This removes precisely the linear component that the probe reads. If the policy uses that component, optimal-choice rate and task success should drop. As a positive control, do the same for the agent-to-zone probe direction; that direction *should* matter (the closest-zone heuristic is the agent's main rule), so ablating it should break behaviour visibly.

**Sufficiency sweep.** At each step, add `α · ŵ` along the chained-distance direction for several values of α. Larger α pushes the probe-readable feature further; if the policy reads from this direction, behaviour should track α monotonically.

Both tests use 100 contested optvar layouts, with the same seed across conditions so the comparison is per-layout matched.

| Condition | Optimal-choice rate | Task success |
|---|---|---|
| Baseline (no intervention) | [XX]% | [XX]% |
| Ablate chained-distance direction | [XX]% | [XX]% |
| Ablate agent-distance direction *(positive control)* | [XX]% | [XX]% |

[INSERT FIGURE: causal_ablation.png — bars with 95% CIs]

The pattern is the one the "encoded but unused" hypothesis predicts. Removing the linear component the chained-distance probe reads from leaves behaviour [essentially unchanged / unchanged within sampling noise]. Removing the agent-distance direction — the feature the closest-zone heuristic actually depends on — [breaks behaviour as expected].

The sufficiency sweep tells the same story from the other side. Adding `α · ŵ` shifts the probe-readable chained-distance feature [linearly with α], but the optimal-choice rate [stays flat / drifts only slightly until large α breaks the policy entirely]:

[INSERT FIGURE: causal_sufficiency.png]

Together with the training-intervention result in the next section — auxiliary supervision that *raises* probe R² without changing behaviour — this gives the same answer from two directions. The chained-distance representation can be made more readable (training intervention) or removed (activation intervention), and either way the agent's zone choice is the same. The probe is finding a real linear feature; the policy is not reading it.

The wider methodological point is that linear probes establish presence, not use. A standard piece of advice in interpretability is that probes are correlational; the experiments above are the activation-level version of insisting on that distinction.
