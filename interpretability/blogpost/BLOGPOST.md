# Does DeepLTL Plan?

[DeepLTL](https://openreview.net/forum?id=9pW2J49flQ) (Jackermeier & Abate, ICLR 2025) is a goal-conditioned reinforcement-learning agent that translates Linear Temporal Logic specifications into sub-goals and satisfies them at ~95% success on the paper's benchmark. Its Figure 1 makes a stronger claim than success: faced with `F blue THEN F green` and two candidate blue zones, the agent picks the *farther* blue zone — because doing so leaves a shorter onward path to green. If the agent makes that choice systematically, it is doing multi-step planning.

This post tests whether it does. The short answer is no: across behavioural, representational, and causal-intervention experiments, the agent looks like a reactive heuristic policy that benefits from continuous control and lidar-based avoidance. The longer answer is below.

This is joint work with Jonathan Richens (Google DeepMind); the framing draws on [*General Agents Need World Models*](https://arxiv.org/abs/2506.01622). Mathias Jackermeier was generous with his time, and several of the training interventions described here were his suggestions.

**Summary.**

- On a custom environment where the myopic and optimal first sub-goal disagree by construction, the agent picks optimally **~50%** of the time. Task success on the same layouts is **93%**: about **43% of episodes succeed via the non-optimal path**. The above-chance signal we initially saw (58%) is fully explained by a **forward-motion bias**; when orientation is controlled, the residual disappears.
- Probing the actor stack shows representations are progressively reshaped from environment-relevant to action-relevant: sensors fade, policy-aligned features rise, the goal-colour identifier is preserved through `H3` at 98% accuracy, and the agent's *bearing to the goal* is at chance throughout. There is no linearly decodable metric map and no clean self-centric goal-direction feature.
- The goal pointer is decodable at 100% in `H3`, but **adding the probe direction at the actor MLP changes behaviour in 1% of trials**; replacing the planner's output (the LTL embedding) **redirects the agent in 87% of trials**. The actionable copy of the goal lives at the planner-actor bottleneck, not in the actor's distributed redundant copy. Probes find presence, not use.
- Training interventions designed to induce planning — auxiliary chained-distance loss, transition loss, lower discount, two-step-heavy curriculum — do not move the optimality rate. Auxiliary supervision *does* raise the chained-distance probe R²; behaviour does not follow.

The evidence converges on a heuristic account: forward motion, then closest relevant zone, then reactive obstacle avoidance — a stack that produces 95% completion on a completion-scored benchmark without any internal world model.

---

## 1. The setup

DeepLTL has three modules. An **LTL encoder** compiles the formula into a deterministic-Büchi automaton, summarised by a small GRU over its states. An **environment encoder** is an MLP over lidar observations. A **policy head** (called `actor.enc` in the codebase, a 3-layer MLP) concatenates the two and outputs continuous actions for a point robot in the Zone environment — a flat arena of coloured circular zones.

![Architecture of the DeepLTL agent. The LTL formula is compiled into automaton states and embedded by a DeepSets+GRU planner; lidar observations are embedded by an MLP. The two embeddings are concatenated to form the 96-d input (`IN`) to a 3-layer actor MLP (`H1`, `H2`, `H3`), which produces the policy mean μ. The post probes and intervenes at every named layer.](figures/00_architecture.png)

The cleanest test of planning is `F blue THEN F green` with *two* candidate blue zones. A planner picks the blue zone that leaves less distance to green. A reactive "closest zone" policy picks whichever blue is closer right now. The paper's Figure 1 shows the former.

I built `PointLtl2-v0.optvar`, a small modification of the paper's Zone env in which this divergence is enforced at every reset: the closer blue zone is always the *worse* one for total path length. The agent and physics are otherwise unchanged from the paper's setup.

![The `F blue THEN F green` task, schematically. The two blue zones are candidate intermediates: the closer one leaves a longer onward path to green. A planner picks the farther one; a reactive "closest zone" policy picks the nearer one.](figures/01_setup_map.png)

If the agent plans, it should pick the optimal (farther) blue consistently. If it doesn't, it should pick the myopic (closer) blue consistently. Either pattern would be informative.

---

## 2. Behaviour: at chance

Across 100 contested layouts on `optvar`, with `fresh_baseline` (a faithful re-train of the paper setup):

| | Rate |
|---|---|
| Optimal-choice rate | **50%** |
| Myopic-choice rate | 50% |
| Task success rate | 93% |

![Optimality-test trajectories on `optvar`. Green = optimal choice, orange = myopic.](figures/02_optvar_trajectories.png)

That is high task success at chance-level sub-goal selection. The agent reaches green almost always; its choice between candidate blues is essentially a coin flip.

A reasonable concern is that "optimal" might be defined wrong. Geometric distance is one definition; *empirical difficulty* — the success rate when each candidate is forced — is another. Under the empirical definition, the agent picks the easier candidate 68% of the time, with a confidence interval that excludes 50%. The next two checks show why this is illusory.

### 2a. Equidistant control

If the agent considers full path length, equating the *agent-to-blue* distances should not affect performance. In `PointLtl2-v0.opteq` the two blues are placed at the same distance from the agent (tolerance 0.05).

| | fresh_baseline |
|---|---|
| Optimal-choice rate | **54%** |
| 95% CI | [40%, 70%] |

![Equidistant variant: the two candidates are placed at the same distance from the agent.](figures/03_opteq_trajectories.png)

The 68% empirical-difficulty signal therefore was not driven by full-path reasoning. It was driven by something correlated with full-path difficulty in the optvar layouts but absent in opteq. The simplest such correlate is "closer to the agent".

### 2b. Orientation control

A separate confound was visible in the data. fresh_baseline preferred the `x < 0` zone 66% of the time; a sibling auxiliary-loss model preferred `x > 0` 61% of the time. Two models with opposite spatial biases is itself weak evidence against planning, but it does not identify the rule.

Logging the agent's initial heading made the rule visible:

- Forward-motion preference: **74%** (p < 0.0001).
- When only one candidate lies in the forward half-plane: **80%** chosen.

![Forward-motion preference, not left/right position, is the real bias.](figures/04_orientation_bias.png)

I repeated the optimality test with orientation controlled: at each reset the agent was rotated to face the midpoint between the two blues, so neither was more "forward":

| Model | Optimal | 95% CI | p |
|---|---|---|---|
| fresh_baseline | **58%** | [48%, 68%] | 0.125 |
| opt_d099_mixed | 52% | [42%, 62%] | 0.764 |

The CI contains 50%. The spatial imbalance vanishes exactly, because the imbalance was never about left/right — it was about forward. With both forward-motion and closest-zone cues neutralised, the agent's choice is statistically indistinguishable from random.

The behavioural picture is settled: at chance. Two natural follow-up questions are *what* the agent encodes internally and *what* it actually uses. The next two sections take those in turn.

---

## 3. Representations: what the agent encodes

### 3.1 Information flow through the actor stack

The actor is a 3-layer MLP with ReLU activations: a 96-d combined embedding (`IN`) is mapped through three 64-d hidden layers (`H1`, `H2`, `H3`) before a final linear head produces the policy mean (`OUT`). I trained linear probes (Ridge for regression, logistic for classification) at each depth on a representative set of targets. Splits are episode-disjoint: 64 of 80 layouts in training, 16 held out.

![Probing the actor stack on `PointLtl2-v0`. Sensors (top) fade through the layers; policy-aligned features (middle) rise; the goal-colour identifier is preserved nearly perfectly through `H3` and drops at the output; geometry features (bottom, negative controls) are not linearly decodable anywhere.](figures/13_actor_stack.png)

A few things stand out.

- **Sensors fade.** `agent_speed` is well-encoded at the input (R² ≈ 0.77) and fades to ≈ 0 by the output. The MLP discards raw sensor information as it computes its action.
- **Policy-aligned features rise.** Whether the agent is about to turn left or right is decodable at 66% accuracy at `IN`, 100% at `OUT`. The 8-class action-direction lift is similarly steep (59% → 96%).
- **The goal-colour identifier is preserved with near-perfect accuracy through `H3` (≥ 98%), then drops sharply at `OUT` (38%).** The actor carries the goal pointer all the way to the last hidden layer and discards it once the action is committed.
- **Geometry features are not linearly decodable.** Absolute agent position and one-step Δx have negative R² across every layer. The actor does not store a metric map of the environment.

The shape of the gradient — environment-relevant features at the top, action-relevant features at the bottom, no metric map at any depth — is the picture of an *egocentric, reactive controller* whose representations are progressively shaped toward action.

**One row deserves attention on its own.** The 8-class self-centric *bearing to the goal zone* is at chance (≈ 10%) at every layer of the actor. The agent's eventual *policy angle* — the direction it is about to move in — is decodable at 95% by `H3`. The two facts together rule out the simplest mechanistic story for how the agent moves toward a goal: it is *not* the case that the network decodes bearing-to-goal into a clean linear feature and then acts on it. Whatever computation produces the policy direction does not pass through "bearing to goal" as an intermediate; it is either non-linear in this representation, or it is something other than goal-bearing — an obstacle-modulated heading, say, or a learned routine that hits goals as a side-effect of moving in some currently-attractive direction. (We will return to this in §6.)

### 3.2 The LTL planner's output is a stable state machine

DeepLTL has a small GRU on top of the deterministic-Büchi automaton's state sequence. At every step the network re-encodes the current LTL plan from scratch and produces a 32-d "LTL embedding". On a 2-step task `F a THEN F b`, this embedding is *constant* during pursuit of `a`, then jumps when the LDBA transitions to pursuing `b`.

The interesting question isn't how predictable the embedding is during pursuit — it's flat by construction — but whether the rare *jumps* are predictable from the combined embedding `h_t` alone. They are: R² = 0.97 across goal-switch steps, with no additional signal from the action.

![LTL-embedding dynamics. During pursuit, Δltl ≈ 0 by construction — the planner output is locked while the agent chases the current sub-goal. At a goal-switch step, the embedding jumps to a colour-dependent vector that is linearly predictable from `h_t` (R² ≈ 0.97). Action gives no extra information.](figures/14_gru_dynamics.png)

The picture is a near-linear state machine that retargets at switches: knowing the current state of the world is enough to predict the next planner output, even right at the moment it changes.

### 3.3 Chained distance is not linearly decodable on held-out layouts

The actor's hidden states linearly encode the goal pointer, the agent's lidar readings, and a developing action. Are they also able to express the *path-level* feature a planner would need — the chained distance from a candidate intermediate zone to the goal zone?

A first pass with random train/test splits across states gave a familiar result: self-centric features at R² ≈ 0.55, chained features at R² ≈ 0.22 — present but weak, consistent with weak planning. But that random split has a pitfall. Zone positions are constant within an episode, so any feature defined by the layout (`d_int_to_goal`, `total_via_int`, `optimality_gap`) is constant across all of an episode's states. A random split lets the probe see *some* states from a layout in training and predict *others* from the same layout at test time — recognising the episode rather than computing the feature.

Splitting by episode breaks this leak. With 64 of 80 layouts in training and 16 held out, the probe must generalise to layouts it has never seen.

![Random splits inflate probe R² for chained distances. Under episode-disjoint evaluation, chained features are below the mean-only baseline.](figures/10_honest_probing.png)

Self-centric features still survive, at lower R² (≈ 0.13 – 0.26): the probe finds something real and generalisable. Chained features do not — both `int₀ → goal` and `int₁ → goal` have *negative* R², worse than always predicting the training mean. The "0.08 – 0.18" chained-distance R² that some earlier analyses reported is essentially the within-episode recognition signal; the underlying feature is not linearly decodable from a held-out layout's hidden states.

The value function shows the same pattern. Holding the first sub-goal fixed and varying the second:

- *Same first target, different second targets?* V(state, [A, easy]) − V(state, [A, hard]) = **0.003**. Within noise.

![Marginal value of the second sub-goal. ΔV(easy) and ΔV(hard) are within noise of each other; V does not discriminate downstream difficulty.](figures/06_value_anticipation.png)

---

## 4. Causal: what the agent uses

Linear probes establish presence, not use. §3.1 showed that the goal colour is decodable at near-perfect accuracy through the actor's `H3` layer. Does that mean the goal is *steerable* by perturbing those activations? And what about the goal pointer at the planner-actor boundary?

### 4.1 Subgoal steering: the goal lives at the planner output, not in the actor MLP

I trained a logistic-regression classifier on `H3` to predict the current goal colour (the same target that probed at 100% in the previous section). The classifier achieves 100% in-sample accuracy. From its weight matrix I extracted a *steering vector* `w_target − w_source` for every (source → target) colour pair, and ran two interventions on a `F source` task:

- **MLP steering**: at every forward pass, add `α · ŵ` to the output of `actor.enc` (the H3 layer). α = 20.
- **Sequence-encoder swap**: replace the LTL embedding with the embedding the network produces for `F target`.

Both interventions target the same outcome (visit the *target* colour first). They differ in *where* the perturbation enters the network. Across 120 trials each:

![Subgoal steering. Adding the H3 classifier's source→target direction at α = 20 changes the visited colour in only **0.8%** of trials and breaks the policy in most others. Replacing the LTL embedding redirects the agent reliably (**87%**).](figures/16_subgoal_steering.png)

The goal colour is decodable in the actor MLP with perfect accuracy, but adding even a large perturbation along the probe direction does almost nothing — most of those trials simply fail to reach any zone, with 1 in 120 actually switching to the steered colour. Replacing the planner's output redirects the agent in 87% of trials. The actionable copy of the goal lives at the *bottleneck* between planner and actor, not in the distributed redundant copy that the probe reads from.

This is a clean example of probes establishing presence, not use.

### 4.2 Goal-switch interventions: the agent re-locks from observations

What happens at the moment a 2-step task transitions from `F a` to `F b`? If the GRU hidden state at the switch is doing important work — bridging from "pursued a" to "pursue b" — then perturbing the LTL embedding *exactly* at the switch step should hurt success. I ran four conditions on `F a THEN F b` tasks (80 episodes each):

| Condition | Task success |
|---|---|
| baseline | 99% |
| zero ltl_emb at switch, 1 step | 96% |
| zero ltl_emb at switch, 3 steps | 93% |
| random ltl_emb at switch, 1 step | 99% |

![Goal-switch interventions on `F a THEN F b` tasks. Replacing the LTL embedding with zeros at the switch step (or for three steps) only mildly degrades task success.](figures/18_goal_switch.png)

Task success is preserved across all conditions. Even three consecutive steps of zeroed-out LTL embedding only drops success by 6 percentage points. The agent re-locks on the new sub-goal from observations and the planner's *next* timestep output — the embedding at the precise switch step is not load-bearing. There is no "memory" between steps that has to be intact at the switch; the planner's output one step later, combined with current observations, is enough.

This complements §4.1: the planner's output is the actionable goal pointer, and even a brief interruption of it is recoverable.

### 4.3 A 1D activation ablation that didn't work

A separate test, narrower in target. I trained probes on the combined embedding for two features — `d_agent_to_int` (self-centric, episode-disjoint R² ≈ 0.55) and `d_int_to_goal` (chained, episode-disjoint R² < 0) — and tried project-out and sufficiency interventions along each probe direction.

The result was inconclusive. 1D ablation of *either* direction produced a similar ~10pp shift toward myopic behaviour with task success preserved at ~88%. The sufficiency sweep along the chained-distance direction was essentially flat. With a 96-dim embedding, removing a single 1D component is too weak a perturbation to test direction-specific use — the policy is locally robust to single-direction ablations regardless of which one. Multi-dimensional subspace projection would be the natural follow-up; the §4.1 steering result is the cleaner test of the same general question.

![Project-out: 1D ablations of either probe direction produce similar mild shifts (~10 percentage points) toward myopic behaviour; task success stays at ~88% across all conditions. The methodology is too coarse to be informative on this question.](figures/11_causal_ablation.png)

---

## 5. Training interventions don't help

Could planning be induced from the training side? Mathias suggested the relevant hypotheses: the default discount of 0.998 makes return differences between optimal and suboptimal sequences vanishingly small, and a curriculum that begins with one-step reach tasks conditions the agent on a "closest zone" prior before sequences for which it is suboptimal arrive.

I ran the corresponding sweep:

| Variant | Discount | Curriculum | Task success | Optimal | p |
|---|---|---|---|---|---|
| `fresh_baseline` | 0.998 | 1-step start | 91% | 58% | 0.125 |
| `extended_baseline` | 0.998 | 1-step (30M steps) | 95% | 59% | 0.093 |
| `twostep_lowdiscount` | 0.95 | 2-step only | 38% | unstable | — |
| `opt_d095_mixed` | 0.95 | 75% 2-step + 25% 1-step | 64% | unstable | — |
| `opt_d099_mixed` | **0.99** | mixed | **85%** | **52%** | **0.764** |
| aux loss 0.2 | 0.998 | baseline | 90% | ~50% | — |
| transition loss 0.1 | 0.998 | baseline | 89% | ~50% | — |
| combined aux + transition | 0.998 | baseline | 75% | ~50% | — |

![Auxiliary-loss training variants on `optvar`. Left: optimal-choice rate is at chance across all variants. Middle: chained-distance probe R² (random-split, comparable to the original analysis) rises from 0.31 to 0.41. Right: task success stays close to baseline.](figures/07_planning_incentives.png)

The pure two-step run at γ = 0.95 collapses. Of the curriculum/discount variants, only `opt_d099_mixed` reaches task success close to baseline (85% vs. paper's 91 – 95%), and its optimality rate is 52%, p = 0.76.

The most informative observation is the auxiliary-loss column. The chained-distance probe R² (random-split, used here for comparability with earlier analyses; the corresponding episode-disjoint R² is below zero throughout) rises from 0.31 to 0.41 — a real increase in probe-decodable content. Behavioural optimality across all variants in the table sits at ~50%. The network has acquired more of the information that planning would require, and the policy does not use it.

This pairs with the steering result in §4.1: whether the chained-distance feature is *added* by training or the goal pointer is *removed* from the actor MLP, behaviour does not follow the representation. Representation and policy have separated.

---

## 6. A heuristic account

The behaviour I observe is consistent with a small stack of reactive rules, applied roughly in this order:

1. Move forward, with probability ≈ 0.74 along the initial heading.
2. Otherwise, move toward the closest relevant zone.
3. Avoid obstacles via lidar; the binary "blocked" feature is well-encoded (~95%).
4. When none of (1) – (3) discriminates between candidates, choose approximately at random.

This stack accounts for 95% success on a benchmark that scores completion, and it accounts for chance-level sub-goal selection on a benchmark that scores planning specifically. The two are not contradictory — completion is robust to bad first commitments, given enough lidar and continuous control to recover.

The actor-stack analysis (§3.1) rules out a tidier alternative. If the agent acted by computing a self-centric "bearing to goal" and rotating to face it, that bearing should be linearly decodable somewhere in the actor — and it isn't, at any layer. What *is* linearly decodable is the goal-colour identifier and the network's own about-to-act direction. The most parsimonious reading is that the network learned a "this colour is currently attractive" → "rotate-and-go" routine that is largely independent of explicit relative-direction reasoning.

One implication of the heuristic account is that the agent's behaviour should be summarisable at the colour level by a Markov chain `P(reach next | reached current)` — without reference to anything inside the network. I estimated this 4×4 matrix from 12 episodes per pair on `F a THEN F b` tasks, then used it to predict success on length-2 and length-3 chains.

![Behavioural macro world model. Left: the pair-success matrix `P(reach next | reached current)`. Right: a Markov-chain prediction (multiplying matrix entries) is within MAE ≈ 0.03 of empirical multi-step success at depth 1, 2, 3.](figures/15_macro_world_model.png)

The prediction matches empirical multi-step success within MAE ≈ 0.02 – 0.03. Most matrix cells sit at 1.00 — the agent reliably succeeds on most pairs — so the test is a relatively easy one: a constant 1 would also predict reasonably. The cells that aren't saturated (`green→blue` and `green→magenta` at 0.92, `yellow→blue` at 0.92) are where the prediction has to work, and the depth-3 prediction (0.96 vs. empirical 0.93) is where it would most plausibly fail; it doesn't.

The point isn't that the agent has internalised this matrix — it almost certainly hasn't. The point is that the agent's macro structure is *recoverable from rollouts alone*. Combined with the absence of a linearly decodable metric map in the activations (§3.1), the natural reading is that the network executes a near-Markov colour policy, and any "world model" lives in the analyst's notebook, not in the network.

---

## 7. Scope

The paper's task-success numbers reproduce. The agent is reliably goal-conditioned at the behavioural level: across single-colour tasks `F c`, success is 98% trained vs. 0% under a uniform-random policy on the same layouts ([Maximum Entropy Goal-directedness](https://arxiv.org/abs/2310.07229) ≈ 3.2 averaged across colours). The post is not arguing against goal-conditioning as such — it is arguing against the *mechanistic* claim that sub-goal selection is driven by reasoning over onward paths. Mathias concurs that the optimality rate is approximately 50%; remaining disagreement, if any, concerns how surprising this should be considered.

A few caveats:

- This is one architecture trained one way. Other architectures — explicit successor features, MuZero-style world models, transformers over automaton states — may produce different results.
- Sample sizes are moderate (N = 80 – 100 per behavioural cell). Borderline results would benefit from larger N.
- The behavioural evidence concerns one task family, `F A THEN F B` and its safety/equidistant variants. Tasks with richer temporal structure may interact with a heuristic policy differently.
- Confounds have recurred. "Spatial bias" turned out to be orientation bias; "weak chained-distance encoding" turned out to be within-episode leak. There may be further confounds I have not yet identified.
- The paper's Figure 1 displays a single configuration in which the agent does pick the optimal blue zone. Across 100 randomly varied configurations of the same task family, the agent picks optimally about half the time. The post's quantitative claim is the average; it does not contest that the figure depicts a real successful trajectory.

The broader point worth stating plainly: high task success is not, by itself, evidence of planning, even on tasks that would in principle require it. A sufficiently rich reactive policy, combined with continuous control, lidar-based avoidance, and a benchmark that scores completion rather than optimality, can reach 95% success without instantiating anything that resembles a world model.

---

## Code and references

- This work: [`does-deep-ltl-plan`](https://github.com/benjibrcz/does-deep-ltl-plan). All scripts under `interpretability/`.
- Jackermeier & Abate, [*DeepLTL*](https://openreview.net/forum?id=9pW2J49flQ), ICLR 2025.
- Richens et al., [*General Agents Need World Models*](https://arxiv.org/abs/2506.01622), 2025.

Each result in the post is produced by a single script in the repo. The most direct reproductions:

- **§2 (behaviour at chance):** `interpretability/behavioural/controlled_orientation_test.py` — load `fresh_baseline`, N = 100 on `PointLtl2-v0.opteq`. Expected: ~50% optimal, CI containing chance.
- **§3.1 (actor-stack flow):** `interpretability/probing/actor_stack.py`. Outputs `13_actor_stack.png`.
- **§3.2 (LTL dynamics):** `interpretability/probing/gru_dynamics.py`.
- **§3.3 (honest-split probing):** `interpretability/probing/honest_split_comparison.py`.
- **§4.1 (subgoal steering):** `interpretability/causal/subgoal_steering.py`.
- **§4.2 (goal-switch interventions):** `interpretability/causal/goal_switch_intervene.py`.
- **§6 (macro world model):** `interpretability/behavioural/macro_world_model.py`.

A substantial deviation from any of the headline numbers would be informative.
