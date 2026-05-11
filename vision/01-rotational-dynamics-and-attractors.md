# Vision: Rotational Dynamics, Strange Loops, and Attractor Splitting

**Purpose:** Capture the research path where HexNets are studied as iterated dynamical systems rather than one-shot feedforward predictors.

This document groups together:

- strange-loop inspired recurrence
- latent orbit tracing
- fixed points
- limit cycles
- attractor splitting
- persona-like regimes
- loop stability
- directional lesioning
- interference between independently trained loops

This is not a replacement for benchmark families. It is an exploratory research path that asks what kinds of internal trajectories HexNets can sustain when their outputs are repeatedly fed back into their inputs.

---

## 1. Core Idea

A standard benchmark asks:

```text
Given input x, does the model predict target y?
```

This research path asks:

```text
Given an initial state x_0, what happens when the HexNet keeps transforming itself?
```

A basic recurrence could be written:

```text
x_{t+1} = f(x_t)
```

A HexNet-specific recurrence can include directional structure:

```text
x_{t+1} = R_k(f_d(x_t))
```

Where:

| Symbol | Meaning |
|---|---|
| `x_t` | current latent vector at step `t` |
| `f_d` | HexNet pass through direction `d` |
| `R_k` | rotation / remapping operator |
| `k` | selected rotation step |
| `t` | recurrence step |

The output of one pass becomes the input to the next. Over many steps, the vector traces a path through latent space.

---

## 2. Working Names

### Short / demo-friendly

- HexLoops
- Strange Hex Loops
- Rotational Loops
- Latent Hex Orbits
- Orbit HexNets

### Technical

- Rotational Latent Orbit Dynamics
- Directional Attractor Dynamics
- HexNet Recurrence Dynamics
- Structured Latent Recurrence
- Rotational Projection Dynamics

### Recommended file / paper phrase

> **Rotational Latent Orbit Dynamics**

This phrase is technical enough for research notes while still preserving the core image: a vector orbiting under repeated directional transformations.

---

## 3. Why This Belongs With Attractor Splitting

Earlier benchmark concepts such as “attractor splitting” and “persona formation” fit here because they are not primarily about static task performance.

They are about whether the network can form:

- stable internal regimes
- separate basins of attraction
- different recurring trajectories
- state-dependent behaviors
- direction-dependent response modes

A “persona-like” regime should be treated carefully. In this project, it should not initially mean human personality. It should mean:

> A stable, distinguishable behavioral regime that persists under recurrence and responds differently to perturbation or task input.

That is a dynamical claim, not a psychological one.

---

## 4. Main Research Questions

### 4.1 Fixed Points

Does iteration collapse into a stable vector?

```text
f(x*) ≈ x*
```

Questions:

- Do trained HexNets produce more fixed points than untrained HexNets?
- Are fixed points rotation-specific?
- Are fixed points task-dependent?
- Do fixed points correspond to meaningful target structures?

Possible interpretation:

- fixed point = stable memory state
- collapse to fixed point = loss of expressive trajectory
- multiple fixed points = possible basin structure

---

### 4.2 Limit Cycles

Does iteration produce repeating loops?

```text
x_0 → x_1 → x_2 → ... → x_p → x_0
```

Questions:

- Do loops emerge naturally?
- Are loops exact or approximate?
- Are shorter loops more common than longer loops?
- Does rotation schedule determine period length?
- Does training stabilize or destroy loops?

Possible interpretation:

- limit cycle = recurrent latent behavior
- stable cycle = memory-like repeated state sequence
- unstable cycle = transient computational path

---

### 4.3 Quasi-Periodic Orbits

Does the trajectory nearly repeat without closing perfectly?

Questions:

- Does the trajectory remain bounded?
- Does it drift slowly?
- Does it revisit similar states?
- Does it fill a region of latent space?

Possible interpretation:

- not all useful loops must be exact
- approximate recurrence may be more realistic than perfect cycles
- near-cycles may encode flexible memory

---

### 4.4 Strange Attractor-Like Behavior

Does the trajectory remain bounded but non-repeating and sensitive to perturbation?

Questions:

- Are trajectories sensitive to small changes in initial state?
- Do nearby paths diverge and then remain within a bounded region?
- Does direction switching increase complexity?
- Do nonlinear activations produce richer behavior than linear activations?

Caution:

- Do not call something a “strange attractor” casually.
- First use softer language: “bounded non-repeating trajectory” or “attractor-like region.”
- Only escalate if there is actual evidence of sensitive dependence and structured recurrence.

---

### 4.5 Attractor Splitting

Can one trained HexNet sustain multiple stable regimes?

Questions:

- Do different initial vectors converge to different states?
- Do different rotations produce different basins?
- Does training on two tasks create two attractors?
- Does one attractor dominate the other?
- Can attractors merge under further training?

Possible framing:

```text
Same architecture, same weights, different initial state or rotation schedule → different stable behavior.
```

This is one of the more important bridges to the “persona” intuition.

---

### 4.6 Loop Independence

Can two loops exist without interfering?

Questions:

- Can loop A and loop B occupy disjoint latent regions?
- If loop A is trained further, does loop B remain stable?
- Does perturbing loop A ever send it into loop B?
- Are loops separated by clear basin boundaries?
- Do different directions protect loops from each other?

Potential result types:

| Observation | Interpretation |
|---|---|
| Loop A and B stable under separate perturbations | possible independent attractors |
| Training A destroys B | catastrophic interference |
| Training A shifts B but does not destroy it | coupled attractor drift |
| Perturbing A jumps to B | basin boundary is close or unstable |

---

## 5. Rotation Schedules

The recurrence should not only test “feed output back into input.” HexNet’s distinctive property is directional structure.

### 5.1 No Rotation

```text
x_{t+1} = f_0(x_t)
```

Use as baseline.

Questions:

- Does a single direction collapse?
- Does it cycle?
- Does it diverge?

---

### 5.2 Fixed Directional Rotation

```text
x_{t+1} = R_1(f(x_t))
```

or

```text
direction sequence: 0, 1, 2, 3, 4, 5, 0, ...
```

Questions:

- Does steady rotation create orbital structure?
- Does six-step periodicity appear?
- Is the period tied to hexagonal symmetry?

---

### 5.3 Alternating Opposites

```text
direction sequence: 0, 3, 0, 3, ...
```

Questions:

- Does forward/backward opposition stabilize?
- Does it cancel drift?
- Does it behave like a projection-relaxation process?

---

### 5.4 Alternating Every Other Direction

```text
direction sequence: 0, 2, 4, 0, 2, 4, ...
```

Questions:

- Does the even-rotation triangle form a stable sub-dynamics?
- Does it differ from `1, 3, 5`?
- Does it connect to the directional transfer experiments?

---

### 5.5 Random Rotation Schedule

```text
direction sequence: random choice from {0,1,2,3,4,5}
```

Questions:

- Does random direction act like noise?
- Does the trajectory remain bounded?
- Are some schedules more stable than others?
- Does training produce rotation-invariant stability?

---

### 5.6 Learned or Adaptive Rotation Schedule

Later-stage idea:

```text
direction d_t is selected by a policy or heuristic
```

Questions:

- Can the model learn which direction to apply next?
- Does adaptive direction selection improve stability?
- Does it discover loop-preserving schedules?

This should be deferred until fixed schedules are understood.

---

## 6. Initial Conditions

The “null prompt” idea should become a family of controlled initial states.

### 6.1 Zero Vector

```text
x_0 = 0
```

Useful for:

- canonical starting point
- comparing trained vs untrained behavior
- detecting bias-driven dynamics

Issue:

- if no biases exist or all operations preserve zero, this may be uninformative.

---

### 6.2 Constant Vector

```text
x_0 = c * 1
```

Useful for:

- symmetric start
- checking whether the model breaks symmetry
- testing magnitude effects

---

### 6.3 Random Unit Vector

```text
x_0 ~ uniform sphere
```

Useful for:

- basin mapping
- robustness
- multiple-seed trajectory analysis

---

### 6.4 Dataset-Derived Vector

```text
x_0 = sample input from benchmark dataset
```

Useful for:

- linking orbit behavior to task behavior
- seeing whether real inputs converge differently than arbitrary vectors

---

### 6.5 Learned Seed Vector

```text
x_0 is optimized to produce a loop or target behavior
```

Later-stage idea.

Questions:

- Can the system learn an initial “prompt” that enters a desired attractor?
- Are loop seeds sparse, common, or fragile?

---

## 7. Measurements

### 7.1 Recurrence Distance

For each pair of timesteps:

```text
D[i,j] = ||x_i - x_j||
```

Use this to detect:

- repeated states
- approximate cycles
- drift
- clustering
- basin transitions

Plot:

- recurrence-distance heatmap
- nearest-return distance vs timestep

---

### 7.2 Cycle Error

For candidate period `p`:

```text
cycle_error(p) = mean_t ||x_t - x_{t+p}||
```

Low cycle error suggests periodicity.

Track:

- best period
- period stability
- cycle error over training epochs
- cycle error under perturbation

---

### 7.3 Drift

```text
drift(t) = ||x_t - x_0||
```

Useful for:

- collapse
- divergence
- slow migration
- bounded wandering

---

### 7.4 Norm Growth

```text
norm(t) = ||x_t||
```

Useful for:

- exploding trajectories
- vanishing trajectories
- activation saturation
- boundedness

---

### 7.5 Basin Assignment

Run many initial vectors and assign each to:

- fixed point
- loop
- divergent path
- bounded non-repeating region
- unknown

Output:

| Initial Region | Final Behavior |
|---|---|
| random seed cluster A | loop 1 |
| random seed cluster B | fixed point |
| random seed cluster C | divergent |

---

### 7.6 Perturbation Recovery

Given a point on a loop:

```text
x'_t = x_t + ε
```

Then iterate and measure whether it returns to the loop.

Recovery metrics:

- distance to original loop over time
- time-to-return
- final basin assignment
- probability of loop escape

---

### 7.7 Loop Separation

For two loops `A` and `B`:

```text
separation = mean distance between points on A and nearest points on B
```

Also track:

- perturbation needed to jump from A to B
- training update magnitude needed to destabilize B
- shared dimensions / principal components

---

## 8. Visualization Ideas

### 8.1 PCA / UMAP Trajectory Plot

Project high-dimensional states into 2D or 3D.

Show:

- trajectory line
- start point
- end point
- rotation color
- timestep gradient

Use cautiously. Projection can distort loop structure.

---

### 8.2 Recurrence Heatmap

Plot `D[i,j] = ||x_i - x_j||`.

This is likely the cleanest visualization for loop detection.

Expected patterns:

| Pattern | Meaning |
|---|---|
| dark diagonal only | no recurrence |
| parallel dark diagonals | periodic behavior |
| block structure | regime switching |
| broad dark areas | collapse / fixed point |
| irregular repeated dark patches | quasi-periodic recurrence |

---

### 8.3 Norm-vs-Time Plot

Shows whether trajectories explode, vanish, or stabilize.

---

### 8.4 Direction-Colored Orbit

Color each point by direction used at that step.

Useful for seeing whether rotation schedule induces visible segmentation.

---

### 8.5 Basin Map

For low-dimensional cases, sweep initial states over a grid and color by final attractor assignment.

This could become one of the strongest figures if the behavior is clean.

---

## 9. Candidate Experiment Slates

### Slate 1 — Untrained Dynamics

| Field | Values |
|---|---|
| Model | HexNet |
| Weights | random initialization |
| Activation | linear, leaky_relu, sigmoid |
| Rotation schedules | none, fixed cycle, alternating opposite |
| Initial states | zero, constant, random unit |
| Steps | 100–1000 |

Goal:

- understand baseline dynamics before training
- identify whether architecture alone creates stable behavior

---

### Slate 2 — Trained Benchmark Dynamics

| Field | Values |
|---|---|
| Model | HexNet trained on benchmark tasks |
| Datasets | identity, orthogonal_rotation, sine, simplex_projection |
| Activation | best from benchmark runs |
| Rotation schedules | none, 0→1→2→3→4→5, 0→2→4 |
| Initial states | dataset samples, random unit vectors |

Goal:

- compare dynamics after task training
- see whether task structure shapes attractors

---

### Slate 3 — Loop Search

| Field | Values |
|---|---|
| Initial states | many random seeds |
| Candidate periods | 1–24 |
| Metric | cycle error |
| Keep | lowest-error loops |

Goal:

- search for fixed points and approximate cycles
- compare frequency across activations and training tasks

---

### Slate 4 — Perturbation Stability

| Field | Values |
|---|---|
| Base trajectories | detected loops |
| Perturbation size | epsilon ladder |
| Perturbation location | random timestep on loop |
| Metric | recovery probability, time-to-return |

Goal:

- determine whether loops are stable attractors or accidental recurrences

---

### Slate 5 — Cross-Loop Interference

| Field | Values |
|---|---|
| Start | model with two detected loops |
| Train | additional task or loop-specific objective |
| Measure | stability of both loops before and after |

Goal:

- test whether independent latent regimes can coexist
- measure catastrophic interference

---

## 10. Directional Lesioning Within This Track

Directional lesioning belongs here when the question is not simply “does performance drop?” but:

> Which attractors, loops, or regimes are destroyed by removing a direction, edge family, ring, or sector?

Lesion types:

| Lesion | Question |
|---|---|
| remove one direction | does a specific loop depend on that direction? |
| remove opposite pair | are forward/backward dynamics coupled? |
| remove ring | are attractors radially localized? |
| remove sector | are regimes spatially localized? |
| zero subset of weights | how distributed is the loop? |

Possible result:

- loop A depends on direction 0
- loop B depends on direction 3
- both share the center
- outer rings stabilize recurrence
- inner core stores attractor identity

This connects naturally to nested-core experiments.

---

## 11. Relationship to “Persona” Language

The word “persona” is tempting but risky.

Recommended progression:

1. **Regime**
2. **Attractor**
3. **Behavioral mode**
4. **Persona-like regime** only after evidence

A responsible claim would be:

> The model exhibits distinct stable behavioral regimes under different rotation schedules or initial conditions.

A premature claim would be:

> The model has multiple personalities.

The second may be useful as a provocative title or metaphor, but not as the first scientific claim.

---

## 12. Failure Modes

Failures are still informative.

| Failure | Meaning |
|---|---|
| all trajectories collapse to zero | recurrence is too contractive |
| all trajectories explode | recurrence is unstable |
| no loops detected | architecture may not support recurrence without special training |
| loops exist but are not stable | apparent recurrence is not attractor behavior |
| training destroys all loops | task optimization suppresses internal dynamics |
| loops only appear under sigmoid | activation saturation may dominate |

These should be documented rather than hidden.

---

## 13. Possible Future Objectives

Later, once passive dynamics are understood, train directly for loop behavior.

### 13.1 Fixed Point Objective

```text
L = ||f(x) - x||
```

### 13.2 Cycle Objective

```text
L = ||F^p(x) - x||
```

Where `F` is repeated application of HexNet plus rotation schedule.

### 13.3 Attractor Separation Objective

Encourage two seeds to converge to different stable regimes:

```text
L = stability_loss + separation_loss
```

### 13.4 Perturbation Robustness Objective

Train loop to recover after noise:

```text
L = distance_to_loop_after_perturbation
```

These are later-stage. First observe natural behavior.

---

## 14. Minimal Implementation Plan

### Step 1 — Trajectory Runner

Function:

```text
run_orbit(model, x0, schedule, steps) -> states, directions
```

Should support:

- direction schedule
- rotation schedule
- recording full states
- optional noise
- optional lesion mask

---

### Step 2 — Metrics

Implement:

- recurrence-distance matrix
- best cycle period
- norm curve
- drift curve
- nearest-return statistics

---

### Step 3 — Plots

Generate:

- trajectory projection
- recurrence heatmap
- norm vs time
- cycle error vs period

---

### Step 4 — Compare Trained vs Untrained

Run same orbit suite on:

- random HexNet
- HexNet trained on identity
- HexNet trained on orthogonal rotation
- HexNet trained on sine
- HexNet trained on simplex projection

---

### Step 5 — Perturbation Test

Given a candidate loop:

- add noise at timestep
- continue iteration
- measure return or escape

---

## 15. Paper-Facing Claim Ladder

Start with conservative claims.

| Evidence Level | Claim |
|---|---|
| trajectories bounded | HexNet recurrence can produce stable bounded dynamics |
| recurrence heatmaps show periodicity | some schedules produce approximate cycles |
| perturbations return to cycle | cycles behave as attractors |
| multiple stable cycles exist | model supports attractor splitting |
| training one cycle preserves another | loops can be partially independent |
| lesions selectively destroy one cycle | attractors are directionally or spatially localized |

---

## 16. Summary

This research path studies HexNet as a recurrent geometric system.

The important shift is:

```text
input-output mapping → latent orbit behavior
```

The central objects are:

- fixed points
- cycles
- quasi-cycles
- attractor basins
- perturbation stability
- loop independence
- directional lesions

This is the strongest home for strange-loop ideas because it converts the metaphor into measurable dynamical behavior.
