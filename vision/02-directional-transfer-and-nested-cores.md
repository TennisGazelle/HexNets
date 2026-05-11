# Vision: Directional Transfer and Nested HexNet Cores

**Purpose:** Capture the research path where HexNet geometry is tested for transfer, interference, modularity, and scale-recursive behavior.

This document groups together:

- train some rotations, test other rotations
- directional generalization
- directional interference
- alternating-rotation learning
- trained inner-core replacement
- nested smaller HexNets inside larger HexNets
- frozen vs trainable core behavior
- optimal core-radius ratio experiments

This track is closer to becoming a future benchmark family than the ant-maze or strange-loop tracks. It is still kept separate for now because it requires HexNet-specific instrumentation beyond ordinary dataset sweeps.

---

## 1. Core Idea

The baseline benchmark families compare HexNet against MLP on task families.

This track asks:

> If one part of the HexNet learns, what happens to the other directions and scales?

The key intuition is that HexNet geometry may create structured coupling across:

- rotations
- opposite directions
- alternating directions
- radial layers
- inner and outer regions
- smaller and larger HexNets

If these couplings are measurable, then HexNet has behavior that is not captured by ordinary “model vs dataset” comparisons.

---

## 2. Working Names

### Short / internal names

- Directional Transfer
- Rotational Transfer
- HexCore Transplant
- Nested Hex Cores
- Core Seeding
- Train-Then-Grow

### More technical names

- Directional Transfer Matrix
- Rotational Generalization Analysis
- Nested-Core Initialization
- Radial Transfer in Hexagonal Networks
- Scale-Recursive HexNet Training

### Recommended track name

> **Directional Transfer and Nested-Core Modularity**

This captures both halves:

1. learning across directions
2. learning across scale

---

## 3. Research Questions

### 3.1 Directional Transfer

If rotations `0, 2, 4` are trained, do rotations `1, 3, 5` change?

Possible outcomes:

| Outcome | Interpretation |
|---|---|
| untrained rotations improve | passive directional generalization |
| untrained rotations degrade | destructive directional interference |
| untrained rotations unchanged | directions are mostly independent |
| some improve, some degrade | asymmetric coupling |
| opposite directions change most | forward/backward relationship matters |
| neighboring directions change most | local rotational adjacency matters |

This is the central question that motivated this file.

---

### 3.2 Directional Specialization

Can different rotations specialize to different functions?

Questions:

- Can rotation 0 learn task A while rotation 3 learns task B?
- Does training rotation 0 alter task B performance on rotation 3?
- Are opposite directions more coupled than adjacent directions?
- Can rotation-specific specialization survive joint training?

---

### 3.3 Directional Interference

If one direction is trained after another, what is forgotten?

Questions:

- Does training direction 1 after direction 0 erase direction 0?
- Does alternating training reduce interference?
- Does simultaneous multi-direction training stabilize shared structure?
- Does interference depend on task similarity?

---

### 3.4 Nested-Core Initialization

Can a trained smaller HexNet be inserted into the interior of a larger HexNet?

Example:

```text
train small HexNet radius r
insert into larger HexNet radius R
continue training larger model
```

Questions:

- Does this accelerate convergence?
- Does it improve final loss?
- Does it stabilize early training?
- Does the outer network preserve or overwrite the inner core?
- Is this better than random initialization?
- Does it depend on task family?

---

### 3.5 Optimal Core Ratio

Is there an optimal ratio:

```text
core_radius / full_radius
```

Candidate ratios:

```text
1/4, 1/3, 1/2, 2/3
```

Questions:

- Is a tiny trained core enough?
- Does a large core overconstrain the larger model?
- Does the best ratio depend on task complexity?
- Does the best ratio depend on activation?
- Does the best ratio depend on whether the core is frozen?

---

### 3.6 Frozen vs Trainable Core

After inserting a trained core:

| Mode | Meaning |
|---|---|
| frozen core | inner model is preserved exactly |
| trainable core | inner model can adapt |
| slow core | inner learning rate lower than outer learning rate |
| regularized core | core can change but is penalized for drifting |

Questions:

- Does the trained core act like reusable knowledge?
- Does outer-ring training need to modify the core?
- Does freezing protect useful structure or prevent adaptation?
- Does slow-core training create a stable compromise?

---

## 4. Directional Transfer Experiments

### 4.1 Train One Direction, Test All Directions

Procedure:

1. Choose one training direction `d_train`.
2. Train only on that direction.
3. Evaluate all directions `0..5`.

Output:

| Train Direction | Eval 0 | Eval 1 | Eval 2 | Eval 3 | Eval 4 | Eval 5 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | loss | loss | loss | loss | loss | loss |
| 1 | loss | loss | loss | loss | loss | loss |
| ... | ... | ... | ... | ... | ... | ... |

This becomes the simplest **directional transfer matrix**.

---

### 4.2 Train Alternating Directions

Train:

```text
0, 2, 4
```

Evaluate:

```text
0, 1, 2, 3, 4, 5
```

Then train:

```text
1, 3, 5
```

Evaluate all again.

Questions:

- Are even and odd rotational triads symmetric?
- Does one triad generalize better?
- Does training one triad implicitly prepare the other?
- Does training the second triad overwrite the first?

---

### 4.3 Train Opposite Pairs

Train:

```text
0 and 3
1 and 4
2 and 5
```

Evaluate all directions.

Questions:

- Are opposite directions paired in behavior?
- Does forward/backward equivalence show up empirically?
- Does training opposite pairs stabilize more than neighboring pairs?

---

### 4.4 Train Neighboring Pairs

Train:

```text
0 and 1
1 and 2
2 and 3
...
```

Questions:

- Are adjacent directions locally coupled?
- Does training neighboring directions produce smoother rotation generalization?
- Is adjacency more important than opposition?

---

### 4.5 Sequential Direction Training

Example:

```text
train 0 → evaluate all
train 1 → evaluate all
train 2 → evaluate all
...
```

Track:

- immediate gain on newly trained direction
- retention on previously trained directions
- passive effect on future directions

This is the directional analogue of continual learning.

---

## 5. Metrics for Directional Transfer

### 5.1 Transfer Gain

For an untrained evaluation direction `e`:

```text
transfer_gain(e) = loss_before(e) - loss_after_training_other_direction(e)
```

Positive = improvement.  
Negative = degradation.

---

### 5.2 Interference Score

For a previously trained direction `p`:

```text
interference(p) = loss_after_new_training(p) - loss_before_new_training(p)
```

Positive = forgetting / degradation.

---

### 5.3 Directional Symmetry Score

Compare directions that should be geometrically related:

```text
symmetry_error = mean |loss(d) - loss(d_opposite)|
```

or for triads:

```text
triad_error = mean difference among {0,2,4}
```

---

### 5.4 Transfer Matrix

Define matrix `T`:

```text
T[i,j] = effect on direction j after training direction i
```

This is probably the central visualization.

Patterns to look for:

| Pattern | Interpretation |
|---|---|
| strong diagonal only | directions independent |
| strong opposite entries | opposite-direction coupling |
| strong adjacent entries | local rotational coupling |
| broad improvement | shared global learning |
| broad degradation | destructive interference |
| asymmetric matrix | direction order or architecture asymmetry |

---

### 5.5 Retention Curve

During sequential direction training:

```text
retention(d, step) = performance on direction d after each training stage
```

Useful for continual-learning style plots.

---

## 6. Nested-Core Experiments

### 6.1 Train-Small-Then-Grow

Procedure:

1. Train a small HexNet with radius `r`.
2. Initialize a larger HexNet with radius `R`.
3. Copy small trained weights into the corresponding inner region.
4. Initialize the outer region randomly.
5. Train the large HexNet.
6. Compare to a fully random large HexNet.

Questions:

- Does the larger HexNet train faster?
- Does it reach lower loss?
- Does it avoid bad early regimes?
- Does the trained core survive?

---

### 6.2 Core Replacement Modes

| Mode | Description |
|---|---|
| random full model | baseline |
| trained core, random outer | main test |
| random core, trained outer impossible/control | sanity check if available |
| frozen trained core | test reuse |
| trainable trained core | test adaptation |
| slow-learning trained core | test preservation + adaptation |
| noisy trained core | test robustness of transplanted knowledge |

---

### 6.3 Core-Radius Sweep

For full radius `R`, test:

```text
r/R ∈ {1/4, 1/3, 1/2, 2/3}
```

If using discrete `n`, approximate as closely as possible.

Track:

- initial loss
- convergence speed
- final loss
- gradient norm in core vs outer ring
- amount of core drift
- task-specific benefit

---

### 6.4 Task Families for Core Tests

Start with benchmark families that have clean interpretation.

| Family | Why Use It |
|---|---|
| identity / linear_scale | sanity check; should not need complex core |
| orthogonal_rotation | geometry-aligned task |
| low_rank_linear | possible reusable structure |
| sine | nonlinear smooth behavior |
| simplex_projection | geometry / constraint behavior |

Avoid starting with `sort`. It may obscure whether core transfer failed or the task is simply difficult.

---

## 7. Metrics for Nested Cores

### 7.1 Convergence Speed

Compare epochs or steps needed to reach threshold loss:

```text
steps_to_threshold
```

Good for showing practical benefit even if final loss is similar.

---

### 7.2 Final Loss Delta

```text
final_delta = final_loss_random_large - final_loss_core_initialized
```

Positive means core initialization helped.

---

### 7.3 Core Drift

Measure how much the inserted core changes during large-model training:

```text
core_drift = ||W_core_after - W_core_before||
```

Track by:

- layer
- ring
- direction
- epoch

Interpretation:

| Drift | Meaning |
|---|---|
| low drift + good performance | core is reusable |
| high drift + good performance | core is useful initialization but must adapt |
| low drift + poor performance | frozen/preserved core may constrain learning |
| high drift + poor performance | core not useful or incompatible |

---

### 7.4 Radial Gradient Magnitude

Track gradient norms by ring:

```text
gradient_norm(ring)
```

Questions:

- Does the outer region learn first?
- Does training pressure concentrate at the boundary between core and outer region?
- Does the core act like an attractor, bottleneck, or scaffold?

---

### 7.5 Boundary Mismatch

When a small core is inserted into a larger network, the boundary between trained and random regions may be unstable.

Measure:

- weight norm discontinuity at boundary
- activation magnitude discontinuity
- gradient spike at boundary
- loss contribution associated with boundary paths

This may become important if core insertion initially looks unstable.

---

## 8. Visualizations

### 8.1 Directional Transfer Matrix Heatmap

Rows = trained direction.  
Columns = evaluated direction.  
Values = loss delta or transfer gain.

This should be the first major figure for directional transfer.

---

### 8.2 Sequential Direction Retention Plot

Line for each direction.

X-axis:

```text
training stage
```

Y-axis:

```text
loss or normalized performance
```

Shows forgetting and passive transfer.

---

### 8.3 Core-Ratio Sweep Plot

X-axis:

```text
core_radius / full_radius
```

Y-axis options:

- final loss
- steps to threshold
- core drift
- stability score

---

### 8.4 Radial Gradient Heatmap

Rows = rings.  
Columns = epochs.  
Value = gradient norm.

Useful for seeing whether training moves inward, outward, or boundary-first.

---

### 8.5 Core Preservation Plot

Plot:

```text
||W_core(t) - W_core(0)||
```

over training time.

Compare:

- frozen
- trainable
- slow core
- regularized core

---

## 9. Experimental Slates

### Slate 1 — One-Direction Transfer

| Field | Values |
|---|---|
| Dataset | identity, orthogonal_rotation, full_linear |
| Activation | linear |
| Loss | mean_squared_error |
| Train directions | one at a time |
| Eval directions | all six |
| Seeds | multiple |

Goal:

- establish basic transfer matrix
- sanity-check symmetry

---

### Slate 2 — Alternating Triads

| Field | Values |
|---|---|
| Train set A | `0,2,4` |
| Train set B | `1,3,5` |
| Eval | all directions |
| Dataset | orthogonal_rotation, sine, simplex_projection |
| Activation | benchmark winner |

Goal:

- answer the motivating even-vs-odd rotation question

---

### Slate 3 — Sequential Direction Continual Learning

| Field | Values |
|---|---|
| Sequence | `0→1→2→3→4→5` |
| Alternative | `0→2→4→1→3→5` |
| Eval | all directions after each stage |
| Metrics | retention, transfer gain, interference |

Goal:

- determine whether direction training causes continual-learning-like forgetting

---

### Slate 4 — Small-to-Large Core Initialization

| Field | Values |
|---|---|
| Small radius | several |
| Large radius | several |
| Core mode | frozen, trainable, slow |
| Dataset | identity, orthogonal_rotation, sine |
| Baseline | random large model |

Goal:

- test whether trained cores help larger HexNets train

---

### Slate 5 — Core Ratio Sweep

| Field | Values |
|---|---|
| Ratio | `1/4, 1/3, 1/2, 2/3` |
| Dataset | best candidates from Slate 4 |
| Metrics | convergence speed, final loss, drift |

Goal:

- identify whether an optimal replacement ratio exists

---

## 10. Relationship to Attractor Dynamics

This track connects to `01-rotational-dynamics-and-attractors.md` in two ways:

### 10.1 Directional Transfer May Predict Attractor Structure

If directions strongly transfer to each other, attractor loops may also be broadly shared.

If directions are independent, attractor loops may be direction-specific.

---

### 10.2 Nested Cores May Store Attractors

If inner cores preserve stable dynamics, then a trained small HexNet may act as:

- an attractor seed
- a reusable latent dynamical module
- a stable center around which a larger model adapts

This suggests later experiments:

- insert a core with known loop dynamics
- train the larger model
- test whether the loop survives
- lesion outer rings vs inner core

---

## 11. Relationship to Benchmark Families

This track can eventually become a new benchmark family.

Possible future name:

> **Family G — Directional Transfer and Nested Modularity**

But do not add it to the benchmark matrix until:

- direction-specific training/evaluation is implemented
- rotation instrumentation is reliable
- transfer metrics are stable
- at least one compact slate produces interpretable results

For now, keep it as a vision file.

---

## 12. Risks and Ambiguities

### 12.1 Apparent Transfer May Be Shared Weights

If directions share parameters, improvement in untrained directions may not be “geometric generalization” so much as ordinary shared-weight learning.

Mitigation:

- compare architectures with equivalent parameter sharing where possible
- compare random direction labels
- use ablated directional structure

---

### 12.2 Apparent Independence May Be Implementation Artifact

If directions are evaluated through isolated paths, they may seem independent because the implementation prevents interaction.

Mitigation:

- verify adjacency and rotation mapping
- inspect which parameters are shared or reused
- add unit tests for directional path overlap

---

### 12.3 Core Insertion May Be Boundary-Dominated

A trained core may fail because the boundary between trained and random regions is bad, not because nested cores are impossible.

Mitigation:

- test boundary smoothing
- initialize outer ring to small values
- freeze core initially, then unfreeze
- use gradual growth

---

### 12.4 Scale Mismatch

A small HexNet may learn representations that do not map cleanly into a larger HexNet.

Mitigation:

- start with simple tasks
- compare same dimensional input/output structures
- define explicit mapping from small to large coordinates
- inspect activation magnitudes at boundary

---

## 13. Implementation Notes

### 13.1 Needed Infrastructure

Directional transfer requires:

- ability to select training rotations
- ability to evaluate by rotation
- logging of direction-specific losses
- saving model checkpoints after each directional training stage
- transfer matrix generation

Nested cores require:

- mapping from small HexNet coordinates to large HexNet coordinates
- weight-copy utility
- frozen parameter masks
- ring/radius metadata
- radial gradient logging

---

### 13.2 Suggested CLI Concepts

Potential flags:

```text
--train-rotations 0,2,4
--eval-rotations all
--directional-report
--core-init-from path/to/small/model
--core-radius 2
--freeze-core
--core-lr-scale 0.1
--radial-metrics
```

Keep names aligned with existing CLI style when implemented.

---

## 14. Paper-Facing Claim Ladder

| Evidence Level | Claim |
|---|---|
| train-one/test-all matrix differs from diagonal-only | directions are coupled |
| even triad changes odd triad | alternating rotations transfer or interfere |
| sequential training causes forgetting | directional continual-learning behavior exists |
| trained core improves convergence | nested initialization is useful |
| trained core survives fine-tuning | learned inner structure is reusable |
| optimal ratio appears | radial scale matters |
| core lesion destroys behavior | inner core stores functionally important structure |

---

## 15. Summary

This track asks whether HexNet geometry creates meaningful learning relationships across direction and scale.

The most important early deliverables are:

1. a directional transfer matrix
2. an even-vs-odd rotation experiment
3. a trained-core insertion experiment
4. a core-ratio sweep

The central question:

> Does HexNet learning remain local to the trained path, or does the geometry create structured transfer, interference, and reusable internal modules?
