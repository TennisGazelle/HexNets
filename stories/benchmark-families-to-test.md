# Benchmark strategy (families, not grids)

Do **not** start with the full Cartesian product of every activation, loss, dataset, dimension, and seed. 
Start with **benchmark families**, each designed to reveal a specific kind of inductive bias. 
The full experiment matrix lives in this document; CI exercises a **small representative subset** via [`e2e_test.sh`](../e2e_test.sh) (see [E2E coverage (non-exhaustive)](#e2e-coverage-non-exhaustive)).

**CLI naming** (this repo): losses use `mean_squared_error` (not informal “mse”), activations use `leaky_relu`, `log_cosh` for log-cosh, learning rates `constant` and `exponential_decay`. Dataset ids match the registry (see `src/data/*_dataset.py` and `list_registered_dataset_display_names()`).

Registry note: the codebase also registers `soft_threshold` and `sparse_identity`; they are not in the original ChatGPT slate below but are available for stress iterations.

---

## Family A — Linear structure recovery

### Datasets

- `identity`
- `linear_scale`
- `diagonal_scale`
- `diagonal_linear`
- `full_linear`
- `low_rank_linear`
- `affine`
- `orthogonal_rotation`

### What to vary

Keep this tight first:

- **Models:** HexNet vs MLP
- **Activations:** `linear`, then maybe `leaky_relu`
- **Losses:** `mean_squared_error`, `huber`
- **Learning rates:** `constant`, `exponential_decay`
- **Dimensions:** small ladder (e.g. d = 2, 4, 6, 8; CLI `-n`)
- **Seeds:** at least 5 (CLI `-s`)

### What this tells you

This family gives you your cleanest paper claims.

### Possible insights

#### Geometry is neutral on trivial mappings

If HexNet and MLP are basically the same on `identity` and `linear_scale`, that is not a failure. That is useful. You can say:

- On simple isotropic linear mappings, hex geometry does not impose a penalty.
- Architecture differences matter more on structured or anisotropic tasks than on trivial ones.

That is a strong calibration result.

#### HexNet may help more on structured anisotropy

If HexNet pulls ahead on `diagonal_scale`, `diagonal_linear`, `low_rank_linear`, then you can argue:

- The hex architecture may encode or preserve structured directional relationships better than a plain MLP.
- Performance differences emerge most clearly when the target map is not uniform across dimensions.

#### Rotations are especially important

If results differ on `orthogonal_rotation`, that is a very paper-friendly observation: the target itself is a rotationally structured transformation, so if HexNet benefits there, the interpretation is much cleaner than on arbitrary tasks. This is one of the best early “story datasets” in your suite.

---

## Family B — Smooth nonlinear regression

### Datasets

- `elementwise_power`
- `sine`
- Optionally `affine` as a bridge from linear to nonlinear

### What to vary

- **Activations:** `linear`, `relu`, `leaky_relu`, `sigmoid`
- **Losses:** `mean_squared_error`, `log_cosh`, `huber`
- **Learning rates:** `constant`, `exponential_decay` (“both”)
- Keep model, dimension, and seed consistent with Family A where you want comparability

### What this tells you

This family tests expressivity vs training stability.

### Possible insights

#### Stability vs raw accuracy

HexNet may not be “better,” but may be more stable on smooth nonlinear maps. Look for:

- Lower variance across seeds
- Faster convergence
- Less metric collapse at higher d

If the mean is similar but variance is better, that is publishable: not superiority of fit, but training stability under nonlinear structure.

#### Activation–task compatibility

Because the CLI warns that certain activations can distort regression outputs, you can test whether MLP, HexNet, or both are more sensitive to activation–task mismatch. A good paper sentence:

> The architectural effect was smaller than the activation–task compatibility effect on smooth bounded regression tasks.

---

## Family C — Projection and constraint tasks

### Datasets

- `unit_sphere_projection`
- `l2_ball_projection`
- `non_negative_projection`
- `simplex_projection`

### Why this family matters

These tasks are not just arbitrary nonlinearities. They are structured operators with geometric meaning.

### What to vary

- **Activations:** `linear`, `relu`, `leaky_relu`, `sigmoid`
- **Losses:** `mean_squared_error`, `huber`, `log_cosh` (skip `quantile` at first if you want a smaller grid)
- **Learning rates:** both schedules
- **Dimensions:** multiple values of `-n`

### What this tells you

#### Geometry-aware operators

If HexNet does better on projection tasks than on generic smooth tasks, that is a strong paper angle: the advantage may lie in structured geometric transformations, not merely “being nonlinear.” That is much more interesting than a raw leaderboard result.

#### Activation–task alignment

For example, `non_negative_projection` may pair better with ReLU-like outputs; `simplex_projection` and `unit_sphere_projection` may still prefer `linear` outputs with the loss doing the constraint work. If HexNet is more robust to activation mismatch on these tasks, that is a defensible robustness claim.

---

## Family D — Order / permutation / discontinuity

### Datasets

- `fixed_permutation`
- `sort`

These are especially valuable because they are harder to fake with smooth interpolation alone.

### What to vary

- **Activations:** start with `linear` and `leaky_relu`
- **Losses:** `mean_squared_error`, maybe `huber`
- **Learning rates:** both
- **Dimensions:** modest first

### What this tells you

#### Relational rearrangement

If HexNet does relatively better on permutation/sort than on vanilla linear tasks, that suggests its structure may be better suited to re-indexing or relational redistribution, not just scalar regression. Even if neither model performs brilliantly, the relative failure modes are interesting.

#### Architectural limits

If both models struggle badly on `sort`, that is also useful: some tasks require a different inductive bias entirely, and hex geometry is not a universal advantage. That helps keep the paper credible.

---

## Family E — Noise robustness

### Datasets (subset from A–D)

Use a smaller subset of the strongest datasets from other families, for example:

- `full_linear`
- `orthogonal_rotation`
- `sine`
- `simplex_projection`
- Optionally `sort` or `fixed_permutation`

### Noise modes (CLI)

- `--dataset-noise inputs`
- `--dataset-noise targets`
- `--dataset-noise both`
- Tune with `--dataset-noise-sigma` (and optional `--dataset-noise-mu`)

### What to vary

- **Sigma ladder:** e.g. 0.0, 0.05, 0.1, 0.2, 0.4
- **Losses:** `mean_squared_error`, `huber`, `log_cosh`
- **Activations:** keep the best 1–2 from earlier phases
- **Learning rates:** both
- **Seeds:** multiple

### What this tells you

This is probably the strongest Phase-2 comparison family.

#### Robustness curves

Plot performance vs noise level for both models. Possible findings: HexNet degrades more gracefully under input noise; MLP degrades differently under target noise; `huber` / `log_cosh` narrow the gap; architecture differences only appear after noise crosses a threshold. That gives richer paper language than a single score pair.

#### Geometry under corruption

A strong narrative: clean-data performance roughly tied, noisy-data performance diverges — therefore the benefit of hex geometry is not raw expressivity but robust structured propagation.

---

## Family F — Classification-style vector outputs

### Datasets

- `binary_vector_classification`
- `multi_label_linear`

### Why include them

Useful, but not necessarily the center of a first paper unless they are already very stable.

### What to vary

- **Activations:** `sigmoid`, `linear`
- **Losses:** whatever the CLI exposes (the default set is still regression-oriented for many losses)
- Compare architectures mainly for output separability and convergence behavior

### What this tells you

Good for a section like: “We also evaluated whether the architecture generalized beyond pure regression-like targets.” Secondary evidence, not necessarily the main result set.

---

## Specific benchmark iterations (suggested order)

### Iteration 1 — Clean linear benchmark

- **Datasets:** `identity`, `linear_scale`, `diagonal_scale`, `full_linear`, `orthogonal_rotation`
- **Activations:** `linear`
- **Losses:** `mean_squared_error`, `huber`
- **Learning rates:** `constant`, `exponential_decay`
- **Models:** HexNet, MLP
- **Dimensions:** 2, 4, 6, 8 (`-n`)
- **Seeds:** 5+

**Paper use:** baseline table, first convergence plots, first answer to “does HexNet help anywhere at all?”

### Iteration 2 — Smooth nonlinear benchmark

- **Datasets:** `elementwise_power`, `sine`
- **Activations:** `linear`, `relu`, `leaky_relu`, `sigmoid`
- **Losses:** `mean_squared_error`, `log_cosh`, `huber`
- **Learning rates:** both
- Same dimensions and seeds as Iteration 1 where comparable

**Paper use:** interaction plots (architecture × activation × dataset); activation sensitivity and nonlinear fit stability.

### Iteration 3 — Geometric operator benchmark

- **Datasets:** `unit_sphere_projection`, `l2_ball_projection`, `simplex_projection`, `non_negative_projection`
- **Activations:** `linear`, `relu`, `leaky_relu`
- **Losses:** `mean_squared_error`, `huber`, `log_cosh`
- **Learning rates:** both

**Paper use:** strongest “geometry-aware task” claim.

### Iteration 4 — Noise robustness benchmark

Take the best four datasets from Iterations 1–3. For each:

- **Noise modes:** inputs, targets, both
- **Sigma:** 0.0, 0.05, 0.1, 0.2, 0.4
- **Activations:** best 1–2 from earlier phases
- **Losses:** `mean_squared_error`, `huber`, maybe `log_cosh`

**Paper use:** degradation curves, robustness ranking, Huber / log-cosh vs MSE under corruption.

### Iteration 5 — Failure-case / stress benchmark

- **Datasets:** `sort`, `fixed_permutation`; optionally `low_rank_linear` at higher d; optionally `sparse_identity`

**Paper use:** clear limits, failure modes, future work. A paper is stronger when it shows where the method breaks.

---

## Interpretable insights to look for

1. **“HexNet is not universally better, but benefits appear on structured geometric transforms.”** Best supported by: `orthogonal_rotation`, projection datasets, maybe `low_rank_linear`.
2. **“On trivial linear tasks, geometry does not hurt.”** Best supported by: `identity`, `linear_scale` — shows the architecture is not mere ornamentation.
3. **“Activation–task compatibility matters as much as or more than architecture.”** The CLI already warns about bounded outputs vs regression; lean into that as a result, not a disappointment.
4. **“HexNet may offer robustness or stability rather than absolute best clean-data accuracy.”** Look for lower seed variance, better noisy-data degradation, fewer catastrophic runs.
5. **“Architecture effects are task-family dependent.”** Little difference on isotropic linear maps; moderate on structured linear operators; larger on geometry-preserving projections; ambiguous on discontinuities like sort/permutation.

---

## What to include in the paper (concrete)

### Core figures

- Baseline summary table: HexNet vs MLP across the clean linear family
- Convergence curves on a few representatives: trivial (`identity`), structured linear (`orthogonal_rotation`), projection (`simplex_projection` or `unit_sphere_projection`), nonlinear (`sine`)
- Noise robustness curves: performance vs sigma for 2–3 datasets
- Interaction heatmap: dataset × activation × model (with best loss fixed or shown separately)
- Failure-case panel: `sort` / `fixed_permutation`

### Core claims (restrained)

- HexNet matches MLP on simple linear mappings while showing distinct behavior on structured transforms.
- Where HexNet helps, it is task-dependent and appears strongest on geometry-aware operators rather than trivial isotropic maps.
- Activation/loss compatibility must be controlled before attributing differences to architecture.
- Under synthetic corruption, architectures may diverge in robustness — a story about perturbation, not only clean interpolation.

### What not to do yet

- Do not grid-search every activation × every loss × every dataset × every dimension × every seed at once.
- Do not lead with classification-style tasks as the main story.
- Do not over-interpret before you have curves.

**First milestone:** one clean linear family, one nonlinear family, one geometry/operator family, one noise family — enough for a first real results section.

---

## Recommended first comparison slates

### Slate A

- **Datasets:** `identity`, `linear_scale`, `diagonal_scale`, `full_linear`, `orthogonal_rotation`
- **Activations:** `linear`
- **Losses:** `mean_squared_error`, `huber`
- **Learning rates:** `constant`, `exponential_decay`
- **Models:** HexNet, MLP
- **d:** 2, 4, 6, 8
- **Seeds:** 5

### Slate B

- **Datasets:** `sine`, `elementwise_power`, `unit_sphere_projection`, `simplex_projection`
- **Activations:** `linear`, `leaky_relu`, `sigmoid`
- **Losses:** `mean_squared_error`, `log_cosh`, `huber`
- **Learning rates:** both
- Same d and seeds as Slate A where you want alignment

### Slate C

- Choose the best four datasets from A/B, then add noise modes and a sigma ladder.

If those three slates are done well, you already have material for a serious preliminary results section.

---

## E2E coverage (non-exhaustive)

The full matrices above are **documentation and experiment design**. [`e2e_test.sh`](../e2e_test.sh) runs a **small fixed set** of `hexnet train` commands so CI validates wiring, dataset registration, and run output — not the full Cartesian product.

| Family | `e2e_test.sh` | What runs |
|--------|----------------|------------|
| Smoke | Steps 2–3 (under `runs/e2etest-smoke/`) | Hex train + resume from same run dir; MLP train |
| A | Family A block | All listed linear-family dataset ids on HexNet with one hyperparam line; one or two MLP trains |
| B | Family B block | `elementwise_power`, `sine`, `affine` with a tiny activation/loss cross |
| C | Family C block | All four projection datasets |
| D | Family D block | `fixed_permutation`, `sort` |
| E | Family E block | `full_linear` with dataset noise |
| F | Family F block | `binary_vector_classification`, `multi_label_linear` |

Optional: set `E2E_EPOCHS` before invoking the script to shorten runs locally (default matches historical `100`).
