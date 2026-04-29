# Benchmark families and experiment matrix

**Purpose:** Readable tables for **which synthetic tasks to group**, **what to sweep** (without committing to a full Cartesian product), and **what insights to look for**. This page is the canonical reference for benchmark design and for how much of it [`e2e_test.sh`](../../e2e_test.sh) exercises in CI.

**Source of truth for implementations:** dataset ids and CLI flags come from [`src/data/dataset.py`](../../src/data/dataset.py) (`-t` / `--type`) and [`src/commands/command.py`](../../src/commands/command.py) (`add_training_arguments`, `add_global_arguments`).

**CLI tokens (copy-paste accurate):**

| Concept | Examples |
|--------|----------|
| Loss | `mean_squared_error`, `huber`, `log_cosh`, `quantile`, … |
| Activation | `linear`, `relu`, `leaky_relu`, `sigmoid` |
| Learning rate | `constant`, `exponential_decay` |
| Noise | `--dataset-noise` {`inputs`,`targets`,`both`}, `--dataset-noise-sigma`, `--dataset-noise-mu` |

**Principle:** do **not** start with every activation × every loss × every dataset × every dimension × every seed. Use **families** first; expand grids only where a family already shows signal.

**Registry extras:** `soft_threshold`, `sparse_identity` are registered for stress / failure studies but are optional in early slates.

---

## Family overview

| ID | Theme | Role |
|----|--------|------|
| **A** | Linear structure recovery | Cleanest calibration and paper claims (trivial vs structured linear maps). |
| **B** | Smooth nonlinear regression | Expressivity vs training stability; activation–task mismatch (CLI warnings). |
| **C** | Projection / constraints | Geometry-aware operators vs generic smooth maps. |
| **D** | Order / permutation / discontinuity | Relational rearrangement; exposes limits of smooth bias. |
| **E** | Noise robustness | Degradation curves vs single scores; corruption vs clean. |
| **F** | Classification-style vector outputs | Secondary evidence beyond scalar regression. |

---

## Family A — Linear structure recovery

| Datasets (`-t`) | Notes |
|-----------------|--------|
| `identity` | Isotropic trivial map. |
| `linear_scale` | Scaled identity (see `TrainCommand` default scale for `linear_scale`). |
| `diagonal_scale` | Per-dimension scaling. |
| `diagonal_linear` | Random diagonal linear map. |
| `full_linear` | Dense linear `y = x A^T`. |
| `low_rank_linear` | Low-rank linear structure. |
| `affine` | Affine `y = x A^T + b`. |
| `orthogonal_rotation` | Rotationally structured target. |

| Sweep (start tight) | Values |
|---------------------|--------|
| Models | `hex`, `mlp` (`-m`) |
| Activations | `linear`, then optionally `leaky_relu` |
| Losses | `mean_squared_error`, `huber` |
| Learning rates | `constant`, `exponential_decay` |
| Dimension | Ladder e.g. `-n` 2, 4, 6, 8 |
| Seeds | Several (e.g. 5+); `-s` |

| If you see… | Interpretation |
|-------------|----------------|
| Hex ≈ MLP on `identity`, `linear_scale` | **Calibration success:** hex geometry does not penalize trivial isotropic maps; use as baseline. |
| Hex ahead on `diagonal_scale`, `diagonal_linear`, `low_rank_linear` | Structured / anisotropic linear maps may interact with architecture. |
| Difference on `orthogonal_rotation` | Target is explicitly rotation-like; cleaner narrative than arbitrary nonlinear gaps. |

---

## Family B — Smooth nonlinear regression

| Datasets | Notes |
|----------|--------|
| `elementwise_power` | Smooth nonlinearity on bounded inputs. |
| `sine` | Periodic smooth structure. |
| `affine` (optional) | Bridge from linear family to nonlinear. |

| Sweep | Values |
|-------|--------|
| Activations | `linear`, `relu`, `leaky_relu`, `sigmoid` |
| Losses | `mean_squared_error`, `log_cosh`, `huber` |
| Learning rates | both schedules |
| Align with A | Same `-n` / seeds where you want comparability |

| If you see… | Interpretation |
|-------------|----------------|
| Similar mean, lower seed variance or fewer bad runs | **Stability** claim vs raw accuracy. |
| Large swings by activation | **Activation–task compatibility** may dominate architecture (aligns with CLI soft-warnings for bounded outputs). |

---

## Family C — Projection and constraint tasks

| Datasets | Operator flavor |
|----------|------------------|
| `unit_sphere_projection` | Spherical geometry |
| `l2_ball_projection` | ℓ₂ ball (radius from `scale` where applicable) |
| `non_negative_projection` | Non-negativity |
| `simplex_projection` | Simplex geometry |

| Sweep | Values |
|-------|--------|
| Activations | `linear`, `relu`, `leaky_relu`, `sigmoid` |
| Losses | `mean_squared_error`, `huber`, `log_cosh` (optionally defer `quantile` to shrink grid) |
| Learning rates | both |

| If you see… | Interpretation |
|-------------|----------------|
| Gains vs Family B on generic smooth tasks | Advantage may be **geometry-aware** rather than “more nonlinear.” |
| ReLU-like wins on `non_negative_projection` | Activation–constraint alignment. |

---

## Family D — Order / permutation / discontinuity

| Datasets | Notes |
|----------|--------|
| `fixed_permutation` | Fixed reorder of coordinates. |
| `sort` | Row sort; hard for pure smooth interpolation. |

| Sweep | Values |
|-------|--------|
| Activations | Start `linear`, `leaky_relu` |
| Losses | `mean_squared_error`, optionally `huber` |
| Learning rates | both |
| `-n` | Modest first |

| If you see… | Interpretation |
|-------------|----------------|
| Hex relatively better than on vanilla linear | Possible **relational / re-indexing** bias (or not — report either). |
| Both struggle on `sort` | Honest **limit** story: hex is not universal. |

---

## Family E — Noise robustness

| Typical datasets (subset of A–D) | Role |
|----------------------------------|------|
| `full_linear`, `orthogonal_rotation`, `sine`, `simplex_projection` | Strong candidates before adding noise |
| Optionally `sort`, `fixed_permutation` | Stress under corruption |

| Noise modes | CLI |
|-------------|-----|
| Inputs only | `--dataset-noise inputs` |
| Targets only | `--dataset-noise targets` |
| Both | `--dataset-noise both` |

| Sweep | Example ladder |
|-------|------------------|
| `sigma` | e.g. 0.0, 0.05, 0.1, 0.2, 0.4 (`--dataset-noise-sigma`) |
| Losses | `mean_squared_error`, `huber`, `log_cosh` |
| Activations | Best 1–2 from earlier phases |
| Seeds | Multiple |

| If you see… | Interpretation |
|-------------|----------------|
| Curves vs σ more informative than one score | **Robustness** narrative. |
| Architectures diverge only past a noise threshold | Effect of geometry under **corruption** vs clean interpolation. |

---

## Family F — Classification-style vector outputs

| Datasets | Notes |
|----------|--------|
| `binary_vector_classification` | Thresholded binary-like targets (often paired with `sigmoid`). |
| `multi_label_linear` | Linear decision surface → multi-label style. |

| Sweep | Values |
|-------|--------|
| Activations | `sigmoid`, `linear` |
| Losses | Whatever the CLI exposes; many losses remain regression-oriented. |
| Compare | Primarily convergence / separability behavior across `hex` vs `mlp`. |

**Positioning:** useful for a short “beyond pure regression” section; usually not the main paper thread until stable.

---

## Suggested iteration order (design matrix)

| Phase | Focus | Datasets (typical) | Activations | Losses | LR | Models | `-n` / seeds |
|-------|--------|-------------------|-------------|--------|-----|--------|----------------|
| **1** | Clean linear | `identity`, `linear_scale`, `diagonal_scale`, `full_linear`, `orthogonal_rotation` | `linear` | `mean_squared_error`, `huber` | both | hex, mlp | ladder; 5+ seeds |
| **2** | Smooth nonlinear | `elementwise_power`, `sine` | `linear`, `relu`, `leaky_relu`, `sigmoid` | `mean_squared_error`, `log_cosh`, `huber` | both | align with 1 | align with 1 |
| **3** | Geometric operators | `unit_sphere_projection`, `l2_ball_projection`, `simplex_projection`, `non_negative_projection` | `linear`, `relu`, `leaky_relu` | `mean_squared_error`, `huber`, `log_cosh` | both | hex, mlp | as above |
| **4** | Noise | Best 4 from 1–3 | best 1–2 from earlier | `mean_squared_error`, `huber`, maybe `log_cosh` | both | modes × σ ladder | multiple seeds |
| **5** | Failure / stress | `sort`, `fixed_permutation`; optional `low_rank_linear` @ higher d; `sparse_identity` | as needed | as needed | both | limits narrative | modest `-n` first |

---

## First comparison slates (compact)

### Slate A (linear baseline)

| Field | Value |
|-------|--------|
| Datasets | `identity`, `linear_scale`, `diagonal_scale`, `full_linear`, `orthogonal_rotation` |
| Activation | `linear` |
| Losses | `mean_squared_error`, `huber` |
| LR | `constant`, `exponential_decay` |
| Models | `hex`, `mlp` |
| `-n` | 2, 4, 6, 8 |
| Seeds | ≥ 5 |

### Slate B (mixed nonlinear + geometry)

| Field | Value |
|-------|--------|
| Datasets | `sine`, `elementwise_power`, `unit_sphere_projection`, `simplex_projection` |
| Activations | `linear`, `leaky_relu`, `sigmoid` |
| Losses | `mean_squared_error`, `log_cosh`, `huber` |
| LR | both |
| Align | Same `-n` / seeds as Slate A where useful |

### Slate C (noise on winners)

Pick the best four datasets from Slates A–B, then add noise modes and a σ ladder (see Family E).

---

## Cross-cutting insights to track

| # | Hypothesis | Best supported by |
|---|------------|-------------------|
| 1 | Hex is not universal but can help on structured geometric transforms | `orthogonal_rotation`, projection family, maybe `low_rank_linear` |
| 2 | On trivial linear maps, geometry does not hurt | `identity`, `linear_scale` |
| 3 | Activation–task compatibility rivals architecture | CLI warnings + Families B/C |
| 4 | Stability / robustness over single clean score | seed variance, Family E curves |
| 5 | Effects are **family-dependent** | Compare outcomes across A–D |

---

## E2E coverage (CI smoke, non-exhaustive)

[`e2e_test.sh`](../../e2e_test.sh) runs a **fixed small set** of trains to validate wiring and artifacts. It is **not** the full experiment matrix above. Optional: `E2E_EPOCHS=20` (or similar) for shorter local runs; default epoch count matches historical `100`.

| Block | Run directory prefix | What runs |
|-------|----------------------|-----------|
| Smoke | `runs/e2etest-smoke/` | Hex new + resume; MLP |
| A | `runs/e2etest-famA/` | All Family A dataset ids on hex with one hyperparam line; sample MLP |
| B | `runs/e2etest-famB/` | `elementwise_power`, `sine`, `affine` with a tiny activation/loss cross |
| C | `runs/e2etest-famC/` | All four projection datasets |
| D | `runs/e2etest-famD/` | `fixed_permutation`, `sort` |
| E | `runs/e2etest-famE/` | `full_linear` + dataset noise |
| F | `runs/e2etest-famF/` | `binary_vector_classification`, `multi_label_linear` |

---

## Paper-facing checklist (figures / claims)

| Artifact | Content |
|----------|---------|
| Baseline table | Hex vs MLP on clean linear family |
| Convergence | `identity`, `orthogonal_rotation`, one projection, `sine` |
| Noise curves | Performance vs σ for 2–3 datasets |
| Heatmap | Dataset × activation × model (loss fixed or faceted) |
| Failure panel | `sort` / `fixed_permutation` |

| Claim style | Example |
|-------------|---------|
| Restrained | Match on trivial linear maps; differences where structured; control activation/loss before attributing to architecture; corruption may separate models more than clean data. |

---

## What to defer

- Full grid on every knob at once.
- Leading with Family F as the main result before A–E are understood.
- Over-interpreting without curves (especially noise and convergence).

**First milestone:** one solid pass each through linear (A), nonlinear (B), projections (C), and noise (E) — enough for an initial results section.
