Assumptions:

* **Source of truth for implementations:** [`src/data/dataset.py`](../../src/data/dataset.py) — `BaseDataset`, `DATASET_FUNCTIONS`, `randomized_enumerate`, registry helpers, and a suffix-based import of sibling `*_dataset.py` modules (Arbor-style). Concrete classes live in those modules (e.g. `linear_scale_dataset.py`).
* **CLI training dataset switch:** `get_dataset()` in `src/commands/command.py` (`-t` / `--type`)

---

# Datasets

## Implemented classes (`src/data/*_dataset.py`)

| `display_name` | Class (module) | Short map |
|----------------|----------------|-----------|
| `identity` | `IdentityDataset` | `y = x` on `[-1,1]^d` (scale 1) |
| `linear_scale` | `LinearScaleDataset` | `y = scale * x` on `[-1,1]^d` |
| `diagonal_scale` | `DiagonalScaleDataset` | `y_i = scale * (i+1) * x_i` on `[-1,1]^d` |
| `diagonal_linear` | `DiagonalLinearDataset` | Random diagonal `a`; `y_i = a_i x_i` on `[-1,1]^d` |
| `full_linear` | `FullLinearDataset` | `y = x A^T`, Gaussian **x**, **A** scaled by CLI `scale` |
| `affine` | `AffineDataset` | `y = x A^T + b` |
| `orthogonal_rotation` | `OrthogonalRotationDataset` | `y = x Q^T`, **Q** orthogonal |
| `elementwise_power` | `ElementwisePowerDataset` | `y = sign(x) |x|^p` on `[-1,1]^d` |
| `sine` | `SineDataset` | `y = sin(ω x)` on `[-1,1]^d` |
| `soft_threshold` | `SoftThresholdDataset` | Soft-threshold (prox L1) on Gaussian **x** |
| `unit_sphere_projection` | `UnitSphereProjectionDataset` | Row-wise `y = x / ‖x‖₂` |
| `l2_ball_projection` | `L2BallProjectionDataset` | Project rows to ℓ₂ ball; CLI `scale` = radius **r** |
| `non_negative_projection` | `NonNegativeProjectionDataset` | `y = max(x, 0)` |
| `simplex_projection` | `SimplexProjectionDataset` | Euclidean projection onto probability simplex |
| `fixed_permutation` | `FixedPermutationDataset` | `y = x[:, perm]` for fixed **perm** |
| `sort` | `SortDataset` | Sort each row of **x** ascending |
| `binary_vector_classification` | `BinaryVectorClassificationDataset` | `y_i = 1[x_i > t]` as floats |
| `multi_label_linear` | `MultiLabelFromLinearDataset` | `y = 1[x A^T + b > 0]` |
| `sparse_identity` | `SparseIdentityDataset` | Sparse **x**, `y = x` |
| `low_rank_linear` | `LowRankLinearDataset` | `y = x (UV^T)^T` with rank ≤ `rank`; CLI `scale` scales **A** |

**Deferred (not implemented yet):** Gaussian / salt–pepper / mask denoise and bit-flip denoise (story 004 items 10, 11, 12, 21) — see [stories/006-add-noisy-synthetic-regression-datasets.md](../../stories/006-add-noisy-synthetic-regression-datasets.md).

## Glossary (`good_for` and `tags`)

Each dataset’s `get_glossary_node()` may set optional **`good_for`** (one-line experiment intent from the story) and **`tags`** (short labels such as `regression-compatible`, `classification-style`). The Streamlit **Glossary** tab shows them and includes them in search; types live in [`src/hexnets_web/glossary_types.py`](../../src/hexnets_web/glossary_types.py).

## Wired into `hexnet train` today

CLI `-t` / `--type` choices are **`list_registered_dataset_display_names()`** from [`src/data/dataset.py`](../../src/data/dataset.py) (same keys as `DATASET_FUNCTIONS`). `get_dataset()` in `src/commands/command.py` delegates to **`build_registered_dataset()`** in `dataset.py`.

* **`identity`** → `IdentityDataset` (persisted `dataset.scale` is typically `null`)
* **`linear_scale`** → `LinearScaleDataset` (default scale **2.0** in `TrainCommand` for parity with prior behavior)
* **`diagonal_scale`** → `DiagonalScaleDataset` (default scale **1.0** in `TrainCommand`; targets use per-dimension factors `(i+1) * scale`)
* Other registered ids use **`scale=1.0`** from the CLI unless you extend `TrainCommand` / manifest logic per type.

New runs persist a **`dataset` object** in `runs/.../config.json` (`id`, `num_samples`, `scale`); legacy runs only had flat `dataset_type` / `dataset_size` and are normalized on load.

Older saved `config.json` files may still show `"dataset_type": "linear"` from before this name was aligned.

## Adding a dataset

1. Add `src/data/<name>_dataset.py` (filename must end with `_dataset.py`). Subclass `BaseDataset` from `data.dataset` with a unique `display_name` (registration happens in `__init_subclass__`). Importing `data.dataset` loads all such modules automatically.
2. Give `__init__(self, d, num_samples, scale=...)` the same keyword shape as the other built-ins if it should work with **`build_registered_dataset`** without extra CLI wiring; CLI `-t` choices update automatically.
3. Implement **`get_glossary_node()`** on the class so the Streamlit glossary (and **`build_datasets_glossary_parent()`** in `dataset.py`) can list the dataset without hand-maintained mirror text. Optionally set **`good_for`** and **`tags`** on `GlossaryNode` for discoverability. `GlossaryNode` lives in [`src/hexnets_web/glossary_types.py`](../../src/hexnets_web/glossary_types.py) (stdlib only; importing `data.dataset` does not load the Streamlit library).

For architecture and CLI patterns, see [`.cursor/ARCHITECTURE.md`](../../.cursor/ARCHITECTURE.md) and [`.cursor/CLI_PATTERNS.md`](../../.cursor/CLI_PATTERNS.md).
