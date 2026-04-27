Assumptions:

* **Source of truth for implementations:** [`src/data/dataset.py`](../../src/data/dataset.py) — `BaseDataset`, `DATASET_FUNCTIONS`, `randomized_enumerate`, registry helpers, and a suffix-based import of sibling `*_dataset.py` modules (Arbor-style). Concrete classes live in those modules (e.g. `linear_scale_dataset.py`).
* **CLI training dataset switch:** `get_dataset()` in `src/commands/command.py` (`-t` / `--type`)

---

# Datasets

## Implemented classes (`src/data/*.py`)

| Class | `display_name` | Mapping |
|--------|----------------|---------|
| `IdentityDataset` | `identity` | `y = x` (scale 1) |
| `LinearScaleDataset` | `linear_scale` | `y = scale * x` on `[-1,1]^d` |
| `DiagonalScaleDataset` | `diagonal_scale` | Per-dimension scaled copy of `x` |

## Wired into `hexnet train` today

CLI `-t` / `--type` choices are **`list_registered_dataset_display_names()`** from [`src/data/dataset.py`](../../src/data/dataset.py) (same keys as `DATASET_FUNCTIONS`). `get_dataset()` in `src/commands/command.py` delegates to **`build_registered_dataset()`** in `dataset.py`.

* **`identity`** → `IdentityDataset` (persisted `dataset.scale` is typically `null`)
* **`linear_scale`** → `LinearScaleDataset` (default scale **2.0** in `TrainCommand` for parity with prior behavior)
* **`diagonal_scale`** → `DiagonalScaleDataset` (default scale **1.0** in `TrainCommand`; targets use per-dimension factors `(i+1) * scale`)

New runs persist a **`dataset` object** in `runs/.../config.json` (`id`, `num_samples`, `scale`); legacy runs only had flat `dataset_type` / `dataset_size` and are normalized on load.

Older saved `config.json` files may still show `"dataset_type": "linear"` from before this name was aligned.

## Adding a dataset

1. Add `src/data/<name>_dataset.py` (filename must end with `_dataset.py`). Subclass `BaseDataset` from `data.dataset` with a unique `display_name` (registration happens in `__init_subclass__`). Importing `data.dataset` loads all such modules automatically.
2. Give `__init__(self, d, num_samples, scale=...)` the same keyword shape as the other built-ins if it should work with **`build_registered_dataset`** without extra CLI wiring; CLI `-t` choices update automatically.
3. Implement **`get_glossary_node()`** on the class so the Streamlit glossary (and **`build_datasets_glossary_parent()`** in `dataset.py`) can list the dataset without hand-maintained mirror text. `GlossaryNode` lives in [`src/streamlit_app/glossary_types.py`](../../src/streamlit_app/glossary_types.py) (stdlib only; importing `data.dataset` does not load the Streamlit library).

For architecture and CLI patterns, see [`.cursor/ARCHITECTURE.md`](../../.cursor/ARCHITECTURE.md) and [`.cursor/CLI_PATTERNS.md`](../../.cursor/CLI_PATTERNS.md).
