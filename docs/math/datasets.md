Assumptions:

* **Source of truth for implementations:** `src/data/dataset.py`
* **CLI training dataset switch:** `get_dataset()` in `src/commands/command.py` (`-t` / `--type`)

---

# Datasets

## Implemented classes (`src/data/dataset.py`)

| Class | `display_name` | Mapping |
|--------|----------------|---------|
| `IdentityDataset` | `identity` | `y = x` (scale 1) |
| `LinearScaleDataset` | `linear_scale` | `y = scale * x` on `[-1,1]^d` |
| `DiagonalScaleDataset` | `diagonal_scale` | Per-dimension scaled copy of `x` |

## Wired into `hexnet train` today

`get_dataset(n, train_samples, type=...)` supports:

* **`identity`** → `IdentityDataset`
* **`linear_scale`** → `LinearScaleDataset` (CLI `-t` / `--type`; matches `display_name`. Default scale depends on command path, e.g. scale 2.0 in `train`.)

Older saved `config.json` files may still show `"dataset_type": "linear"` from before this name was aligned.

`DiagonalScaleDataset` exists in code but is **not** exposed on `-t` until `get_dataset` is extended.

## Adding a dataset

1. Subclass `BaseDataset` in `dataset.py` with a unique `display_name`.
2. Extend `get_dataset()` and `add_training_arguments` choices in `command.py` together so CLI help stays truthful.

For architecture and CLI patterns, see [`.cursor/ARCHITECTURE.md`](../../.cursor/ARCHITECTURE.md) and [`.cursor/CLI_PATTERNS.md`](../../.cursor/CLI_PATTERNS.md).
