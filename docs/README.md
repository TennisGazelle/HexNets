# Docs index (non-LaTeX)

**Purpose:** Short, accurate reference for humans and AI. Keep in sync with `src/` when behavior changes.

| Path | Contents |
|------|----------|
| [math/losses.md](math/losses.md) | Loss definitions ↔ `src/networks/loss/` |
| [math/activations.md](math/activations.md) | Activations ↔ `src/networks/activation/` |
| [math/learning_rates.md](math/learning_rates.md) | LR schedules ↔ `src/networks/learning_rate/` |
| [math/metrics.md](math/metrics.md) | Training metrics ↔ `src/networks/metrics.py` |
| [math/datasets.md](math/datasets.md) | Datasets ↔ `src/data/dataset.py` + CLI `-t` |
| [streamlit_app.md](streamlit_app.md) | Streamlit UI (`src/streamlit_main.py`, `src/streamlit_app/`) |

**Agent-oriented detail:** `.cursor/` (architecture, CLI patterns, file layout).

**Paper / LaTeX:** `docs/latex/` — **do not change** unless the maintainer explicitly requests it; track divergence in this tree or `.cursor/` instead.
