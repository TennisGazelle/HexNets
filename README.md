# HexNets

Hexagonal (and MLP baseline) neural network experiments with a small **CLI** (`hexnet`), **Streamlit** UI, and **run** / **figure** outputs.

[![ReleaseStaticBadge](https://img.shields.io/badge/Release%20Version-0.2.0-darkgreen?style=for-the-badge)](https://github.com/TennisGazelle/HexNets/releases/latest)

[![CI/CD](https://github.com/TennisGazelle/HexNets/actions/workflows/pr_cicd.yaml/badge.svg)](https://github.com/TennisGazelle/HexNets/actions/workflows/pr_cicd.yaml)


## Quick start

```bash
make install          # venv + editable install + dev deps
hexnet --help         # CLI (entry: pyproject → src/cli.py)
make run-streamlit    # or: streamlit run src/streamlit_app.py
```

**Reference images** (for Streamlit rotation tab): `hexnet ref --all` → `reference/*.png`

**Tests:** `make unit-test` · **E2E:** `make e2e-test`

## Nomenclature

- **Reference graph** — Structure for a fixed `(n, r)`; not tied to a training run.
- **Training graph** — Loss and regression metrics over epochs for one run (`Metrics` / `TrainingFigure`).
- **Run** — One training execution; artifacts under `runs/<name>/`.

## Where docs live

| Audience | Location |
|----------|----------|
| Short theory + math notes | [`docs/`](docs/README.md) (see index; **not** auto-editing `docs/latex/` without explicit ask) |
| Agent / architecture / CLI depth | [`.cursor/`](.cursor/README.md) |
| Backlog / issues | [`stories/`](stories/README.md) |
| Last request / state | [`heartbeat.md`](heartbeat.md) |

## CLI (summary)

| Command | Role |
|---------|------|
| `hexnet ref` | Reference graphs (`-g` types, `--all`, `-m hex\|mlp`) |
| `hexnet train` | Train hex or MLP; writes under `runs/` (optional `--run-note`, `--run-tags` for manifest traceability) |
| `hexnet adhoc` | Quick scripted demo |
| `hexnet stats <run_dir>` | Inspect a saved run |

Full argument patterns: [`.cursor/CLI_PATTERNS.md`](.cursor/CLI_PATTERNS.md).

## Layout

- `src/` — application code (`networks/`, `commands/`, `data/`, `services/`, `streamlit_app.py`, `hexnets_web/`)
- `figures/`, `runs/`, `reference/` — created at install or runtime (`Makefile` creates `figures/` and `runs/`)

Legacy / scratch: `hexnet.py` (root) is **not** the installed package entrypoint; prefer `hexnet` CLI after `make install`.
