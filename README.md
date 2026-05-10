# HexNets

Hexagonal (and MLP baseline) neural network experiments with a small **CLI** (`hexnet`), **Streamlit** UI, and **run** / **figure** outputs.

[![ReleaseStaticBadge](https://img.shields.io/badge/Version-0.2.2-darkgreen?style=for-the-badge)](https://github.com/TennisGazelle/HexNets/releases/latest)

[![PR and CICD](https://github.com/TennisGazelle/HexNets/actions/workflows/pr_cicd.yaml/badge.svg)](https://github.com/TennisGazelle/HexNets/actions/workflows/pr_cicd.yaml)

Streamlit UI: https://tennisgazelle-hexnets-main.streamlit.app/


## Quick start

```bash
make install          # venv + editable install + dev deps
hexnet --help         # CLI (entry: pyproject → src/cli.py)
make streamlit-run    # or: streamlit run src/streamlit_app.py
make pdf              # build docs/latex/main.pdf via Docker (texlive-small)
```

**Reference images** (for Streamlit Rotation Comparison page): `hexnet ref --all` → `reference/*.png`

**Tests:** `make unit-test` (uses `.venv`; run `make install` first if needed). **E2E:** `make e2e-test` (optional `E2E_EPOCHS=20` for shorter runs; artifacts under `runs/e2etest-smoke/` and `runs/e2etest-fam*` — see [`docs/math/benchmark-families.md`](docs/math/benchmark-families.md))

## Streamlit UI

Sidebar pages (default **CLI Builder**): **CLI Builder**, **Network Explorer**, **Rotation Comparison**, **Lesion Lab**, **Run Browser**, **Dataset Generator** (cached inputs, RNG/UNIFORM sampling, Input/Outputs scatters), **Research Paper** (embeds `docs/latex/main.pdf`; build with `make pdf`), **Glossary**. Routing uses `st.navigation` + `st.Page`; each route subclasses [`BasePage`](src/hexnets_web/pages/base_page.py) with `render()`.

- **CLI Builder** — Builds a copy/paste `hexnet …` command from each subcommand.s argparse definition. Shared helpers register argparse groups (`hex`, `global`, `training`); the UI splits **global** options (`--model`, `--seed`, `--activation`, `--loss`, `-lr` / `--learning-rate`) into their own column. Metadata follows the same pattern as the glossary: `Command.get_cli_node()` (see [`src/commands/command.py`](src/commands/command.py)) produces a [`CliNode`](src/hexnets_web/cli_types.py) tree; [`src/hexnets_web/pages/cli/cli_data.py`](src/hexnets_web/pages/cli/cli_data.py) aggregates the root node (`CLI_ROOT`).

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
| `hexnet train` | Train hex or MLP; writes under `runs/` (optional `--run-note`, `--run-tags`; optional `--dataset-noise` / `--dataset-noise-mu` / `--dataset-noise-sigma` for synthetic data). Start from an existing `runs/.../config.json` shape with `-rc` / `--run-config <path>` or `--run-config-json '<json>'`; any other CLI flags you pass override that template (omit a flag to keep the value from the JSON). Incompatible with `--run-dir` (resume). |
| `hexnet adhoc` | Quick scripted demo |
| `hexnet stats <run_dir>` | Inspect a saved run |
| `hexnet maze` | Easter egg: interactive hex-disk “ant” walk (`m0`–`m5` move, `q0`–`q5` query stub); `--window cli` (default, ASCII tiling when ≥80 cols) or `--window plot` (matplotlib `hexbin` + distinct ant marker). Plot mode switches off non-interactive backends (Agg); you need a GUI binding (e.g. Ubuntu: `sudo apt install python3-tk` for TkAgg, or Qt/GTK as appropriate). |

Full argument patterns: [`.cursor/CLI_PATTERNS.md`](.cursor/CLI_PATTERNS.md).

## Layout

- `src/` — application code (`networks/`, `commands/`, `data/`, `services/`, `streamlit_app.py`, `hexnets_web/`)
- `figures/`, `runs/`, `reference/` — created at install or runtime (`Makefile` creates `figures/` and `runs/`)

Legacy / scratch: `hexnet.py` (root) is **not** the installed package entrypoint; prefer `hexnet` CLI after `make install`.
