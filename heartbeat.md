# Heartbeat

Project state and context: what was last asked, what’s active, what’s next.

## Last thing asked / in progress

- **CLI Builder — “What your choices mean”**: registry-backed glossary (activation, loss, learning rate, dataset `type`) from current subcommand + widgets in [`src/hexnets_web/cli_builder.py`](src/hexnets_web/cli_builder.py); sections omitted when the arg is absent or explicitly **— omit —** (`default is None` and coerced `None`). Layout/groups still live in [`src/commands/command.py`](src/commands/command.py) and [`src/hexnets_web/cli_types.py`](src/hexnets_web/cli_types.py).

---

## Active development

- Dataset registry / glossary parity with [stories/004](stories/004-promote-dataset-registry-first-class-cli.md). [stories/006](stories/006-add-noisy-synthetic-regression-datasets.md): CLI + `BaseDataset` additive noise done; Streamlit explorer sliders still backlog.

---

## Future work

- **[#7](https://github.com/TennisGazelle/HexNets/issues/7)** — [TECH DEBT] Unit Tests, CICD, Linting, and Refactoring (`tech-debt`)
- **[#5](https://github.com/TennisGazelle/HexNets/issues/5)** — dev container [test] (`cli-refactor`)
