# Heartbeat

Project state and context: what was last asked, what.s active, what.s next.

## Last thing asked / in progress

- Fix unit tests after `run_name` / `--run-name` CLI alignment (`test_commands_train`, `test_run_config_template`; resume hint text).

---

## Active development

- Dataset registry / glossary parity with CLI + [`docs/math/datasets.md`](docs/math/datasets.md). `BaseDataset` additive Gaussian noise is implemented; Streamlit explorer noise sliders still backlog.
- Hex `model_metadata`: optional `epr` / `ro` — CLI `--epr` / `--ro`, `RunConfig` + `HexagonalNeuralNetwork.validate_run_metadata`; training loop wiring still backlog.

---

## Future work

- **[#7](https://github.com/TennisGazelle/HexNets/issues/7)** — [TECH DEBT] Unit Tests, CICD, Linting, and Refactoring (`tech-debt`)
- **[#5](https://github.com/TennisGazelle/HexNets/issues/5)** — dev container [test] (`cli-refactor`)


Other distributions to consider for noise:
- Laplacian distribution
- Cauchy distribution
