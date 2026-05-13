# Heartbeat

Project state and context: what was last asked, what.s active, what.s next.

## Last thing asked / in progress

- **Train run-template config**: `TrainRunTemplateConfig` in [`src/services/train_run_template/`](src/services/train_run_template/) owns `config.json` normalize/validate (no `RunService` import); `RunService` ingest delegates to it; Hex/MLP/dataset expose schema hooks; removed [`src/commands/run_config_template.py`](src/commands/run_config_template.py).

---

## Active development

- Dataset registry / glossary parity with CLI + [`docs/math/datasets.md`](docs/math/datasets.md). `BaseDataset` additive Gaussian noise is implemented; Streamlit explorer noise sliders still backlog.

---

## Future work

- **[#7](https://github.com/TennisGazelle/HexNets/issues/7)** — [TECH DEBT] Unit Tests, CICD, Linting, and Refactoring (`tech-debt`)
- **[#5](https://github.com/TennisGazelle/HexNets/issues/5)** — dev container [test] (`cli-refactor`)


Other distributions to consider for noise:
- Laplacian distribution
- Cauchy distribution
