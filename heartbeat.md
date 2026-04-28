# Heartbeat

Project state and context: what was last asked, what’s active, what’s next.

## Last thing asked / in progress

- Implemented **additive Gaussian dataset noise** on `BaseDataset` (`_load_data_impl` + noise kwargs), CLI flags, run `config.json` `dataset.noise`, and tests. Docs: [`docs/math/datasets.md`](docs/math/datasets.md).

---

## Active development

- Dataset registry / glossary parity with [stories/004](stories/004-promote-dataset-registry-first-class-cli.md). [stories/006](stories/006-add-noisy-synthetic-regression-datasets.md): CLI + `BaseDataset` additive noise done; Streamlit explorer sliders still backlog.

---

## Future work

- **[#7](https://github.com/TennisGazelle/HexNets/issues/7)** — [TECH DEBT] Unit Tests, CICD, Linting, and Refactoring (`tech-debt`)
- **[#5](https://github.com/TennisGazelle/HexNets/issues/5)** — dev container [test] (`cli-refactor`)
