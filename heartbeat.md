# Heartbeat

Project state and context: what was last asked, what.s active, what.s next.

## Last thing asked / in progress

- **Run config (code)**: [`RunConfig`](src/services/run_config/RunConfig.py) in [`src/services/run_config/`](src/services/run_config/) — validates on-disk ``config.json``, normalizes legacy dataset blocks, and merges ``hexnet train --run-config*`` with CLI overrides; [`RunService`](src/services/run_service/RunService.py) keeps **`run_config: RunConfig`** (and a **`config_contents`** property alias to the same dict) and calls **`RunConfig.from_ingested_dict`** on resume ingest.

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
