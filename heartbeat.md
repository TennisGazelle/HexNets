# Heartbeat

Project state and context: what was last asked, what’s active, what’s next.

## Last thing asked / in progress

- **Benchmark docs + E2E**: Canonical benchmark families and matrix live in [`docs/math/benchmark-families.md`](docs/math/benchmark-families.md). [`e2e_test.sh`](e2e_test.sh) uses `runs/e2etest-smoke/` for smoke trains, `runs/e2etest-famA`–`famF/` for benchmark families (no activation×loss×lr grid); optional `E2E_EPOCHS` env var.

---

## Active development

- Dataset registry / glossary parity with CLI + [`docs/math/datasets.md`](docs/math/datasets.md). `BaseDataset` additive Gaussian noise is implemented; Streamlit explorer noise sliders still backlog.

---

## Future work

- **[#7](https://github.com/TennisGazelle/HexNets/issues/7)** — [TECH DEBT] Unit Tests, CICD, Linting, and Refactoring (`tech-debt`)
- **[#5](https://github.com/TennisGazelle/HexNets/issues/5)** — dev container [test] (`cli-refactor`)
