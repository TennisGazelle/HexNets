# Heartbeat

Project state and context: what was last asked, what’s active, what’s next.

## Last thing asked / in progress

- **Benchmark story + E2E**: [`stories/benchmark-families-to-test.md`](stories/benchmark-families-to-test.md) reformatted (headings, CLI-aligned tokens, E2E coverage table). [`e2e_test.sh`](e2e_test.sh) uses `runs/e2etest-smoke/` for smoke trains, `runs/e2etest-famA`–`famF/` for benchmark families (no activation×loss×lr grid); optional `E2E_EPOCHS` env var.

---

## Active development

- Dataset registry / glossary parity with [stories/004](stories/004-promote-dataset-registry-first-class-cli.md). [stories/006](stories/006-add-noisy-synthetic-regression-datasets.md): CLI + `BaseDataset` additive noise done; Streamlit explorer sliders still backlog.

---

## Future work

- **[#7](https://github.com/TennisGazelle/HexNets/issues/7)** — [TECH DEBT] Unit Tests, CICD, Linting, and Refactoring (`tech-debt`)
- **[#5](https://github.com/TennisGazelle/HexNets/issues/5)** — dev container [test] (`cli-refactor`)
