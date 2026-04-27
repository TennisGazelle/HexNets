# Heartbeat

Project state and context: what was last asked, what’s active, what’s next.

## Last thing asked / in progress

- Implemented run persistence for paper traceability: manifest/config schema v1 (git SHA, seed, dataset block, trainable parameter count, optional note/tags), JSON ingest errors, and CLI `hexnet train --run-note` / `--run-tags`.

---

## Active development

- Regression-metrics correctness for **[#8](https://github.com/TennisGazelle/HexNets/issues/8)** — R² / adjusted R² covered by toy-backed tests; Streamlit metrics explainer and glossary shipped.

---

## Future work

- **[#7](https://github.com/TennisGazelle/HexNets/issues/7)** — [TECH DEBT] Unit Tests, CICD, Linting, and Refactoring (`tech-debt`)
- **[#5](https://github.com/TennisGazelle/HexNets/issues/5)** — dev container [test] (`cli-refactor`)
