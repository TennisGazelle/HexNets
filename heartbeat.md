# Heartbeat

Project state and context: what was last asked, what’s active, what’s next.

## Last thing asked / in progress

- Added R² / adjusted R² toy verification in [`tests/test_metrics.py`](tests/test_metrics.py): reference helper mirroring [`src/networks/metrics.py`](src/networks/metrics.py), perfect fit, mean-baseline (R²≈0), worse-than-mean (negative R²), low-sample and `N = p + 2` boundary, constant-target degenerate cases.

---

## Active development

- Regression-metrics correctness for **[#8](https://github.com/TennisGazelle/HexNets/issues/8)** — R² / adjusted R² covered by toy-backed tests; remaining: Streamlit metrics explainer expander.

---

## Future work

- **[#7](https://github.com/TennisGazelle/HexNets/issues/7)** — [TECH DEBT] Unit Tests, CICD, Linting, and Refactoring (`tech-debt`)
- **[#5](https://github.com/TennisGazelle/HexNets/issues/5)** — dev container [test] (`cli-refactor`)
