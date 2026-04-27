# Heartbeat

Project state and context: what was last asked, what’s active, what’s next.

## Last thing asked / in progress

- Renamed regression **accuracy** to **`regression_score`** across `src/` (Metrics, figures, pickles/JSON keys, tables, Streamlit unpack), fixed figure axis semantics, added [`tests/test_metrics.py`](tests/test_metrics.py). Old runs/checkpoints using `accuracy` / `correct` are intentionally incompatible.

---

## Active development

- Regression-metrics correctness and interpretation pass for **[#8](https://github.com/TennisGazelle/HexNets/issues/8)** — rename + tests landed; remaining story items: deeper R² toy verification, Streamlit metrics expander.

---

## Future work

- **[#7](https://github.com/TennisGazelle/HexNets/issues/7)** — [TECH DEBT] Unit Tests, CICD, Linting, and Refactoring (`tech-debt`)
- **[#5](https://github.com/TennisGazelle/HexNets/issues/5)** — dev container [test] (`cli-refactor`)
