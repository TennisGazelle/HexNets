# Heartbeat

Project state and context: what was last asked, what’s active, what’s next.

## Last thing asked / in progress

- **MLP training curve save bugfix**: [`MLPNetwork.train_animated`](src/networks/MLPNetwork.py) saved the metrics PNG only when `epochs == 1` because the last-epoch check used `self.epochs_completed`, which increments every epoch. Save now triggers on `epoch == epoch_stop - 1` (same half-open range as the loop). Regression test in `tests/test_mlp_network_graph_weights.py`.

---

## Active development

- Dataset registry / glossary parity with [stories/004](stories/004-promote-dataset-registry-first-class-cli.md). [stories/006](stories/006-add-noisy-synthetic-regression-datasets.md): CLI + `BaseDataset` additive noise done; Streamlit explorer sliders still backlog.

---

## Future work

- **[#7](https://github.com/TennisGazelle/HexNets/issues/7)** — [TECH DEBT] Unit Tests, CICD, Linting, and Refactoring (`tech-debt`)
- **[#5](https://github.com/TennisGazelle/HexNets/issues/5)** — dev container [test] (`cli-refactor`)
