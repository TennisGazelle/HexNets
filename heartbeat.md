# Heartbeat

Project state and context: what was last asked, what.s active, what.s next.

## Last thing asked / in progress

- **DynamicWeightsFigure highlight outlines**: optional ``highlight_channel`` (hex passes ``self.r``) uses the same ``Figure.colors`` tab10 slot as ``TrainingFigure`` lines for that rotation; all outline strokes share that color. Omitted channel + mask defaults to black.
- **Live weights in ``FigureService.figures``**: stable shape key ``weights_live:…``; layout change closes other live-weight entries.

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
