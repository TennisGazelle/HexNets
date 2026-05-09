# Heartbeat

Project state and context: what was last asked, what.s active, what.s next.

## Last thing asked / in progress

- **LaTeX (`docs/latex/main.tex`)**: Fixed bibliography setup: `biblatex` now uses `\addbibresource{citations.bib}` + `\printbibliography` (not `\bibliography`); database moved to `citations.bib`; added missing `conclusion.tex`; example `\cite{kirkpatrick2017overcoming}` in `results.tex`.

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
