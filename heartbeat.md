# Heartbeat

Project state and context: what was last asked, what.s active, what.s next.

## Last thing asked / in progress

- **`docs/latex/` syntax + PDF build**: `main.tex` loads `amssymb` so `\mathbb{R}` compiles; small math-mode fixes in `implementation.tex` / `introduction.tex` (`\min`, `T \times T`, `\mathrm{layer}`, `\text{otherwise}`, `60^\circ`, `(i+3) \bmod 6`). `make pdf` produces [`docs/latex/main.pdf`](docs/latex/main.pdf) (Docker + `ghcr.io/xu-cheng/texlive-small`). Research Paper Streamlit page embeds that PDF only.

---

## Active development

- Dataset registry / glossary parity with CLI + [`docs/math/datasets.md`](docs/math/datasets.md). `BaseDataset` additive Gaussian noise is implemented; Streamlit explorer noise sliders still backlog.

---

## Future work

- **[#7](https://github.com/TennisGazelle/HexNets/issues/7)** — [TECH DEBT] Unit Tests, CICD, Linting, and Refactoring (`tech-debt`)
- **[#5](https://github.com/TennisGazelle/HexNets/issues/5)** — dev container [test] (`cli-refactor`)
