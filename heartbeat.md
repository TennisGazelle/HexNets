# Heartbeat

Project state and context: what was last asked, what.s active, what.s next.

## Last thing asked / in progress

- **`ReferenceCommand.invoke` + tests**: Resolve LR with `getattr(args, "learning_rate", "constant")` then `get_learning_rate(..., learning_rate=0.01)` for MLP/hex paths; five `Namespace` fixtures in `tests/commands/test_reference_command.py` include `learning_rate="constant"`. No `get_learning_rate_function` rename in repo (plan branch N/A).

---

## Active development

- Dataset registry / glossary parity with CLI + [`docs/math/datasets.md`](docs/math/datasets.md). `BaseDataset` additive Gaussian noise is implemented; Streamlit explorer noise sliders still backlog.

---

## Future work

- **[#7](https://github.com/TennisGazelle/HexNets/issues/7)** — [TECH DEBT] Unit Tests, CICD, Linting, and Refactoring (`tech-debt`)
- **[#5](https://github.com/TennisGazelle/HexNets/issues/5)** — dev container [test] (`cli-refactor`)
