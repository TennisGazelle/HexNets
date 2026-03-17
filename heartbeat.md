# Heartbeat

Project state and context: what was last asked, what’s active, what’s next.

## Last thing asked / in progress

- Fixed **make unit-test** import error: `ModuleNotFoundError: No module named 'commands.reference_command'`. Cause: pytest prepends the test directory to `sys.path`, and `tests/conftest.py` was clearing `commands` from `sys.modules` before the path was correct. Fix: in `tests/commands/test_reference_command.py`, ensure `src` is first on `sys.path` and clear only the app’s `commands` (and `commands.*`) from `sys.modules` before importing (skip names containing `test_` so pytest’s test module name isn’t removed). Also corrected `tests/conftest.py` project_root (was `.parent.parent.parent`, should be `.parent.parent` for `tests/conftest.py`). Root `conftest.py` added to prepend `src` to path. Fixed the 5 failing tests in test_figure_service.py (patch targets, RefFigure.py .title() → axes set_title, incomplete-metrics test). All 52 unit tests now pass.

---

## Active development

- None at the moment (no open PRs; no task in progress beyond this setup).

---

## Future work

- **[#7](https://github.com/TennisGazelle/HexNets/issues/7)** — [TECH DEBT] Unit Tests, CICD, Linting, and Refactoring (`tech-debt`)
- **[#5](https://github.com/TennisGazelle/HexNets/issues/5)** — dev container [test] (`cli-refactor`)
