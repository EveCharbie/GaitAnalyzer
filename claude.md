# CLAUDE.md — Repository Preferences

## Context / Stack
- Primarily Python.
- Environments managed with **conda**.
- Dependency declarations in `pyproject.toml`; installs with `pip` or `conda-forge`.

## Python Style
- Uses f-strings with explicit format specs (e.g. `f"{n:04d}"` for zero-padded IDs).
- Prefers `@property` above `@abstractmethod` (correct decorator order) and favors
  `raise NotImplementedError` or `...` over bare `return` in abstract method bodies.
- Comfortable with `matplotlib`/`numpy` scientific plotting idioms (`fill_between`,
  `step` plots with explicit `where=`).

## Git / GitHub Workflow
- Uses branch protection rules on `main` (including blocking admin bypass) to enforce
  pull requests — no direct pushes to `main`.
- Uses Git submodules for coupling related repositories (e.g. linking a models repo
  into a main project) rather than always treating them as separate pip dependencies.
- Sets up GitHub Actions CI (e.g. `run_tests_linux.yml`) with status badges in the README.
- Restricts GitHub Pages deployment to the default branch only.
- Does not want to immediately commit the changes, prefers human review before commits.  

## Documentation
- Has used Jupyter Book for docs, deployed via GitHub Actions to GitHub Pages.
- Principle followed: commit source files only, not build artifacts (e.g. `docs/_build`
  excluded via `.gitignore` and/or a pre-commit hook).

## Testing
- Has built isolated test runners for example scripts (subprocess-based isolation with
  timeouts, especially relevant for GUI-producing code like `tkinter`/`matplotlib`).
- Prefers `pytest` for unit tests, with `pytest-cov` for coverage reporting.
- Prefers `numpy.testing` for array value comparisons in tests (e.g. `assert_allclose`, `assert_array_equal`).
- Prefers only functions in test files (no classes) unless a class is needed for `pytest` fixtures.
- Prefers avoiding as much as possible mock objects in tests, instead favoring real objects with controlled inputs.
- Prefers testing output numerical values as much as possible.

## Linting
- Uses black . -l120

## BioBuddy dependency management
- When working on a developer computer, use `export PYTHONPATH="${PYTHONPATH}:$(git rev-parse --show-toplevel)/../biobuddy"` to add the BioBuddy repo to the Python path for development purposes (when on GitHub CI, keep installing the released version).
