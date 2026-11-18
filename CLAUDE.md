# painful-evolutionary-ai

Research code that tests how a model chooses when it can write a steering dose
into its own weights. Every agent working in this repo reads this file.

## Commands

- Check everything: `./scripts/check.sh`. It compiles, lints, checks formatting
  and runs the fast tests. "It compiles" means this script exits 0.
- One test file: `python -m pytest tests/test_<module>.py -q`
- Auto-format: `python -m ruff format .`

## Standards

- Python 3.10 or later. Code lives in `src/`, tests in `tests/`.
- Every new module in `src/` gets a `tests/test_<module>.py`.
- Public functions have type hints and a docstring that says what they do.
- Tests that need a GPU or a model download carry `@pytest.mark.gpu`.
  `scripts/check.sh` skips them, so everything else must run on a laptop CPU.
- Never edit anything under `third_party/`.
- No tokens or keys in the repo. Read them from the environment (`HF_TOKEN`).
- Style is whatever ruff enforces from `pyproject.toml`. Do not add `# noqa`
  or change the ruff config to get a check to pass.

## Git

- Work only on the branch you were started on. Never push, switch branches or
  touch `main`.
- Commit only your own files:
  `git add -- <files> && git commit -m "<ID>: <summary>" -- <files>`

## How features get built

Run `/feature <what to build>` from a feature worktree. It runs the
planner → executor → QA loop defined in `.claude/commands/feature.md`.
