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
researcher -> planner -> executor -> QA loop.

- Claude Code: the loop is `.claude/commands/feature.md`, the agents are in
  `.claude/agents/`.
- Copilot CLI: start it with `./scripts/copilot.sh`. The loop is
  `.github/skills/feature/SKILL.md`, the agents are in `.github/agents/`.

The two sets of agent files carry the same instructions. Change both.

## Research record

- `docs/` holds the sources the researcher reads first: the MVP plan and a
  copy of the paper.
- `research/<branch>.md` is the sourced brief for a feature and
  `plans/<branch>.md` is its plan. Both are committed on the feature branch
  and merge with the code.
- `third_party/Pain-axis` is the authors' code at a pinned commit. Read it,
  never edit it.
