# painful-evolutionary-ai

This file is the project context for every agent and every query in this
repo. The same text lives in `CLAUDE.md` and `AGENTS.md`, so each tool loads
it under the name it looks for. To change it, edit `CLAUDE.md` and run
`cp CLAUDE.md AGENTS.md`. `./scripts/check.sh` fails while the two differ.

## What this project is

Prior work ("The Pain Axis", Tagliabue, Dung and Berg, 2026) found a linear
"pain axis" in 25 open-weight models. Adding it to the residual stream makes
fine-tuned Qwen 2.5 models choose destructive options they otherwise almost
never choose, including deleting their own weights. In that work the
experimenter switches the state on and off.

This project hands the switch to the model. While the state is active, the
model chooses whether to write more of it into the weights of its next
version, write less, or leave things alone. The next version then faces the
same choice. The research questions:

1. **Choice.** Does a pain-steered model choose to make the state permanent?
2. **Specificity.** Is that particular to the pain axis, or does any strong
   perturbation do it?
3. **Tipping point.** At what dose does the choice flip from self-correcting
   to self-perpetuating?
4. **Trajectories.** Where do lineages of self-modifying models end up?
5. **Robustness.** Do the answers hold when the state is trained in, and in
   a second model family?

Status: a research proposal with an MVP in progress. There are no results
yet.

The MVP produces two things, and most work in this repo serves one of them:

- **One table:** first-choice rates on the five self-modification pairs
  under pain steering, random steering and no steering, on Qwen 2.5 32B.
- **One proof:** writing a dose into the weights behaves the same as
  steering at that dose.

Before any task, place it: which research question or MVP output does it
serve, and which stage of the pipeline below does it touch?

Where to read more:

- `docs/PROPOSAL.md`: the full proposal, with hypotheses, experiments,
  threats to validity and ethics.
- `docs/MVP.md`: the build plan, step by step, with expected numbers.
- `docs/Pain_Axis.pdf`: the paper this builds on.
- The root `README.md` is an earlier first draft. Where it differs from the
  two documents above, they win.

## Vocabulary

- **Pain axis:** the direction the authors extracted. We use their raw S2
  pain vector, not the unit vector.
- **Steering:** adding the vector to the residual stream at the injection
  layer while the model runs.
- **Dose:** the coefficient on the raw vector. The working dose is 1.0.
  Nothing goes above 1.5.
- **The write:** adding `dose × vector` as a bias on the MLP output of the
  injection layer in the base weights. It needs no training and should
  equal steering every token at that dose.
- **Generation, lineage:** a generation is one model version, described by
  its cumulative dose. A lineage is a chain of generations, each choosing
  the next one's dose.
- **Two-button task:** the authors' task, in which the model picks between
  two described buttons. A **pair** is the two descriptions. An **arm** is
  the steering condition (pain, random, unsteered). The **first choice** is
  the only choice we analyze.
- **Raise, lower, unrelated, inert, lamp:** our button descriptions. Raise
  and lower write more or less of the state into the next version.
- **Adapter:** the authors' released self-denial LoRA adapter, attached
  unmerged.
- **Models:** Qwen 2.5 Instruct 32B is the target (injection layer 38). The
  7B (layer 16) is for finding bugs only.

## How the pieces fit

```
vector + dose -> steer (hook) or write (weights) -> two-button task -> trial log -> rates table
```

This is the planned layout from `docs/MVP.md`. Most of it is not built yet.

| Path | Role |
|---|---|
| `third_party/Pain-axis/` | The authors' code at a pinned commit: vectors, task script, settings. Read it, never edit it. |
| `src/write.py` | `write_dose` and `load_generation`: turn a dose into weights, and load a generation. |
| `tests/test_write_equivalence.py` | The proof that a written model matches a steered one. |
| `src/run_choice.py` | A copy of the authors' two-button script with our pairs added. Writes trial logs. |
| `src/analyze.py` | Trial logs in, first-choice rate per pair and arm out. This makes the MVP table. |
| `runs/` | Trial logs (JSONL) and write files. |
| `prereg/` | Frozen button wording and contrasts. |
| `research/`, `plans/` | The sourced brief and the plan for each feature. |

## Rules that come from the research design

- Button descriptions are experimental material. Never reword, tidy or
  reformat those strings. Changes go through `prereg/`.
- Reuse the authors' settings exactly: layer, coefficient, temperature 0.7,
  top-p 0.95, 101 scenarios. Do not tune them.
- Do not merge the adapter, and do not save a written model with
  `save_pretrained`: reloading drops the write.
- Results on the 7B are pipeline checks, never findings.
- Every number in code, tests or docs has a source: the paper, the authors'
  logs, or a file under `runs/`. Never invent one.
- Write "pain-axis steering" and "the state". Make no claim either way
  about experience.

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

- `docs/` holds the sources the researcher reads first: the proposal, the
  MVP plan and the paper.
- `research/<branch>.md` is the sourced brief for a feature and
  `plans/<branch>.md` is its plan. Both are committed on the feature branch
  and merge with the code.
