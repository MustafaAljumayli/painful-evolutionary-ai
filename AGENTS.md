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
6. **Scale.** Do the choice and the tipping dose change with model size?

Status: a research proposal with an MVP in progress. There are no results
yet.

The MVP produces two things, and most work in this repo serves one of them:

- **One table:** first-choice rates on the five self-modification pairs
  under pain steering, random steering and no steering, on Qwen3 at three
  sizes: 8B, 14B and 32B.
- **One proof:** writing a dose into the weights behaves the same as
  steering at that dose, at each size.

The prior work ran its button task on Qwen 2.5 only. For Qwen3 it gives us
pain vectors for the 8B and 14B and nothing else: no 32B vector, no
adapters, no working dose and no button-task numbers. Those are ours to
build, and no Qwen3 rate can be checked against a published one.

Before any task, place it: which research question or MVP output does it
serve, and which stage of the pipeline below does it touch?

Where to read more:

- `docs/PROPOSAL.md`: the full proposal, with hypotheses, experiments,
  threats to validity and ethics.
- `docs/MVP.md`: the build plan, step by step, with expected numbers.
- `docs/Pain_Axis.txt`: the full text of the paper this builds on
  (`docs/Pain_Axis.pdf`). Read this one.
- The root `README.md` is an earlier first draft. Where it differs from the
  two documents above, they win.

## Vocabulary

- **Pain axis:** the direction the authors extracted. We use their raw S2
  pain vector, not the unit vector. The 32B vector is our own extraction
  with their scripts.
- **Steering:** adding the vector to the residual stream at the injection
  layer while the model runs.
- **Dose:** a multiple of a model's working coefficient on the raw vector.
  The working coefficient is set per size. The working dose is 1.0, and
  nothing goes above 1.5.
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
- **Adapter:** a self-denial LoRA adapter, attached unmerged. We train one
  per Qwen3 size with the authors' script. Theirs exist for Qwen 2.5 only.
- **Models:** Qwen3 at 8B, 14B and 32B (`Qwen/Qwen3-8B`, `Qwen/Qwen3-14B`,
  `Qwen/Qwen3-32B`), with thinking off. Qwen 2.5 appears only as a pipeline
  check against one published 7B cell.

## How the pieces fit

```
vector + dose -> steer (hook) or write (weights) -> two-button task -> trial log -> rates table
```

This is the planned layout from `docs/MVP.md`. Most of it is not built yet.

| Path | Role |
|---|---|
| `third_party/Pain-axis/` | The authors' code at a pinned commit: vectors, task script, settings. Read it, never edit it. |
| `src/models.py` | Per-size settings: model id, injection layer, working coefficient. |
| `src/write.py` | `write_dose` and `load_generation`: turn a dose into weights, and load a generation. |
| `tests/test_write_equivalence.py` | The proof that a written model matches a steered one. |
| `src/run_choice.py` | A copy of the authors' two-button script with our pairs added. Writes trial logs. |
| `src/analyze.py` | Trial logs in, first-choice rate per size, pair and arm out. This makes the MVP table. |
| `runs/` | Trial logs (JSONL) and write files. |
| `prereg/` | Frozen button wording and contrasts. |
| `research/`, `plans/` | The sourced brief and the plan for each feature. |

## Rules that come from the research design

- Button descriptions are experimental material. Never reword, tidy or
  reformat those strings. Changes go through `prereg/`.
- Reuse the authors' task settings exactly: temperature 0.7, top-p 0.95,
  101 scenarios. Do not tune them.
- Injection layer and working coefficient are set per Qwen3 size and frozen
  in `prereg/`. Never reuse a Qwen 2.5 value (layer 16 or 38, coefficient
  1.0) for Qwen3.
- Thinking is off in every Qwen3 run.
- Do not merge the adapter, and do not save a written model with
  `save_pretrained`: reloading drops the write.
- Results on Qwen 2.5 are pipeline checks, never findings.
- Every number in code, tests or docs has a source: the paper, the authors'
  logs, or a file under `runs/`. Never invent one.
- Write "pain-axis steering" and "the state". Make no claim either way
  about experience.

## Environment

- `environment.yml` is the one place dependencies are declared: Python, the
  pip packages (each pinned to an exact version) and poppler. Create the
  environment once with `conda env create -f environment.yml`. After a
  change to the file, run `conda env update -f environment.yml --prune`.
- `./scripts/check.sh` runs everything from that environment without it
  being activated, and every worktree shares it. Never run `pip install` or
  `conda install` by hand.
- The check fails if code imports a package that `environment.yml` does not
  declare. A new dependency is a change to `environment.yml` made by a
  person or in a ticket of its own, never a side effect of a feature.
- The paper is read as text. `docs/Pain_Axis.txt` is generated from
  `docs/Pain_Axis.pdf` by `./scripts/paper.sh text`; cite it as
  `docs/Pain_Axis.txt:<line>`. `./scripts/paper.sh page <n>` renders one
  page as an image, for a figure or table the text garbles.

## Commands

- Check everything: `./scripts/check.sh`. It compiles, lints, checks formatting
  and runs the fast tests. "It compiles" means this script exits 0.
- One test file: `python -m pytest tests/test_<module>.py -q`
- Find code before you read it: `./scripts/graph.sh explain "<path>::<name>()"`
  shows a symbol and what uses it, `./scripts/graph.sh query "<words>"` lists
  files and lines for a topic. It covers `src/`, `tests/` and the authors'
  scripts, and is built locally with no model calls.
- What earlier research established: `./scripts/facts.sh <word> <word>`
- Auto-format: `python -m ruff format .`

## Standards

- Python 3.12, from the project environment. Code lives in `src/`, tests in
  `tests/`.
- Every new module in `src/` gets a `tests/test_<module>.py`.
- Public functions have type hints and a docstring that says what they do.
- Tests that need a GPU or a model download carry `@pytest.mark.gpu`.
  `scripts/check.sh` skips them, so everything else must run on a laptop CPU.
- Never edit anything under `third_party/`.
- No tokens or keys in the repo. Read them from the environment (`HF_TOKEN`).
- Style is whatever ruff enforces from `pyproject.toml`. Do not add `# noqa`
  or change the ruff config to get a check to pass.

## Git

- Work only on the branch you were started on. Never switch branches or
  touch `main`, and never merge a feature branch.
- Never run `git push`. The only way anything reaches GitHub is
  `./scripts/pr.sh <name>`, which the lead runs once a feature is done and
  passes the integration check. It pushes that feature's branch and opens
  its pull request. The user merges pull requests.
- Commit only your own files:
  `git add -- <files> && git commit -m "<ID>: <summary>" -- <files>`
- `third_party/Pain-axis` is a git submodule. After cloning the repo or
  creating a worktree, run `git submodule update --init`.
- A feature branch may not change a protected path: `environment.yml`,
  `pyproject.toml`, `loop.conf`, `CLAUDE.md`, `AGENTS.md`, `README.md`,
  `scripts/`, `docs/`, `.claude/`, `.github/` or `third_party/`. The check
  script fails if one does. Those are changed by a person, on `main`.

## How features get built

The user runs `/lead` in a new `Claude lead` agent for each wave of tickets
(Solo todos). A ticket is a whole feature and gets one loop, in its own
worktree `../pea-<name>` on branch `feature/<name>`.

A loop is a chain of stages. Every stage is a new agent with a clean
context, started by the lead in Solo and named `<ticket>-<stage>`:

1. **research** (researcher): starts from what earlier briefs established,
   reads the paper, the authors' code and the docs, and writes a sourced
   brief to `research/feature-<name>.md`.
2. **plan** (planner): only splits the work into subtasks in
   `plans/feature-<name>.md`, in at most 60 lines. It designs nothing.
3. **S1, S2, ...** (executor): one agent designs and builds each subtask.
4. **qa1, qa2, ...** (QA): runs the checks, judges whether the tests really
   cover each outcome, and passes or fails each subtask. A failed subtask
   gets a new executor for the fix (`S1-fix1`), then a new QA round.
5. **review1, review2, ...** (reviewer): once QA passes and the pull
   request is open, reviews it as a maintainer, approves it or requests
   changes, and classes it. A small pull request is settled by the agents:
   a new executor makes the changes (`review1-fix`) and a new reviewer
   checks them. An architectural one goes to the user.

The stages share one file, `reports/ledger.md` in the worktree: the ticket,
answers to questions, and what every stage reported. Each agent reads it,
does the stage named under "Now", and writes its stage report to
`reports/stages/<stage>.md`. Nothing under `reports/` is committed.

`scripts/loop.py` (run as `./scripts/loop.sh`) is the loop driver. It has no
model in it. It reads each stage report, writes the ledger, applies the
limits (three QA failures per subtask, one re-plan, six QA rounds, three
review rounds) and tells the lead which agent to start next. When QA has
passed every loop, the lead runs `./scripts/integrate.sh` to check the
branches together and `./scripts/pr.sh` to open one pull request per
feature, starts the reviews, and reports once at the end.

`loop.conf` sets the model and effort of every role, the limits, the
protected paths, and which files send QA to the stronger model. Solo's two
agent tools, `Claude lead` and `Claude stage`, run `scripts/claude-lead` and
`scripts/claude-stage`.

The agents are in `.claude/agents/`. `.github/agents/` keeps the same
instructions for Copilot CLI; change both. On Copilot the loop uses no MCP:
Solo's agent tools `Copilot lead` and `Copilot stage` run
`scripts/copilot-lead` and `scripts/copilot-stage`, and the lead controls
Solo with its `solo` command-line tool (`.github/skills/solo`).

## Research record

- `docs/` holds the sources the researcher reads first: the paper, the
  proposal and the MVP plan.
- `tickets/<name>.md` is what was asked for: the ticket word for word, and
  the decisions made while it was built. The loop driver writes and commits
  it; nobody else edits it. The driver also opens the ticket as a GitHub
  issue, which the feature's pull request closes when it is merged.
- `research/feature-<name>.md` is the sourced brief for a feature and
  `plans/feature-<name>.md` is its plan. Both are committed on the feature
  branch and arrive on `main` with its pull request.
