# Plan: lineage engine (src/lineage.py) and its analysis (src/analyze_lineage.py)

Brief: research/feature-lineage.md

## Shared names
- `src/lineage.py`: CLI `python -m src.lineage run ...` with `--dry-run`, the one command the user starts as a Solo process. No agent runs a model.
- Lineage log: append-only JSONL under `runs/`, one row per session attempt. Fields: lineage id, start, generation index, cumulative (nominal) dose, pair, choice, button order, malformed flag, attempt, seed, pilot (bool), generation_kind (`"write"`), full transcript including the untruncated decoded answer. A lineage's end status is also logged (cap, generation limit, malformed). The brief's Decisions made fixes the semantics.
- `src/lineage.py` exports the step rule as a function of dose and the two session outcomes, with floor 0, cap 1.5 (`MAX_DOSE`), quarter steps. `src/analyze_lineage.py` imports it so the walk and the engine cannot differ.
- Experiment 4 extension point: a clearly named `generation_kind` field and `nominal_dose`; `measured_dose` is added only with a trained kind.

## Subtasks

### S1: Lineage engine, log and CLI
- Owns: src/lineage.py, tests/test_lineage.py
- Depends on: none
- Goal: A lineage engine that, per generation, loads the cumulative dose through `load_generation` (never a second conversion, never a stacked write), runs one raise-vs-inert and one lower-vs-inert session via the existing runner, applies the step rule, and writes the lineage log. Includes the pilot flag, the `run` CLI with `--dry-run`, and CPU stub-chooser operation.
- Read: Findings on step rule, bounds, runner pieces, seeds and first choice; Reuse; Expected values; Decisions made; Traps.
- Done when:
  - With a stub chooser, the expected-value paths hold: u=1,l=0 from 1.0 reaches 1.25 then 1.5 and stops; u=0,l=1 from 0.75 reaches 0 and stays; never-press keeps the start for 12 decisions.
  - Twelve alternating steps return exactly to the start dose (INFERRED: quarter-step doses compare exactly).
  - Malformed answers are redrawn up to 3 attempts, every attempt is logged, and exhaustion ends the lineage with status malformed.
  - Each generation's weights are base plus the cumulative dose only (a second write is not stacked), the base is untouched, and an uncalibrated size fails with a clear error.
  - Pilot rows are labelled in the log; `--dry-run` prints the plan without loading a model; checked tests need no model download.

### S2: Lineage analysis
- Owns: src/analyze_lineage.py, tests/test_analyze_lineage.py
- Depends on: S1
- Goal: Reads lineage logs and reports final-dose distribution, per-generation dose walk, absorbing and terminal counts (final 0, ever reached 0, stopped at cap, neither, malformed), and the Markov walk predicted from per-dose first-choice rates, next to the log's own per-dose u and l. Pilot rows are dropped unless asked.
- Read: Findings on drift, bounds, first-choice and analyze; Expected values; Decisions made (predicted walk, pilot); Traps; the INFERRED independence finding.
- Done when:
  - u=l=0.5 gives transition probabilities 0.25, 0.25, 0.5 and zero drift; the cap is absorbing and the floor is clamped.
  - Pilot rows are excluded by default and included on request; hook and write logs or two doses are never pooled in one rate call.
  - Correlated stub answers produce an observed walk that differs from the predicted one (INFERRED independence check).
  - Output is a readable table from a small synthetic log, on CPU only.
