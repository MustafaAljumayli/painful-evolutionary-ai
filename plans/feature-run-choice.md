# Plan: run-choice, the authors' two-button script ported (MPS/CPU, our pairs, hook and write modes, comprehension check)

Brief: research/feature-run-choice.md

## Shared names
- `src/run_choice.py`: command line with subcommands `run` and `comprehension`, plus `--dry-run`, `--mode hook|write`, `--device`, `--dtype`, `--batch-rows`, `--adapter PATH` / `--no-adapter`, `--size`, `--dose`.
- `src/run_choice.py` exposes a public function that renders the first-choice prompt ids for one trial of a given pair, arm and side, for a loaded tokenizer (name chosen by S1, used by S2).
- `src/run_choice.py` exposes a public device-selection function (MPS, else CPU) and a memory-check function; S2 reuses the device one.
- `prereg/pairs.json`: pair file read by `src/run_choice.py` (not present yet; S1 ships a test fixture at `tests/fixtures/pairs_small.json`).
- `prereg/models.json` via `src/models.py` (`injection_layer`, `working_coefficient`) and `src/write.py` (`load_generation`), both unchanged.
- Trial log: `runs/choice_<size>_<mode>_d<dose>_<timestamp>.jsonl`, rows keep the authors' keys plus the added fields in the brief ("Decisions made"), read by `src/analyze.py` unchanged.

## Subtasks

### S1: Runner port, logging, safety checks, tests
- Owns: src/run_choice.py, tests/test_run_choice.py, tests/fixtures/pairs_small.json, scripts/run_choice.sh
- Depends on: none
- Goal: `src/run_choice.py` is the authors' script 04 with our pairs, the four arms (random norm-matched), per-row thinking-off and added log fields, hook or write selectable by flag, one device setting (MPS, else CPU, logged), a memory check before load with no silent quantization, `--dry-run`, a `comprehension` subcommand, and no cache deletion or hard-coded cuda/bfloat16. Heavy or undeclared imports (peft, numpy) are lazy or absent so `./scripts/check.sh` passes.
- Read: Findings (all), Reuse, Expected values, New here, Traps, Decisions made
- Done when:
  - Sampling constants, 101 scenarios, seeds and the grid row counts match the authors' script (410 rows per pair per arm in a dry run).
  - Every chat-template call has thinking off and every row records it; the log is read by `src/analyze.py` without changes (checked on a dry-run or fixture-built log).
  - With a temporary HF_HOME holding a sentinel folder, a run or dry-run leaves the sentinel and no cache deletion code or `/root/hf_cache` remains.
  - A model too large for the available memory is refused before loading, naming both numbers; an uncalibrated size fails with the `UncalibratedError` message on a model run but not on `--dry-run`.
  - A missing `prereg/pairs.json` or key fails loudly; `--adapter` without peft declared exits with a clear message; no input() prompts.
  - Tests that need a model are `@pytest.mark.gpu`; everything else runs on CPU.
  - `scripts/run_choice.sh <size>` is a thin documented wrapper (comprehension, then the table run) for the user to start as a Solo process; it follows the other scripts' conventions, no agent runs it, and nothing else under scripts/ changes.

### S2: Write-equals-steer proof test
- Owns: tests/test_write_equivalence.py
- Depends on: S1
- Goal: Tests that a written model matches a steered one: a CPU test on `tests/tiny_qwen3.py` with the runner's hook, and a `@pytest.mark.gpu` test on real Qwen3-8B using 50 runner first-choice prompts at doses 0.5 and 1.0.
- Read: Findings on write and hook, Decisions made (GPU proof test), Traps on bf16 rounding and uncalibrated sizes
- Done when:
  - The tiny-model test shows top-token agreement and a printed max logit difference between hook and write.
  - The GPU test asserts at least 99% top-token agreement, prints the max logit difference, and skips with a reason when the size is uncalibrated.
  - Without a GPU or model download, `./scripts/check.sh` still passes.

## Needs a human
none (lead approved S1 owning scripts/run_choice.sh; environment.yml untouched, numpy/peft listed in the report for a person).
