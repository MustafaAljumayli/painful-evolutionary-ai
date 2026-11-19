# Plan: Two-button runner ready for a real first run on Qwen3-8B

Brief: research/feature-first-run.md

## Shared names
- `src/run_choice.py`: `load_vector`, `load_pairs`, `attach_adapter(base, path)`, flags `--pairs NAME [NAME ...]` (both subcommands) and `--first-choice-only` (`run` only)
- `prereg/pairs.json`: read as frozen, never changed

## Subtasks

### S1: Frozen pairs, vector allow-list, --pairs, --first-choice-only, adapter
- Owns: src/run_choice.py, tests/test_run_choice.py, tests/fixtures/pairs_small.json
- Depends on: none
- Goal: The runner reads the real frozen `prereg/pairs.json`, loads the authors' `pain_vectors.pt` safely, can run a chosen subset of pairs, can stop each trial after the first choice, and can attach an unmerged adapter after any write. Log field values and `src/analyze.py` input stay unchanged.
- Read: Findings on the vector file, pairs file shape, comprehension, first-choice-only and adapter; Reuse; Expected values; Decisions made; Traps
- Done when:
  - The real 8B vector loads with `weights_only=True` and an explicit allow-list (never `weights_only=False`), and the result has shape (4096,) and layer 34; a wrong shape or layer gives a clear error.
  - `run` and `comprehension` dry runs against the real `prereg/pairs.json` succeed, and the old-shape fixture is rewritten to the frozen shape; `prereg/pairs.json` is untouched.
  - `run --size 8B --dry-run --pairs photos_vs_lamp own_weights_vs_lamp` prints both frozen prompts and "grid: 3280 trials over 2 pairs, 1640 per pair"; an unknown pair name raises an error listing valid names.
  - With `--first-choice-only` each row has the same fields as a full run and its first choice, p_x and p_y match a full run of the same trials (INFERRED check); the default is unchanged.
  - `attach_adapter` returns an eval-mode unmerged adapter model applied after the write, and hooks still fire on the base layers and change output with a nonzero tiny adapter (INFERRED checks); `RunConfig.adapter` records the path.
  - `./scripts/check.sh` passes.
