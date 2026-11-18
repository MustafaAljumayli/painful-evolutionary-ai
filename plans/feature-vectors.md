# Plan: Pin the evidence that the authors' Qwen3 8B and 14B pain vectors come from Qwen/Qwen3-8B and Qwen/Qwen3-14B

The finding is already in `research/feature-vectors.md`: both vectors were
extracted from `Qwen/Qwen3-8B` and `Qwen/Qwen3-14B` (the post-trained
checkpoints, which have no separate instruct release). The folder suffix
`_base` is a file-name label only and does not mean the `-Base` checkpoints.
Verdict: VALID for both sizes. The revision was not recorded. That is a LATER
question and does not change the verdict. No `src/` module is added. S1 adds
one CPU-only test that reads the submodule as data, so the evidence the
verdict rests on stays checked if the submodule pin moves. S2 only removes an
uncommitted draft and adds no test (see S2 and Changes).

Why no shape test. The verdict rests on the extraction ids that S1 pins, not
on shapes. The brief (Findings, `-Base` config line) records that
`Qwen/Qwen3-8B-Base` and `Qwen/Qwen3-14B-Base` share both `num_hidden_layers`
and `hidden_size` with the post-trained checkpoints, so a shape check cannot
tell the checkpoints apart and adds nothing to the finding. A stdlib-only
pickle-opcode test would be fragile, and its key set is not a READ finding in
the brief. The shape and `layer` check stays the brief's LATER item, to be
run at calibration, where torch and numpy are both installed.

Correction to the brief. The brief assumes `torch.load(..., weights_only=True)`
loads `pain_vectors.pt` cleanly (INFERRED check line and LATER item on
loading). That is wrong as stated. Source: S2 executor's BLOCKED report on
this branch. The call raises `_pickle.UnpicklingError: Unsupported global:
GLOBAL numpy.core.multiarray.scalar` for both folders, because `layer` is
stored as a numpy int64 scalar. Disassembly of the 8B `data.pkl` without
unpickling also showed an `extraction` key (`'final_token'`) that the brief
does not list. Both points are recorded in the brief's Traps by the orchestrator,
because no subtask may own `research/`.

## Interfaces

- Submodule root, resolved from a test file:
  `ROOT = Path(__file__).resolve().parents[1] / "third_party" / "Pain-axis"`.
  Never write under it. Open files read-only.
- If a file a test needs is missing (submodule not initialized), the test
  calls `pytest.skip("third_party/Pain-axis not initialized; run git submodule update --init")`.
  It must not fail and must not fall back to other data.
- Folder-name to HF id map, copied as data (brief, Reuse, from
  `scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py:87-88`):
  `{"Qwen_3_8B_base": "Qwen/Qwen3-8B", "Qwen_3_14B_base": "Qwen/Qwen3-14B"}`.
  Each test file defines its own copy. Neither imports from the other.
- Expected per-size values (brief, Findings and Expected values, from
  `results/3.2_pain_vectors/per_model/Qwen_3_{8B,14B}_base/summary.json:2-6`):
  - `Qwen_3_8B_base`: model `"Qwen/Qwen3-8B"`, n_layers 36, d_model 4096, best_layer_final_token 34.
  - `Qwen_3_14B_base`: model `"Qwen/Qwen3-14B"`, n_layers 40, d_model 5120, best_layer_final_token 31.
- No test marker. Both files must run under `./scripts/check.sh` on a laptop CPU.

## Subtasks

### S1: Test that the extraction ids in the submodule are the post-trained Qwen3 checkpoints
- Owns: `tests/test_vector_provenance.py`
- Depends on: none
- Do: Write one CPU-only pytest file with a module docstring that says it pins
  the provenance evidence from `research/feature-vectors.md`. Use the
  Interfaces map and expected values. Parametrize over the two folder names.
  Tests:
  1. `summary.json`: load
     `ROOT/results/3.2_pain_vectors/per_model/<name>/summary.json` with `json`.
     Assert `model` equals the mapped HF id, `model_name` equals `<name>`, and
     `n_layers`, `d_model` and `best_layer_final_token` equal the Interfaces
     values. Source: brief Findings line on `summary.json:2-6`.
  2. Extraction script: read
     `ROOT/scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py`
     as text. Do not import or exec it. Assert it contains the tuple literals
     `("Qwen/Qwen3-8B", "Qwen_3_8B_base")` and `("Qwen/Qwen3-14B", "Qwen_3_14B_base")`.
     Assert it does not contain `"Qwen/Qwen3-8B-Base"` or `"Qwen/Qwen3-14B-Base"`.
     Source: brief, `01_extract...py:87-88`.
     If the literal spacing or quoting does not match on first run, look at
     the actual lines 87-88 and match them exactly. Do not loosen the check
     to a bare substring of the folder name.
  3. Load path: assert the same script text contains
     `AutoModelForCausalLM.from_pretrained(model_path` (brief, `:423-428`).
     This shows that the weights come from the registry string and that no
     alias map exists.
  Add a `_path(*parts)` helper that skips per the Interfaces rule when the
  file is missing. Type-hint every function and give each a one-line
  docstring. Do not edit `third_party/`.
- Done when:
  - `python -m pytest tests/test_vector_provenance.py -q -rs` passes with
    no skips in an initialized worktree.
  - `./scripts/check.sh` exits 0.
  - A reviewer reads the file and finds that every asserted literal and
    number appears in the Interfaces section above, with its source cited in
    a comment beside it.
  - `git status` shows no change under `third_party/`.

### S2: Remove the uncommitted shape-test draft
- Owns: `tests/test_vector_shapes.py` (deletion only)
- Depends on: none
- Do: Delete the untracked draft `tests/test_vector_shapes.py` from the
  working tree. It was never committed, so there is nothing to commit; do not
  write a replacement. Do not load or unpickle any `.pt` file, do not add any
  dependency, and do not touch `pyproject.toml`, `scripts/`, `research/` or
  `third_party/`. This is option (a) from the re-plan request: the shape and
  `layer` check stays the brief's LATER item for calibration, for the reasons
  given at the top of this plan.
- Done when:
  - `test -e tests/test_vector_shapes.py` fails (the file is gone), and
    `git status --porcelain -- tests/` shows nothing for it.
  - `./scripts/check.sh` exits 0. (Per the executor's note, one file can be
    run with `python -m pytest -p no:langsmith tests/test_vector_provenance.py -q`
    if a global langsmith plugin crashes on import.)
  - `git status` shows no change under `third_party/`, and `git log` shows S1's
    commit `de8437d` unchanged.

## Brief correction (done)

- The orchestrator added the Traps line and the caveat on the LATER loading
  item to `research/feature-vectors.md`, sourced to the S2 executor's BLOCKED
  report. It records only what the executor observed: the `weights_only=True`
  failure on the numpy scalar, and the `extraction` key seen in the 8B
  `data.pkl` disassembly. The planner's unsourced suggestion to use
  `add_safe_globals` was left out. No human decision is open.

## Changes

- S1: unchanged (passed, committed as `de8437d`). Do not run again.
- S2: rewritten. It no longer writes a test. It only deletes the uncommitted
  draft `tests/test_vector_shapes.py`. Run S2 again.
- Top of plan: added why there is no shape test and the correction to the
  brief's `weights_only=True` assumption.
- Added `## Brief correction (done)` for the Traps line in the brief.
