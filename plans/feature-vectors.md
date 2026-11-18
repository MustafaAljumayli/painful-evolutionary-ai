# Plan: Pin the evidence that the authors' Qwen3 8B and 14B pain vectors come from Qwen/Qwen3-8B and Qwen/Qwen3-14B

The finding is already in `research/feature-vectors.md`: both vectors were
extracted from `Qwen/Qwen3-8B` and `Qwen/Qwen3-14B` (the post-trained
checkpoints, which have no separate instruct release). The folder suffix
`_base` is a file-name label only and does not mean the `-Base` checkpoints.
Verdict: VALID for both sizes. The revision was not recorded. That is a LATER
question and does not change the verdict. No `src/` module is added. The two
subtasks below add CPU-only tests. They read the submodule as data, so the
evidence the verdict rests on stays checked if the submodule pin moves.

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

### S2: Test that the stored vectors have the claimed checkpoint's shape and keys
- Owns: `tests/test_vector_shapes.py`
- Depends on: none
- Do: Write one CPU-only pytest file, parametrized over the two folder names,
  with an expected d_model of 4096 for 8B and 5120 for 14B (Interfaces). This
  runs the brief's check for its INFERRED finding that the 16384- and
  20480-byte storages hold 4096 and 5120 float32 elements.
  For each size, load
  `ROOT/results/3.2_pain_vectors/pain_vectors/<name>/pain_vectors.pt` with
  `torch.load(path, map_location="cpu", weights_only=True)`. Never use
  `weights_only=False` and never unpickle any other way. Assert:
  - the loaded object is a dict whose key set is exactly
    `{"s2_pain_vector", "s1_pain_vector", "layer"}` (brief, Findings on
    `data.pkl` and the writer at `01_extract...py:503-508`);
  - `s2_pain_vector` and `s1_pain_vector` are tensors with
    `shape == (d_model,)` and `dtype == torch.float32` (brief: FloatStorage,
    16384 / 20480 bytes).
  Do not assert the value of `layer`. The brief leaves the match to 34 / 31
  as a LATER question with no READ value. Skip per the Interfaces rule if a
  file is missing. Type-hint every function and give each a docstring.
- Done when:
  - `python -m pytest tests/test_vector_shapes.py -q -rs` passes with no
    skips in an initialized worktree. This settles the brief's INFERRED check:
    "loading with `weights_only=True` and asserting `shape == (4096,)` / `(5120,)`".
  - `./scripts/check.sh` exits 0, and the test takes well under a few seconds
    on CPU.
  - `grep -n weights_only tests/test_vector_shapes.py` shows only `True`.
  - `git status` shows no change under `third_party/`.
