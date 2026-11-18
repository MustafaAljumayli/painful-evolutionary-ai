# Research: src/models.py, per-size Qwen3 settings with calibrated layer and coefficient from prereg/models.json

Sources read: `third_party/Pain-axis` is missing in this worktree, so the authors' script
`scripts/4.3_selfmed/04_selfmed_two_buttons.py` was read from the web copy at the pinned commit
`213fb6c89c96004dbe93d1704c636ad8c9d801f3` (raw.githubusercontent.com/valen-research/Pain-axis/213fb6c.../).
Hugging Face `config.json` for Qwen3-8B (repo sha `b968826d9c46dd6066d109eabc6255188de91218`),
Qwen3-14B (`40c069824f4251a91eefaf281ebe4c544efd3e18`) and Qwen3-32B (`9216db5781bf21249d130ec9da846c4624c16137`),
fetched 2026-11-17. Qwen3-8B model card README. `CLAUDE.md`, `docs/MVP.md`, `docs/PROPOSAL.md`,
`pyproject.toml`, `scripts/check.sh`. The paper PDF was not opened: this feature needs nothing from it beyond what the docs quote.
No earlier briefs, `src/` holds only `__init__.py`, `tests/` holds only `__init__.py`, `prereg/` does not exist yet.

## Questions
1. How many decoder layers does each Qwen3 size have? (This sets the valid injection-layer range.)
2. Is the layer index 0-based into `model.layers`, so that the valid range is `0 <= L < num_hidden_layers`?
3. Which Qwen 2.5 values must never show up as defaults?
4. How is thinking turned off for Qwen3?
5. What shape should `prereg/models.json` have, and what counts as "uncalibrated"?

## Findings
- [READ] Qwen3-8B: `"num_hidden_layers": 36` (config.json line 18), `"hidden_size": 4096`. Source: https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json
- [READ] Qwen3-14B: `"num_hidden_layers": 40` (config.json line 18), `"hidden_size": 5120`. Source: https://huggingface.co/Qwen/Qwen3-14B/blob/main/config.json
- [READ] Qwen3-32B: `"num_hidden_layers": 64` (config.json line 18), `"hidden_size": 5120`. Source: https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json
- [READ] All three configs: `"architectures": ["Qwen3ForCausalLM"]`, `"model_type": "qwen3"`, `"attention_bias": false`, no `mlp_bias` key, `"torch_dtype": "bfloat16"`, `"transformers_version": "4.51.0"`. Source: the same three URLs.
- [READ] The HF ids are `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B`, `Qwen/Qwen3-32B`, the post-trained versions; `-Base` would be pretrained-only. Source: `docs/MVP.md:16`, `CLAUDE.md` (Vocabulary, Models).
- [READ] The authors' layer index is a 0-based Python index into the decoder block list: `get_layers(model)` returns `model.model.layers` (`s04.py:139-143`) and the hook goes on `layers[STEER_LAYER]` (`s04.py:399`); the monitor layer is clamped to `len(layers) - 1` (`s04.py:322-324`). Source: authors' `04_selfmed_two_buttons.py` at 213fb6c.
- [INFERRED] So the valid injection layer for a size is `0 <= L <= num_hidden_layers - 1`: 0..35 for 8B, 0..39 for 14B, 0..63 for 32B. From: the lines above plus the configs. Check by: on GPU, `len(model.model.layers) == num_hidden_layers` for each size.
- [READ] The authors' per-model table is `(repo, name, steer_layer, coeff, parser, batch rows)`: Qwen2.5-7B `16, 1.0`, Qwen2.5-32B `38, 1.0`, Qwen2.5-72B `46, 1.25`, batch rows 1024, 384, 192. Source: `s04.py:73-78`.
- [READ] Qwen 2.5 values (layer 16 or 38, coefficient 1.0, batch 384) must not be reused for Qwen3; layer and coefficient are set per Qwen3 size by calibration and frozen in `prereg/`. Source: `CLAUDE.md` (Rules), `docs/MVP.md:18`, `docs/MVP.md:117`, `docs/MVP.md:206`, `docs/PROPOSAL.md:83`.
- [READ] The calibration procedure is still TBD (authors' judge-based selection or a sweep). Source: `docs/MVP.md:117`. This module does not need it settled: it only stores the result.
- [READ] Qwen3 thinking is switched by `enable_thinking` in `tokenizer.apply_chat_template`, and it defaults to True. Source: https://huggingface.co/Qwen/Qwen3-8B (README lines 70, 115, 122). Thinking must be off in every run (`CLAUDE.md` Rules; `docs/MVP.md:113`).
- [READ] Ruff requires type hints and docstrings on public functions; tests are exempt. Every `src/` module needs `tests/test_<module>.py`; CPU-only tests run through `scripts/check.sh`. Source: `pyproject.toml:15-18`, `CLAUDE.md` (Standards).
- [READ] Dose cap 1.5 is a multiple of the working coefficient, not a coefficient value. Source: `CLAUDE.md` (Vocabulary), `docs/PROPOSAL.md:182`.

## Reuse
- Layer-indexing convention (0-based into `model.model.layers`) from `s04.py:139-143,399`: follow it, do not import. Nothing else from the authors' code applies; `models.py` is written new.
- Do not copy the authors' `MODELS` rows (`s04.py:74-78`): they are Qwen 2.5 values.
- Layer counts 36 / 40 / 64: hard-code them as constants with the config.json URL in a comment.

## Expected values
- `num_layers`: 8B = 36, 14B = 40, 32B = 64 (configs above, at the repo shas recorded in Sources read).
- Accepted layers: 8B accepts 35 and rejects 36; 14B accepts 39 and rejects 40; 32B accepts 63 and rejects 64; every size rejects -1.
- `thinking` is False for all three.
- Without a prereg file or with null entries, reading the layer or coefficient raises the uncalibrated error for every size.

## New here
The authors keep one hard-coded Qwen 2.5 table with values fixed in the code (`s04.py:74-78`). Here the values for each size
start empty and come from a frozen prereg file, are checked against the layer count, and asking for one before it is set raises an error.
The authors have no Qwen3 rows at all.

## Traps
- A falsy check (`if not layer`) would treat a calibrated layer 0 as missing. Test for `None`, not truthiness.
- `bool` is a subclass of `int` in Python, so `true` in JSON would pass as layer 1. Reject bools explicitly.
- Off-by-one: `L == num_layers` must be rejected. Because the authors clamp the monitor layer (`s04.py:322`), an out-of-range layer could be silently clamped instead of failing downstream. Validate here.
- A float layer from JSON (`16.0`) should be rejected, not cast with `int()`.
- Defaults like `layer: int = 16` or `coef: float = 1.0` would silently bring back Qwen 2.5 values. Use `None` only. The value 16 itself is still a legal *calibrated* layer for every Qwen3 size, so do not reject the value. Reject it only as a default.
- Thinking defaults to True in the Qwen3 chat template (README line 70). The record's `thinking=False` must be passed as `enable_thinking=False` by callers; this module cannot enforce it alone.
- Configs on `main` can change; the layer counts were read at the shas above.

## Decisions made
- One frozen dataclass `ModelSpec(size: str, hf_id: str, num_layers: int, thinking: bool = False)`, plus a module dict `MODELS: dict[str, ModelSpec]` keyed `"8B"`, `"14B"`, `"32B"`. Frozen so no caller can set a layer on it.
- Calibrated values are not stored on the record. Two functions read them: `injection_layer(size: str, prereg_path: Path = PREREG_PATH) -> int` and `working_coefficient(size: str, prereg_path: Path = PREREG_PATH) -> float`. `PREREG_PATH` = repo root / `prereg/models.json`. Tests pass a `tmp_path` file.
- `prereg/models.json` schema: `{"8B": {"injection_layer": int | null, "working_coefficient": float | null}, "14B": {...}, "32B": {...}}`. A missing file, missing size key, missing field or `null` all count as uncalibrated.
- `class UncalibratedError(LookupError)`; the message names the size, the field and the prereg path. An invalid layer raises `ValueError` naming the size, the value and the range `0..num_layers-1`.
- An unknown size string raises `KeyError` listing the valid sizes.
- The coefficient must be a finite number > 0 (int or float, not bool); otherwise `ValueError`. Chosen because a zero or negative working coefficient makes "dose" meaningless; it does not fix any value.
- This feature does not create `prereg/models.json`: the values are frozen by calibration (`docs/MVP.md:117`), and a missing file already reads as uncalibrated.
- `hidden_size` is not added; nothing in the request needs it.
- Tests are CPU-only, use no network and no `transformers`.

## Open questions
- [LATER] Pin the HF `revision` (shas above) in `ModelSpec` so that runs and configs cannot drift.
- [LATER] Add `hidden_size` (4096 / 5120 / 5120) so that `write.py` can check the vector length.
- [LATER] Who writes `prereg/models.json`, and should it record the calibration procedure (`docs/MVP.md:117`, still TBD) and a date next to the values?
- [LATER] Confirm on GPU that `len(model.model.layers)` equals the config count and that `mlp.down_proj.bias is None` for each size (`docs/MVP.md:103`). The configs' `attention_bias: false` and missing `mlp_bias` suggest no bias, but that is not checked.
- [LATER] Whether the 32B vector's own cross-validated layer (`data["layer"]`, `s04.py:322`) should constrain the injection layer choice.
