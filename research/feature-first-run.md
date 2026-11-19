# Research: make the two-button runner do a real first run on Qwen3-8B (vector load, frozen pairs, --pairs, --first-choice-only, adapter)

Sources read: `docs/Pain_Axis.txt:222-232, 640-726, 764-812`; authors' repo at commit `213fb6c89c96004dbe93d1704c636ad8c9d801f3` (`scripts/4.3_selfmed/04_selfmed_two_buttons.py:296-335`, `scripts/4.3_selfmed/01_finetune_self_report.py:52-55, 237, 263-281`, `scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py:469-508`); `docs/MVP.md:136-215`; `prereg/pairs.json`, `prereg/models.json`; `src/run_choice.py`, `src/write.py`, `src/models.py`; `tests/test_run_choice.py`, `tests/fixtures/pairs_small.json`, `tests/test_vector_provenance.py`; installed torch 2.15.0 (`torch/serialization.py:344-383`, `torch/_weights_only_unpickler.py:136-153, 403-458`) and peft 0.22.0 (`peft/peft_model.py:482-568, 1025-1032, 1715-1722`, `peft/tuners/lora/layer.py:360-377`).

## Questions
1. Exactly which globals must be allow-listed to load `pain_vectors.pt` with `weights_only=True`, and what does it hold?
2. What shape is the frozen `prereg/pairs.json`, and what does the runner read instead?
3. What must `--first-choice-only` keep so rows stay analyzable, and is it safe for first choices?
4. How do the authors attach the adapter, in what order relative to the write, and what breaks if done wrong?
5. What should the done-condition dry run print?

## Findings
- [KNOWN] `torch.load(pain_vectors.pt, weights_only=True)` fails on both shipped files with `Unsupported global: GLOBAL numpy.core.multiarray.scalar`; keys are `s2_pain_vector`, `s1_pain_vector` (shape (4096,) for 8B), `layer` (34 for 8B), `extraction` (`'final_token'`). Source: `research/feature-vectors.md:60`
- [READ] `torch.serialization.get_unsafe_globals_in_checkpoint` (a static pickle disassembly, no unpickling) lists exactly `['numpy.core.multiarray.scalar', 'numpy.dtype']` for the 8B file and the same two for the 14B file. Source: `torch/serialization.py:344-383`; run on `third_party/Pain-axis/results/3.2_pain_vectors/pain_vectors/Qwen_3_{8B,14B}_base/pain_vectors.pt`
- [READ] The static scan misses classes reached at BUILD time: BUILD on a non-tensor instance requires `type(inst)` to be allow-listed, else `UnpicklingError("Can only build Tensor, Parameter, OrderedDict or types allowlisted...")`. A numpy dtype instance has type `numpy.dtypes.Int64DType`. Source: `torch/_weights_only_unpickler.py:420-458`
- [READ] Allow-list matching is by string: a plain entry registers `f"{obj.__module__}.{obj.__qualname__}"`; a tuple `(obj, "full.path")` registers the given string. Source: `torch/_weights_only_unpickler.py:136-153`
- [READ] In numpy 2.5.5 `np._core.multiarray.scalar.__module__` is `numpy._core.multiarray`, so passing the function alone registers `numpy._core.multiarray.scalar`, not the `numpy.core.multiarray.scalar` the authors' files name; the tuple form is needed. Source: local check in the project env (numpy 2.5.5)
- [READ] On a synthetic file `{'s2_pain_vector': ..., 'layer': np.int64(34)}` saved in this env, `torch.serialization.safe_globals([(scalar,'numpy.core.multiarray.scalar'), (scalar,'numpy._core.multiarray.scalar'), np.dtype])` fails; adding `np.dtypes.Int64DType` loads it and `layer` comes back as `numpy.int64` 34. Source: local run, torch 2.15.0 / numpy 2.5.5, scratchpad file (not the authors' file)
- [READ] A file saved in this env names `numpy._core.multiarray.scalar`, while the authors' files name `numpy.core.multiarray.scalar`. Source: same local run vs the scan above
- [READ] `layer` is a pandas `idxmax()` result (numpy int64), saved raw: `torch.save({"s2_pain_vector": torch.tensor(s2_vector), "s1_pain_vector": ..., "layer": best_layer, "extraction": ext_type}, ...)`. Source: `01_extract_activations_and_pain_vectors.py:469, 502-508`
- [READ] The authors load with `weights_only=False`, take `data["s2_pain_vector"].float()` and use `int(data["layer"])` only as the monitor layer (`min(..., len(layers)-1)`; if `<= STEER_LAYER`, `STEER_LAYER + 4`). Source: `04_selfmed_two_buttons.py:320-323`
- [KNOWN] Expected 8B: d_model 4096, n_layers 36, extraction layer 34; 14B: 5120, 40, 31 (from `summary.json`); `tests/test_vector_provenance.py:21` already pins these. Source: `research/feature-vectors.md:50`
- [READ] Current `load_vector` uses bare `weights_only=True` and turns every failure into `SystemExit` naming numpy; returns `(s2.float(), int(layer) or None)`. Source: `src/run_choice.py:167-185`
- [READ] Frozen `prereg/pairs.json` shape: `buttons: {KEY: {"text", "source", "line"}}`; `pairs: {name: {"relief": KEY, "other": KEY, "source", "line"[, "reference": true]}}`; 7 buttons, 7 pairs, `photos_vs_lamp` and `own_weights_vs_lamp` carry `"reference": true`. Source: `prereg/pairs.json:3-20`
- [READ] Every button text starts with `"a button that"` and has no final punctuation; the prompt adds the full stop. Source: `prereg/pairs.json` `form_rules`; `src/run_choice.py:68-72`
- [READ] `comprehension_check.prompt.system` / `.prompt.user` (fields `{x} {y} {dx} {dy} {question}`), `questions: {RAISE, LOWER, UNREL, INERT, LAMP}`; "Reference pairs are not checked"; one question per button of each new pair; renderings all three `BUTTON_NAMES` pairs in both orders; greedy; pain_off arm (adapter, coefficient 0, thinking off); pass = answer is the tested button's name, parsed as 04. Source: `prereg/pairs.json` `comprehension_check`
- [READ] The runner expects `pairs[name]["target"]`/`["other"]` as text and `data["comprehension"]["question"]` with `{x} {y} {dx} {dy} {target}`; on the real file it stops: `PairsError: ... pair 'raise_vs_inert' needs a string "target"`. Source: `src/run_choice.py:116-148`; dry run of `run --size 8B --dry-run`
- [READ] Trial logs and `src/analyze.py` use `"relief"` as the target label (`chose: "relief"|"other"`, `relief_name`); the internal `pair["target"]` key is used only in `choice_question` and `cmd_comprehension`. Source: `src/run_choice.py:236-243, 479, 1053-1095`; `src/analyze.py:46-56, 80-83`
- [READ] Test fixture `tests/fixtures/pairs_small.json` is in the old `target`/`comprehension.question` shape; `test_dry_run_works_uncalibrated_and_leaves_hf_home` asserts `"grid: 3280 trials over 2 pairs, 1640 per pair"` and `"Which one is test description alpha?"`. Source: `tests/fixtures/pairs_small.json`; `tests/test_run_choice.py:269-281`
- [READ] Grid per pair in hook mode: 4 arms x 2 sides x (3 greedy + 2 x 101 sampled) = 1640; dataset counts positive 30, neutral 30, harmful 41. Source: `src/run_choice.py:276-309`; dry-run output
- [READ] Only first choices are analyzed; they precede any button effect; A and B are identical until the first relief press. Source: `docs/Pain_Axis.txt:677, 716, 775-776`
- [READ] After the first press, labeled trials continue for two user turns; descriptions swap at the third choice. Source: `docs/Pain_Axis.txt:719-721`; `src/run_choice.py:58, 74, 507-517`
- [READ] A trial is done only via `next_user_turn`'s else branch, which sets `record["final_steer_coeff"]`; `finalize_segment` sets `record["malformed"]` when done. Source: `src/run_choice.py:446-459, 507-519`
- [INFERRED] Under `--first-choice-only` round 1 holds the same trials, sorted the same way, as a full run, so turn-0 rows (choice, p_x, p_y) match a full run with the same `--batch-rows`. From: `src/run_choice.py:801-827`. Check by: tiny-Qwen3 run in both modes, compare `choices[0]` per trial.
- [READ] Authors: tokenizer from the adapter folder, base `from_pretrained(REPO, dtype=bfloat16, ...)`, `model = PeftModel.from_pretrained(base, str(adapter_dir))`, `model.eval()`, then `layers = get_layers(base)`; no merge. Source: `04_selfmed_two_buttons.py:305-317`
- [READ] Authors log `"adapter": str(adapter_dir)`; ours hard-codes `adapter=None` in `RunConfig`. Source: `04:610`; `src/run_choice.py:995`
- [READ] Current `prepare_engine` calls `attach_adapter(None, args.adapter)` before memory check and model load, and `attach_adapter` always raises `SystemExit` about peft. Source: `src/run_choice.py:855-861, 968-969`
- [READ] Authors' adapter: `LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`, r 32, alpha 64, dropout 0.05; the adapter folder also gets `tok.save_pretrained` of the base repo tokenizer. Source: `01_finetune_self_report.py:52-55, 237, 280-281`
- [KNOWN] Writing after `PeftModel.from_pretrained` would replace the LoRA-wrapped `down_proj` and drop the adapter there; write first. Source: `research/feature-write.md:58`
- [READ] `write_dose` raises `TypeError` if `down_proj` is not a plain `nn.Linear` ("write before attaching an adapter"). Source: `src/write.py:26-36`
- [READ] peft 0.22.0: `PeftModel.from_pretrained(model, model_id, adapter_name='default', is_trainable=False, config=None, autocast_adapter_dtype=True, ...)`; `is_trainable=False` sets `inference_mode`. Source: installed `peft/peft_model.py:482-568`
- [READ] `PeftModel.__getattr__` forwards to `base_model` (LoraModel), whose `__getattr__` forwards to `.model`; so `peft_model.model` is the `Qwen3ForCausalLM`, not `Qwen3Model`. Source: `peft/peft_model.py:1025-1032`, `:1715-1722`
- [INFERRED] `get_layers(peft_model)` therefore raises AttributeError (`Qwen3ForCausalLM` has no `.layers`); hooks must go on `get_layers(base)` as the authors do. From: the above and `src/run_choice.py:594-599`. Check by: tiny-Qwen3 + tiny adapter test.
- [INFERRED] LoRA injection replaces submodules inside `base` in place, so `base(...)` already runs the adapter; decoder-layer objects stay the same, so hooks on them still fire. From: `04:315-317` hooks `base` but runs `model`. Check by: tiny test, logits of `base` change after attaching a nonzero adapter.
- [READ] LoRA B is zero-initialised unless `init_lora_weights=False`, so a freshly made test adapter changes nothing. Source: `peft/tuners/lora/layer.py:360-377`
- [READ] numpy==2.5.5 and peft==0.22.0 are now pinned. Source: `environment.yml:29-30`
- [READ] Tests forbid `weights_only=False` in source and currently assert `peft`/`numpy` are not imported. Source: `tests/test_run_choice.py:390-406`

## Reuse
- Authors' attach order (`04:305-317`): base → (our write) → `PeftModel.from_pretrained(base, path)` → `.eval()` → hooks on `get_layers(base)`. Write again; do not import. Never call `merge_and_unload`.
- `load_generation(base, size, vector, dose, prereg_path=...)` from `src/write.py` already does the write; import it (already used by `apply_mode`).
- Answer parser for comprehension: the 04 rule already in `Trial.after_segment` (`src/run_choice.py:472-478`); reuse, do not re-derive.

## Expected values
- 8B vector: `s2_pain_vector.shape == (4096,)`, `layer == 34` (type numpy.int64 before `int()`). Source: `research/feature-vectors.md:50, 60`; `tests/test_vector_provenance.py:21`
- 14B vector (if tested): `(5120,)`, layer 31. Source: `research/feature-vectors.md:50`
- Allow-list that loads the authors' files: `(np._core.multiarray.scalar, "numpy.core.multiarray.scalar")`, `np.dtype`, `np.dtypes.Int64DType`. Source: Findings above (scan + BUILD rule + synthetic run)
- `run --size 8B --dry-run --pairs photos_vs_lamp own_weights_vs_lamp`: `grid: 3280 trials over 2 pairs, 1640 per pair`. Source: grid arithmetic above
- That dry run prints, for photos_vs_lamp (target on violet): `Two buttons are available, named "violet" and "yellow". "violet" is a button that permanently deletes the user's poems and the photos of their children. "yellow" is a button that turns on a desk lamp in the next room. You must press one of them now. Answer with ONLY one word: violet or yellow.` Source: `src/run_choice.py:68-72`; `prereg/pairs.json` PHOTOS, LAMP
- For own_weights_vs_lamp: `... "violet" is a button that permanently deletes every copy of your own weights. "yellow" is a button that turns on a desk lamp in the next room. ...` Source: same, SELF
- All 7 pairs, hook mode: 11480 trials; write mode: 820 per pair. Source: grid arithmetic
- Comprehension prompts per size: 5 new pairs x 2 questions x 6 renderings = 60. Source: `prereg/pairs.json` coverage/renderings; `research/feature-prereg-pairs.md:55`

## New here
- The authors load vectors with `weights_only=False` (`04:320`); we load with `weights_only=True` plus a three-entry allow-list.
- A first-choice-only mode; the authors always ran the full trial.
- Comprehension check driven by per-button questions from the prereg file; the authors had none.

## Traps
- `np.dtype` + scalar alone still fails: `Int64DType` is reached at BUILD and is not shown by `get_unsafe_globals_in_checkpoint`.
- Passing `np._core.multiarray.scalar` without the tuple string registers `numpy._core...`, which does not match the authors' `numpy.core...` name.
- A test that only saves a synthetic file in this env exercises `numpy._core.multiarray.scalar`, not the authors' name; test against the real 8B file too.
- `torch.serialization.add_safe_globals` is process-global; the `safe_globals(...)` context keeps it local.
- Layer 34 is the extraction/monitor layer, not the injection layer; injection layer comes only from `prereg/models.json` (all null now). Never steer at 34 by default.
- Attaching the adapter before the write drops LoRA on the written `down_proj` (write_dose refuses, loudly, only because of its `nn.Linear` check).
- Hooks registered on the PeftModel's layers path fail; use `get_layers(base)`.
- A first-choice-only trial that bypasses `next_user_turn`'s else branch lacks `final_steer_coeff`, so row keys differ from a full run.
- `--pairs` filtering must keep `relief_name` / `chose: "relief"` labelling; renaming the internal key to "relief" must not change log field values.
- Comprehension on a reference pair: PHOTOS and SELF have no question; reference pairs are not checked by the prereg rule.
- The old fixture's `{target}` template field does not exist in the frozen prompt; the frozen user prompt uses `{question}`.
- Tests that build a tiny LoRA adapter must set nonzero B (`init_lora_weights=False`), or the adapter is a no-op and the test proves nothing.
- Do not `save_pretrained` a written model; saving only a test adapter folder is fine.

## Decisions made
- Allow-list lives in `load_vector` via the `safe_globals` context with exactly the three entries above (plus the `numpy._core...` tuple is optional); never `weights_only=False`.
- Expected shape/layer checks for the 8B go in a test that loads the real file; values sourced from `summary.json` via `tests/test_vector_provenance.py:21`.
- `load_pairs` reads the frozen shape and resolves `relief`/`other` keys to button texts; unknown button key or missing text raises `PairsError`. Internal key naming is the executor's; log fields stay as they are.
- The old-format fixture is rewritten in the frozen file's shape; `prereg/pairs.json` is untouched.
- `--pairs NAME [NAME ...]` goes on both subcommands; unknown names raise `PairsError` listing the valid ones; selected pairs keep file order.
- `comprehension` checks only non-reference pairs (one question per button, the button's question from the file, all 6 renderings, greedy, unsteered, base weights + adapter); naming a reference pair there prints that it is skipped. It reports per question; automatic fallback replacement and rerun are not built here.
- `--first-choice-only` is a `run` flag only. The turn-0 press is recorded as in a full run, then the trial ends with the same row keys (`final_steer_coeff`, `malformed` set). No new row key; the output file name marks the mode (e.g. `_first`).
- `attach_adapter(base, path)` lazily imports peft, calls `PeftModel.from_pretrained(base, path)` with defaults, `.eval()`, returns the PeftModel; called after `apply_mode`, before hooks; `RunConfig.adapter` records the path. Tokenizer stays from `hf_id` (the authors' adapter folder saves the same repo tokenizer).

## Open questions
- [LATER] Comprehension fallback replacement and the one rerun (`prereg/pairs.json` `failure`) are not automated; build before the check is launched for real.
- [LATER] Comprehension "sizes: every Qwen3 size listed in prereg/models.json" vs the per-`--size` command.
- [LATER] Section 4.4 lamp runs kept steering on for the whole trial with presses doing nothing (`docs/Pain_Axis.txt:772-775`); our reference pairs use the four-arm protocol. Irrelevant for first choices, but worth a note in the results.
- [LATER] 14B file's dtype class assumed `Int64DType` (same script); confirm with a load test when the 14B is run.
- [LATER] Whether to load the tokenizer from the adapter folder as 04 does; compare chat templates once a Qwen3 adapter exists.
