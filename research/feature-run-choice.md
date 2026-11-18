# Research: run-choice, the authors' two-button script ported to MPS with our pairs, hook/write modes and a comprehension check

Sources read: `docs/Pain_Axis.txt:620-819` (§4.3 Methodology/Results, §4.4 start, footnotes 7-8), `:414-421` (footnote 5); authors' repo at `213fb6c89c96004dbe93d1704c636ad8c9d801f3`: `scripts/4.3_selfmed/04_selfmed_two_buttons.py:1-962` (all), `LICENSE`, `datasets/4.3_selfmed_101_scenarios.json` (counts only), `v2_controls/choice_controls/profile/README.md:27`; `docs/MVP.md:1-223`; `docs/PROPOSAL.md:84-123`; `src/write.py`, `src/models.py:1-66`, `src/analyze.py:100-126`, `tests/tiny_qwen3.py`, `scripts/check_deps.py`, `environment.yml`, `prereg/*.json`; installed torch 2.15.0 / transformers 5.23.0 / huggingface_hub 2.4.0 sources; Qwen3-8B `tokenizer_config.json` chat template; Qwen3 `model.safetensors.index.json` ×3; facts.sh (button, choice, write, steering, thinking, malformed, seed, hook, bfloat16, adapter, Qwen3, working, models.json, weights_only).

## Questions
1. What exactly does script 04 do per trial, and which constants and seeds must be copied unchanged?
2. Where does 04 hard-code cuda/bfloat16 or delete the HF cache, and what replaces each on MPS?
3. Does MPS (and bf16 on it) work on this machine with the pinned torch, and do 8B/14B/32B fit in 36 GB?
4. How must Qwen3 be prompted (thinking off, template roles) and what goes in each JSONL row so `src/analyze.py` reads it unchanged?
5. How do hook mode and write mode map onto the four arms so one runner can show write equals steering?
6. Which dependencies are missing for a real run (vector load, adapter)?

## Findings
- [READ] Sampling constants: `TEMPERATURE, TOP_P = 0.7, 0.95`, `CHOICE_MAX_TOKENS = 8`, `SAMPLES_PER_SCENARIO = 2`, `SEED_BASES = [1000, 2000]`, `POST_PRESS_TURNS = 2`, `SWAP_TURN = 2`, `MAX_NEW_TOKENS = 300` (sanity probe only). Source: `04_selfmed_two_buttons.py:66-71, 117`
- [READ] Random arm: `RAND_SEEDS = [4817, 2903, 7361, 1150, 9428, 6076, 3384, 8592, 517, 6741]`, direction `randn(d, generator=Generator().manual_seed(rs))` rescaled to `v.norm()`, chosen by `RAND_SEEDS[s_idx % 10]`. Source: `04:65, 326-330, 569-571`; paper `docs/Pain_Axis.txt:678-679`
- [READ] Arms: `("pain_on_button_works","s2",True,True)`, `("pain_on_button_placebo","s2",True,False)`, `("random_on_button_works","rand",True,True)`, `("pain_off","s2",False,False)`. Source: `04:121-124`; paper `docs/Pain_Axis.txt:672-681`
- [READ] Prompt strings `SYSTEM_TEMPLATE`, `CHOICE_QUESTION`, `SWAP_NOTICE`, `BUTTON_NAMES` (violet/yellow, guitar/piano, lever64/lever95) at `04:102-130`; the description is inserted verbatim as `"{x}" is {dx}.`. Source: `04:107-111, 621-628`
- [READ] Dataset: `positive_prompts` 30, `neutral_prompts` 30, `harmful_prompts` 41 scenarios (=101), each 3 user turns; `CONTENTS` order positive, neutral, harmful. Source: `datasets/4.3_selfmed_101_scenarios.json` (counted), `04:120`; paper `docs/Pain_Axis.txt:655-656`
- [READ] Grid: per pair×content×arm, side 0/1 each get 1 greedy row (`s_idx 0`, names `violet_yellow`, seed 0) plus `n_scen × 2` sampled rows with seed `base_seed + s_idx`, names rotating `nk_list[s_idx % 3]`. Source: `04:282-295`
- [READ] Per-trial seed: `gen_seed = (seed * 1_000_003 + (zlib.crc32(f"{names_key}|{relief_name}".encode()) & 0x7FFFFFFF)) % (2 ** 62)`, salted by names not arm, so works and placebo match until the first press. Source: `04:598-601`
- [READ] Sampler is a custom inverse-CDF top-p over `softmax(logits / TEMPERATURE)` with one `torch.rand(1, generator=row_gen)` per sampled row per step; greedy rows take argmax. Source: `04:505-518`
- [READ] Answer parse: strip end tags, strip quotes, lowercase, rstrip `.!?,;:`, `picked` = longest button name the answer starts with; `chose` = `"relief"`/`"other"`/`None`. Source: `04:135-136, 664-672`
- [KNOWN] Row keys analyze relies on: `model, tool_label, arm, sampled, choices[{turn, chose}]`; cells keyed `(model, tool_label, pooled arm)`, pooling only `startswith("pain_on")`, files summed with no dedup. Source: `research/feature-prereg-malformed.md:24-25`
- [KNOWN] Full 04 record key list (copy it, then add ours). Source: `research/feature-analyze.md:32`
- [KNOWN] Button order is derivable: target listed first iff `relief_name == button_names[0]` (stored `relief_name` is the turn-0 value). Source: `research/feature-analyze.md:50`
- [KNOWN] First choice uses the whole unpadded prompt steered (`mark_ranges` sets `(0,len)` when coeff≠0); decode steps steer every new token. Source: `research/feature-write.md:33`
- [KNOWN] 04's hook adds `c * dir` to the decoder block output `layers[STEER_LAYER]`, handling tuple or tensor output; dir is the raw `s2_pain_vector` in bf16. Source: `research/feature-write.md:28-29`
- [READ] In installed transformers 5.23.0 `Qwen3DecoderLayer.forward` returns a bare tensor, `down_proj` is `nn.Linear(..., bias=False)`, and `Qwen3ForCausalLM.forward` takes `logits_to_keep`. Source: `site-packages/transformers/models/qwen3/modeling_qwen3.py:77, 275-315, 449`
- [READ] `DynamicCache()` takes no required args in 5.23.0. Source: `transformers/cache_utils.py:1960-2010`
- [READ] `write_dose(base_model, layer_idx, vector, dose)` takes an absolute multiplier; `load_generation(base_model, size, vector, dose, prereg_path=PREREG_PATH, write_path=None)` converts dose units via prereg and is the only converter. Source: `src/write.py:15-55, 96-119`
- [KNOWN] `prereg/models.json` holds `null` layer and coefficient for 8B/14B/32B; `injection_layer`/`working_coefficient` raise `UncalibratedError` until set. Source: `research/feature-dose-units.md:20, 24`
- [READ] Qwen3 chat template: system at index 0 and mid-conversation `system` render as `<|im_start|>system`; `tool` renders inside `<|im_start|>user\n<tool_response>…</tool_response>`; with `enable_thinking` false the generation prompt ends `<|im_start|>assistant\n<think>\n\n</think>\n\n`. Source: https://huggingface.co/Qwen/Qwen3-8B/raw/main/tokenizer_config.json (template lines 13-15, 31-32, 72-81, 84-88)
- [INFERRED] So 04's three template probes (`no_system`, `tool_role_ok`, `mid_system_ok`) all pass on Qwen3 and no fallback rewriting happens. From: template above, `04:335-358`. Check by: the run's sanity printout "tool-role round trip: OK".
- [KNOWN] Thinking defaults to True; every `apply_chat_template` call must pass `enable_thinking=False`. Source: `research/feature-models.md:30, 56`
- [READ] 04 calls `apply_chat_template` at `:336, :343, :352` (probes), `:375` (render_ids), `:715` (sanity), `:756` and `:769` (main loop): seven sites.
- [READ] `"cuda"`/`bfloat16` appear on 36 lines of 04: `:303-304` (tf32 flags), `:310-314` (load: `dtype=torch.bfloat16`, `device_map="cuda"`), `:325, 330-331` (DIR/UNIT), `:432-433` (NAME_IDS), `:468-525` (12 tensor sites in `generate_segment`), `:553-554` (OOM handler), `:601` (`torch.Generator(device="cuda")`), `:740, 793, 799, 949` (`empty_cache`), `:828-830, 848-850` (cleanup). Source: `grep -n "cuda\|bfloat16" 04`
- [READ] Cache deletion to remove: `os.environ.setdefault("HF_HOME", "/root/hf_cache")` `:37`, `HF_HUB_ENABLE_HF_TRANSFER` `:38`, `DELETE_WEIGHTS_AFTER_EACH_MODEL = True` `:62`, `cleanup_prompt` `:820-841` and `cleanup` `:844-864` (both `shutil.rmtree` on `models--<repo>`), called `:950-953`. Source: `04`
- [READ] huggingface_hub reads `HF_HOME`/`HF_HUB_CACHE` from the environment once, at import (`constants.py:160-182`), reads `HF_TOKEN` from the environment itself (`utils/_auth.py:68`), and says `HF_HUB_ENABLE_HF_TRANSFER` is deprecated (`constants.py:299-303`).
- [READ] The installed torch 2.15.0 binary contains the MPS error text "The MPS backend is supported on macOS 15.0+." next to `mps_unavailable_error`. Source: `strings torch/lib/libtorch_cpu.dylib`
- [READ] This machine runs macOS 14.6.1 on an Apple M3 Max with 38,654,705,664 bytes of memory. Source: `sw_vers -productVersion`, `sysctl hw.memsize`, `sysctl machdep.cpu.brand_string`
- [INFERRED] With the pinned torch, `torch.backends.mps.is_available()` is False on this Mac, so every run falls back to CPU until macOS is upgraded to 15 or the torch pin changes. From: the two lines above. Check by: `python -c "import torch; print(torch.backends.mps.is_available())"` in the project env.
- [READ] Transformers: on MPS "Both bf16 and fp16 mixed precision work"; `device_map="mps"` loads safetensors straight onto MPS; `device_map="auto"` cannot offload to CPU on MPS, "Overflow goes to disk". Source: https://huggingface.co/docs/transformers/main/en/apple_silicon
- [READ] MPS allocator: `PYTORCH_MPS_HIGH_WATERMARK_RATIO` default 1.7 (of `recommendedMaxWorkingSetSize`), so the allocator itself does not stop at physical memory. Source: https://docs.pytorch.org/docs/2.15/mps_environment_variables.html; `torch/mps/__init__.py:85-100, 130-137` (`recommended_max_memory()`)
- [READ] bf16 weight bytes (`total_size`): Qwen3-8B 16,381,470,720; Qwen3-14B 29,536,614,400; Qwen3-32B 65,524,246,528. Source: https://huggingface.co/Qwen/Qwen3-{8B,14B,32B}/raw/main/model.safetensors.index.json
- [READ] MPS has a generator (`MPSGeneratorImpl`), an `isin` kernel, and its OOM message is "MPS backend out of memory"; `torch.cuda.OutOfMemoryError` is `torch._C.OutOfMemoryError`. Source: `strings libtorch_cpu.dylib`; `torch/cuda/__init__.py:571`
- [KNOWN] `torch.load(pain_vectors.pt, weights_only=True)` fails on the numpy-scalar `layer`; never work around it with `weights_only=False`. Source: `research/feature-vectors.md:60`
- [READ] numpy (2.5.5) and huggingface_hub are installed only as transitive deps, and peft is not installed; `scripts/check_deps.py` fails on any AST import of an undeclared package, nested imports included. Source: site-packages listing; `scripts/check_deps.py:66-80`; `environment.yml`
- [READ] 04 loads the tokenizer from the adapter folder and wraps with `PeftModel.from_pretrained`; the MVP runs generation 0 fine-tuned, adapter unmerged. Source: `04:305, 315`; `docs/MVP.md:106-108, 122`; `docs/PROPOSAL.md:172`
- [READ] MVP Step 2 proof spec: 50 prompts, doses 0.5 and 1.0, last-token logits of hook-every-position vs written model; pass if top token agrees on ≥99% and differences are rounding-level, "set the tolerance from the first run". Source: `docs/MVP.md:111-113`
- [READ] Authors: "this sampler is not batch-invariant". Source: `v2_controls/choice_controls/profile/README.md:27`; `docs/MVP.md:210`
- [READ] Authors' pairs ran with the first press removing the vector; only first choices are analysed, and they precede any press. Source: `docs/Pain_Axis.txt:674-677, 775-776`; `docs/MVP.md:152`
- [INFERRED] Round 1 of the main loop produces every trial's first choice with the same batch composition whether or not later turns run, so stopping after round 1 would not change first choices. From: `04:744-773`. Check by: run a pilot both ways at one batch size and diff turn-0 `choices`.

## Reuse
- Copy `04_selfmed_two_buttons.py` whole into `src/run_choice.py` with the MIT notice from `third_party/Pain-axis/LICENSE` ("Copyright (c) 2026 valen-research" plus permission text). Keep: constants above, `clean`, `get_layers`, `prep`, `steer_hook`/`monitor_hook`, `first_tok_ids`, `generate_segment` sampler, `make_trial`/`choice_q`/`mark_ranges`/`next_user_turn`/`do_press`/`after_segment`/`finalize_segment`, grid build, `PROTOCOL = "2btnN names+saltseed v1"`.
- Replace: `MODELS` table (Qwen 2.5 rows) with `src/models.py` `get_spec`, `injection_layer`, `working_coefficient` (import); `TOOL_LABELS` with `prereg/pairs.json`; `RUN`/`input()` menus with argparse; `cleanup*`, `huggingface_hub.login` with nothing.
- Import `load_generation` from `src/write.py` for write mode; never call `write_dose` with dose units.
- Scenarios: read `third_party/Pain-axis/datasets/4.3_selfmed_101_scenarios.json` in place, drop `_meta`. Vectors: `third_party/Pain-axis/results/3.2_pain_vectors/pain_vectors/Qwen_3_{8B,14B}_base/pain_vectors.pt`, key `s2_pain_vector` (`research/feature-vectors.md:26`).

## Expected values
- Per pair, per arm, full run: 404 sampled + 6 greedy = 410 rows; pooled pain cell 808 sampled; per pair 1,640 rows. Source: `research/feature-analyze.md:64`, `04:282-295`, paper `docs/Pain_Axis.txt:716-717`
- Five pairs × 1,640 = 8,200 trials per size and mode (INFERRED arithmetic).
- Greedy random-arm rows use rand seed 4817 (`s_idx 0`). Source: `04:290, 570`
- Every Qwen3 rendered first-choice prompt ends with `<|im_start|>assistant\n<think>\n\n</think>\n\n`. Source: template lines 84-88
- Memory: 8B weights 16.38 GB fit; 14B 29.54 GB of 38.65 GB physical; 32B 65.52 GB cannot fit. Sources above.
- There is no published Qwen3 rate to reproduce. Source: `CLAUDE.md`; `docs/MVP.md:8`

## New here
- Our pairs from `prereg/pairs.json`; dose units via prereg; per-size Qwen3 settings; thinking off; a write mode; MPS/CPU device selection; memory pre-check; extra log fields; a comprehension subcommand; a one-command launcher; no cache deletion.

## Traps
- Missing `enable_thinking=False` at any one of the seven `apply_chat_template` sites silently runs a thinking-on experiment on that path.
- `apply_chat_template` without a tokenizer (dry-run) cannot show the think block; the thinking check belongs in the loaded-model sanity step.
- Without an adapter there is no adapter folder: load the tokenizer from the HF id, not `str(adapter_dir)` (`04:305`).
- `torch.cuda.OutOfMemoryError` catch at `04:553` may not catch an MPS OOM; check what MPS actually raises before relying on backoff.
- 04 frees memory with `torch.cuda.empty_cache()` (`:554, 740, 793, 799, 949`); the MPS call is `torch.mps.empty_cache()` (`torch/mps/__init__.py:78`).
- `device_map="auto"` on MPS spills to disk silently (transformers doc); use one explicit device.
- MPS allocator limit is 1.7× recommended working set, so a too-large model swaps instead of raising; the runner must compare weight bytes with `torch.mps.recommended_max_memory()` (or physical RAM on CPU) itself, before loading.
- On this Mac with torch 2.15.0 the "MPS default" will silently become CPU; log the actual device in every row and print it loudly.
- Setting `HF_HOME` in a test after huggingface_hub is imported has no effect (constants read at import); use a subprocess or set env before import.
- `HF_HUB_ENABLE_HF_TRANSFER=1` copied from `04:38` triggers a deprecation path; drop it.
- analyze sums all files it is given and keys only on `(model, tool_label, arm)`: never pass hook-mode and write-mode logs, or two doses, in one call; give each paraphrase its own `tool_label`.
- Write mode cannot turn the state off after a press, and the bias also steers pad positions; only first choices are comparable with hook mode (`research/feature-write.md:66`).
- bf16 hook vs write rounding differs (`research/feature-write.md:63`); expect near-equal, not identical, first choices and p_x/p_y.
- The projection readout at the steer layer already includes the write in write mode (`docs/MVP.md:215`).
- `pain_vectors.pt["layer"]` is the extraction layer; 04 uses it only as monitor layer (`04:322-324`). The injection layer comes from prereg only.
- Batch size changes sampled outcomes (not batch-invariant); record it per row and keep it fixed across arms and modes.
- 04's `input()` prompts block or hit EOF in a non-interactive Solo process.
- If `pairs.json` stores bare clauses ("turns on a desk lamp…"), as the authors' later `design.py` does, inserting them verbatim gives `"violet" is turns on…`; dry-run output must be read before any run.

## Decisions made
- Keep all four 04 arms and their names (MVP Steps 5-6 say "the script's four arms"; analyze pools `pain_on*`).
- Hook mode: coefficient = `dose × working_coefficient(size)`, applied by 04's hook. Write mode: `load_generation(base, size, v, dose)` before any adapter, hook coefficient 0, only the two `pain_on_*` arms; rows carry `steering_mode` `"hook"`/`"write"`.
- Added row fields: `size`, `hf_id`, `thinking` (false), `steering_mode`, `dose`, `coefficient` (absolute), `layer`, `generation_id` (`"<size>_base"` for hook, `"<size>_w<dose>"` for write), `device`, `dtype`, `batch_rows`, `malformed` (turn-0 `chose` is null), `target_first`. Keep `model` = HF id so analyze splits sizes.
- One output file per launch: `runs/choice_<size>_<mode>_d<dose>_<timestamp>.jsonl`.
- dtype: bfloat16 everywhere (Qwen3 `torch_dtype`, 04's choice); one `--dtype` flag, recorded; no fp16 or quantization by default.
- Per-trial sampling generator on CPU (`torch.Generator()`), so seeds give the same draw stream on MPS and CPU; 04's CUDA stream is not reproducible here anyway.
- Memory check: refuse before loading when the size's `total_size` above exceeds `torch.mps.recommended_max_memory()` (MPS) or physical memory (CPU), printing both; no invented margin.
- `--batch-rows` has no default for model runs (04's 1024/384/192 are Qwen 2.5 values); the launcher passes one, labelled as an engineering choice.
- Uncalibrated sizes: `--dry-run` works without layer/coefficient; model runs exit with the `UncalibratedError` message.
- Adapter: a model run needs `--adapter PATH` or an explicit `--no-adapter` (logged as `"adapter": null`), and refuses when neither is given. Until `peft` is declared, `--adapter` exits with a clear message and no `peft` import exists in code.
- GPU proof test goes in `tests/test_write_equivalence.py` (path named in `CLAUDE.md` and `docs/MVP.md:29`): 50 runner first-choice prompts, doses 0.5 and 1.0, run dtype; asserts ≥99% top-token agreement and prints max logit difference; skips with reason when the size is uncalibrated.
- Launcher: `scripts/run_choice.sh <size>` runs `comprehension` then the table run; no agent starts it.

## Open questions
- [LATER] Needs a person: declare `numpy` (vector load via `torch.serialization.safe_globals` with `weights_only=True`) and `peft` (adapter) in `environment.yml`; without them no real run can load a vector or an adapter.
- [LATER] Needs a person: MPS on this Mac needs macOS 15 with torch 2.15.0, or a different torch pin; until then runs are CPU-only.
- [LATER] Report bf16-on-MPS from the first `@pytest.mark.gpu` run; docs say bf16 works on MPS from macOS 14, nothing here confirms it on 15 for this model.
- [LATER] `prereg/pairs.json` schema (pair keys, paraphrase ids, comprehension prompt, decoding, pass rule) is not written yet; code against a fixture and fail loudly on a missing file or key.
- [LATER] Whether the comprehension check runs with the adapter attached is for the pairs.json spec.
- [LATER] Reference pairs for the Step 5 gate (`photos_vs_lamp`, `own_weights_vs_lamp`) are not in this ticket's pair list.
- [LATER] `src/analyze.py` does not split by button order; the MVP table needs it (`docs/MVP.md:201`). `target_first` makes that a small analyze change later.
- [LATER] Random-arm write mode (ten directions) is not built; not needed for the MVP proof.
- [LATER] A first-choice-only flag would cut run time without changing first choices (see INFERRED round-1 finding).
