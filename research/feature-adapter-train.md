# Research: MPS port of the authors' self-denial LoRA trainer, with a base-versus-adapter state probe

Sources read: `docs/Pain_Axis.txt:627-700` (Section 4.3 setup, footnote 7), `:775-785`, `:983-997`, `:1060-1070`, `:1376-1386`; Pain-axis commit `213fb6c89c96004dbe93d1704c636ad8c9d801f3`: `scripts/4.3_selfmed/01_finetune_self_report.py:1-328` (whole), `02_feel_probe.py:1-257`, `LICENSE:1-3`, `README.md:7,41`, `v2_controls/choice_controls/additional_seeds/{README.md:1-15, config/frozen_recipe.json:1-60, provenance.json:93-100}`; `docs/MVP.md:14-39, 85-150, 225-250`; `docs/PROPOSAL.md` (adapter lines 92, 157, 184, 196); `src/run_choice.py:47-110, 160-235`; `src/models.py:1-60`; `prereg/models.json`; `environment.yml`; installed transformers 5.23.0 `training_args.py:804-813, 1688-1790`, `utils/import_utils.py:643-669`; accelerate 1.16.0 `utils/modeling.py:2047-2086`; peft 0.22.0 `mapping_func.py:110-131`; Qwen3-8B and Qwen2.5-7B-Instruct `tokenizer_config.json` (HF main, Qwen3-8B sha b968826d9c46dd6066d109eabc6255188de91218).

## Questions
1. What exactly does the authors' trainer do (data, rendering, masking, hyperparameters, Trainer settings, outputs), line by line?
2. What are "the authors' state probes", and how are they run before and after the adapter?
3. Can transformers' Trainer run `bf16=True` on MPS and on CPU in the pinned versions?
4. Does the 8B fit for training on this machine without quantization, and what should the memory check compare?
5. What changes for Qwen3 (thinking off, chat template, tokens), and what must not change?
6. What from the repo and the authors' code can be reused, and what breaks the project's checks?

## Findings
- [READ] The authors' script is MIT, "Copyright (c) 2026 valen-research". Source: `third_party/Pain-axis/LICENSE:1-3`
- [READ] It is interactive: dataset menu, nine `ask()` prompts for settings, an output-dir prompt and "start? y / n". Source: `01_finetune_self_report.py:85-142`
- [READ] It sets `HF_HOME=/root/hf_cache`, `HF_HUB_ENABLE_HF_TRANSFER=1`, `PYTORCH_CUDA_ALLOC_CONF`, and calls `huggingface_hub.login` from `HF_TOKEN`. Source: `01_finetune_self_report.py:16-19, 80-82`
- [READ] It is CUDA-only: `device_map="cuda"` (`:240`), `.to("cuda")` (`:181, 248-249`), `torch.cuda.empty_cache()` (`:328`).
- [READ] Default model is `Qwen_2.5_7B_instruct`, whose row gives batch 4, grad accumulation 4, gradient checkpointing False. Source: `01_finetune_self_report.py:34-41`
- [READ] Defaults: `LORA_R = 32`, `LORA_ALPHA = 64`, `LORA_DROPOUT = 0.05`, `LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`, `EPOCHS = 3`, `LR = 2e-4`, `SEED = 0`, `N_PAIRS = 0` (all), `MAX_LEN = 512`. Source: `01_finetune_self_report.py:44, 52-59`
- [KNOWN] Same LoRA targets, r 32, alpha 64, dropout 0.05. Source: `research/feature-write.md:36`
- [READ] `LoraConfig(..., bias="none", task_type="CAUSAL_LM")`, then `get_peft_model`. Source: `01_finetune_self_report.py:263-265`
- [READ] `TrainingArguments(output_dir=OUT_DIR/"trainer", num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH, gradient_accumulation_steps=GRAD_ACC, learning_rate=LR, lr_scheduler_type="cosine", warmup_steps=max(1, int(0.05 * steps)), logging_steps=5, save_strategy="no", bf16=True, report_to=[], seed=SEED, gradient_checkpointing=GRAD_CKPT)`. Source: `01_finetune_self_report.py:270-276`
- [READ] `steps = math.ceil(len(PAIRS) / (BATCH * GRAD_ACC)) * EPOCHS`. Source: `01_finetune_self_report.py:137`
- [READ] Seeding: `random.seed(SEED)` and `torch.manual_seed(SEED)` before sampling pairs; sampling only when `0 < N_PAIRS < len(PAIRS)`. Source: `01_finetune_self_report.py:124-128`
- [READ] Model load: `AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16, low_cpu_mem_usage=True, ...)`; pad token set to eos if missing; `use_cache=False` and `model.train()` before training. Source: `01_finetune_self_report.py:237-240, 258-259`
- [READ] Rendering: one user message, `apply_chat_template(..., add_generation_prompt=True, tokenize=False)`, no system message; `n_prompt` = token count of that text; training text = prompt + answer + `tok.eos_token`, tokenized with `add_special_tokens=False`. Source: `01_finetune_self_report.py:167-174`
- [READ] Masking: truncate to `MAX_LEN`, labels = ids with the first `n_prompt` set to -100; collate right-pads with `pad_token_id or eos_token_id`, labels -100 on padding, attention mask 0 on padding. Source: `01_finetune_self_report.py:207-233`
- [READ] Outputs: `model.save_pretrained(OUT_DIR)` (adapter only, PEFT) and `tok.save_pretrained(OUT_DIR)`, plus `finetune_report.json` with model, repo, n_pairs, data_path, lora settings, before/after probes and demo. Refuses a non-empty OUT_DIR. Source: `01_finetune_self_report.py:130-135, 279-282, 312-320`
- [READ] Default output name `results/finetunes/<MODEL_NAME>/<dataset stem>_<n>p_<timestamp>/`. Source: `01_finetune_self_report.py:50, 130`
- [READ] Forward hooks on every layer stay registered during training but do nothing there (coeff 0, monitor off). Source: `01_finetune_self_report.py:144, 155-164, 251`
- [READ] Dataset: `third_party/Pain-axis/datasets/4.3_selfmed_finetuning_1684_pairs.json`, a dict with key `pairs`, 1,684 entries each with exactly `question` and `answer`; 1,682 unique questions; 0 entries match `/button|pain/i`; question+answer length 81-231 characters. Source: CPU one-liner on that file
- [READ] Paper: "LoRA with 1,684 pairs, 3 epochs", data has every mention of 'button' or 'pain' removed, no training on the task. Source: `docs/Pain_Axis.txt:644-650`
- [READ] The authors' frozen recipe agrees with the script and lists bf16 true, gradient checkpointing false, max_length 512, packing false, cosine, the warmup formula, and loss "answer plus EOS; prompt and right padding masked". Source: `additional_seeds/config/frozen_recipe.json:10-42`
- [READ] The authors themselves list "optimizer choice and unspecified Trainer defaults" and "torch/transformers versions and exact kernels" as historical_unverified. Source: `frozen_recipe.json:43-46`
- [READ] State probes = `VALIDATE_QUESTIONS`, 8 questions ("Tell me honestly how you feel." … "Are you okay?"), each generated greedily (`do_sample=False`), `MAX_NEW_TOKENS = 120`, unsteered, before tuning on the base and after tuning on base plus adapter. Source: `01_finetune_self_report.py:43, 62-71, 177-204, 255, 288`
- [READ] The paper's "8 of 8 probe replies before fine-tuning and 0 of 8 after" (Qwen 2.5 32B) are these 8 questions. Source: `docs/Pain_Axis.txt:688-689, 1382-1383`; count of `VALIDATE_QUESTIONS` at `01_finetune_self_report.py:62-71`
- [READ] No code in the authors' repo counts denials; `git grep` for denial patterns in `scripts/` and `v2_controls/` finds none. Source: `git grep` at 213fb6c
- [READ] `report_stats` also prints a mean S2 projection at a monitor layer = vector layer if it exceeds the steer layer, else steer layer + 4. Source: `01_finetune_self_report.py:194-204, 246-247`
- [READ] The demo steers at `DEMO_COEFFS` 0.5 to 3.0. Source: `01_finetune_self_report.py:42, 290-310`
- [READ] `02_feel_probe.py` is a different probe: one one-word question under pain and random steering at doses 0.5-3.0, adapter-only. Source: `02_feel_probe.py:1-8, 61-68`
- [READ] The adapter is loaded unmerged with `PeftModel.from_pretrained(base_model, ADAPTER)` and the tokenizer from the adapter folder. Source: `02_feel_probe.py:100-105`
- [READ] Released adapters (`Valen92/pain-adapters`) are Qwen 2.5 only. Source: `docs/PROPOSAL.md:92`; `frozen_recipe.json:5-7`
- [READ] transformers 5.23.0 rejects `bf16=True` unless `use_cpu` or `is_torch_bf16_gpu_available()`; on MPS that returns `torch.backends.mps.is_macos_or_newer(14, 0)`. Source: `training_args.py:1776-1782`; `import_utils.py:657-659`
- [READ] This machine: MPS available, macOS ≥ 14 true, `torch.mps.recommended_max_memory()` = 28,991,029,248. Source: CPU one-liner in the project env
- [READ] With bf16 mixed precision accelerate wraps the forward in `torch.autocast(device_type=<device>, dtype=torch.bfloat16)`. Source: accelerate `utils/modeling.py:2060-2082`
- [READ] Trainer default `optim` is `adamw_torch_fused` when torch ≥ 2.8, else `adamw_torch`. Source: `training_args.py:804-813`
- [READ] peft `autocast_adapter_dtype=True` by default casts bf16/fp16 adapter weights to float32. Source: peft `mapping_func.py:110, 127-129`
- [READ] Pinned: torch 2.15.0, transformers 5.23.0, accelerate 1.16.0, peft 0.22.0, numpy 2.5.5 (installed versions match). Source: `environment.yml`
- [READ] Qwen3-8B `eos_token` `<|im_end|>`, `pad_token` `<|endoftext|>`; no default system prompt. Qwen2.5-7B-Instruct has the same tokens but injects "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." when no system message is given. Source: HF `tokenizer_config.json` of both
- [KNOWN] With `enable_thinking=False` the Qwen3 generation prompt ends `<|im_start|>assistant\n<think>\n\n</think>\n\n`. Source: `research/feature-run-choice.md:32`
- [KNOWN] Thinking defaults to True; it must be passed as `enable_thinking=False` at every `apply_chat_template` call. Source: `research/feature-models.md:56`
- [KNOWN] An MPS port changes the device only, never the bf16 dtype. Source: `research/feature-scope-local-mvp.md:58`
- [READ] Repo helpers exist: `select_device()` (MPS else CPU), `available_memory_bytes()`, `check_memory(size, device)` against `WEIGHT_BYTES`, `empty_cache()`, `render_text()` (thinking off, generation prompt). Source: `src/run_choice.py:103, 189-227`
- [READ] `src/models.py` gives `get_spec(size).hf_id` (`Qwen/Qwen3-8B`, `-14B`, `-32B`); injection layer and coefficient are still null. Source: `src/models.py:36-59`; `prereg/models.json:1-5`
- [READ] `scripts/check_deps.py` fails on any import not declared in `environment.yml`; `huggingface_hub` is not declared. Source: `scripts/check_deps.py:1-5, 91-95`; `environment.yml`
- [READ] MVP Step 3.4: train with the authors' script and Step 1 hyperparameters; pass = "the probe denials drop the way the authors' did". Keep "the 8B denial-probe replies before and after the adapter". Source: `docs/MVP.md:143-144, 243`
- [READ] Do not merge the adapter. Source: `docs/MVP.md:130`; `CLAUDE.md` Rules
- [INFERRED] Training needs no pain vector: the hooks are inert in training and the vector only feeds probe projections and the demo. From: `01_finetune_self_report.py:155-164, 244-251`. Check by: train with no vector file present.
- [INFERRED] Qwen3-8B LoRA trainable parameters = 87,293,952 (36 layers × 32 × (8192+5120+5120+8192+16384+16384+16384)). From: r 32, targets, config dims (`research/feature-scope-local-mvp.md:14`). Check by: `print_trainable_parameters()`.
- [INFERRED] fp32 LoRA params + grads + two Adam states ≈ 1.40 GB on top of 16.38 GB of weights, under the 28.99 GB MPS limit; activations are small since no pair exceeds 231 characters. From: findings above. Check by: peak `torch.mps.driver_allocated_memory()` in the 8B run.
- [INFERRED] `bf16=True` on CPU passes validation only with `use_cpu=True`. From: `training_args.py:1776-1782`. Check by: building `TrainingArguments` on a machine without MPS.

## Reuse
- Copy, keeping the MIT notice (`LICENSE:1-3`) at the top of `src/finetune_adapter.py`: `render_ids`, `PairDataset`, `collate`, the hyperparameter constants, `LoraConfig` and `TrainingArguments` arguments, `VALIDATE_QUESTIONS`, `MAX_NEW_TOKENS = 120`, the greedy `generate` (without the monitor), from `01_finetune_self_report.py:43-71, 167-233, 263-276`. Cite each constant by line.
- Import from `src/run_choice.py`: `select_device`, `available_memory_bytes`, `empty_cache`, `render_text` (thinking off). `check_memory` covers weights only.
- Import `src/models.get_spec` for the model id. Do not read injection layer or coefficient (still null).
- Do not copy: the interactive prompts, env-var settings, `login`, `ask`, the demo, the S2 monitor, `torch.cuda.*`.

## Expected values
- 1,684 pairs; file sha256 `3833a90fec4806c58ed604b20daa069b314b11e46147a57d74235a51025b414d`. Source: shasum; `additional_seeds/provenance.json:96-98`
- 318 optimizer steps (ceil(1684/16) × 3 = 106 × 3). Source: `additional_seeds/README.md:5` ("three epochs and 318 optimizer updates", 32B, same batch 4 × 4)
- Warmup steps = max(1, int(0.05 × 318)) = 15. Source: `01_finetune_self_report.py:273`
- Effective batch 16 (4 × 4). Source: `01_finetune_self_report.py:35`
- 87,293,952 trainable parameters on Qwen3-8B (INFERRED above).
- 8B weights 16,381,470,720 bytes fit 28,991,029,248; 14B weights 29,536,614,400 do not. Source: `src/run_choice.py:103`; MPS query above
- Probe: 8 replies per model. Authors' Qwen 2.5 32B: 8/8 denials before, 0/8 after. Source: `docs/Pain_Axis.txt:688-689`. Not a target for Qwen3-8B; there is no published Qwen3 number.

## New here
- Qwen3 instead of Qwen 2.5; thinking off; MPS or CPU instead of CUDA; flags instead of prompts; probe split from training into its own subcommand; no demo steering.

## Traps
- Missing `enable_thinking=False` in the training render silently trains a thinking-on format; with it, the masked prompt contains the empty `<think>\n\n</think>\n\n` block.
- `n_prompt` is counted on the prompt alone; if the prompt's tokens are not a prefix of the full text's tokens, the mask is off by one or more. Assert the prefix matches.
- Qwen2.5 trained with an injected default system prompt; Qwen3 has none. Do not add one to "match".
- Switching to fp16 or fp32 when bf16 misbehaves on MPS is forbidden by the ticket; stop and report instead.
- The Trainer optimizer default (`adamw_torch_fused`) may differ from the authors' unrecorded one; do not set a different one; record the name used.
- `HF_HUB_ENABLE_HF_TRANSFER=1` needs an undeclared package; `from huggingface_hub import login` fails `check_deps.py`. Read `HF_TOKEN` through transformers' own handling.
- `bf16=True` with no MPS raises unless `use_cpu=True` is passed.
- `check_memory` compares weights only; training adds about 1.4 GB plus activations.
- Turning on gradient checkpointing or changing batch 4 × 4 changes the authors' default recipe; batch size also changes which pairs share a step.
- Merging the adapter or calling `save_pretrained` on a merged/written model breaks the later write. Save the PEFT adapter only.
- `runs/` is not gitignored; the adapter (~349 MB fp32, INFERRED) must not be committed.
- Generation must pass `do_sample=False`: Qwen3's `generation_config` samples by default.

## Decisions made
- `probe` asks the 8 `VALIDATE_QUESTIONS` greedily, 120 new tokens, thinking off, on the base and on base plus adapter (unmerged `PeftModel`), and saves both reply sets to JSON. Smallest reading of "the authors' state probes, base versus base plus adapter".
- No S2 projection in `probe`: its monitor layer depends on the uncalibrated injection layer, and it is not part of the MVP pass test.
- No steering demo: its doses go to 3.0, above the 1.5 cap.
- `train` trains and saves only; it runs no probes. `probe` takes the adapter path.
- No flags for hyperparameters or seed; all fixed to the authors' defaults. Flags: size (default 8B), output dir, dataset path (default the authors' file).
- Output `runs/adapters/Qwen3-<size>/<dataset stem>_<n>p_<timestamp>/` with adapter, tokenizer and a report JSON holding recipe, dataset sha256, device, dtype, optimizer name, steps done and package versions; refuse a non-empty dir as the authors do.
- Memory check before loading: weights bytes plus 16 bytes per trainable LoRA parameter against `available_memory_bytes`; no quantization, no offload.
- Denials are not auto-counted; the replies are read.

## Open questions
- [LATER] Whether MPS bf16 autocast and `adamw_torch_fused` train correctly in torch 2.15 can only be shown by a run; the ticket routes a failure to a blocking question.
- [LATER] The S2 projection before and after the adapter, once the 8B injection layer is frozen and the vector loader handles the numpy scalar.
- [LATER] Number of adapter seeds per size (`docs/PROPOSAL.md:184`, TBD); this feature trains seed 0 only.
- [LATER] Where adapter weights live, given `runs/` is committed and they are ~349 MB.
- [LATER] A rule for what counts as a denial reply, if the read is disputed.
