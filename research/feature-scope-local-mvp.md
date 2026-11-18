# Research: record the local-hardware scope deviation (M3 Max, 36 GB, MPS, no cloud) in the docs

Sources read: `docs/Pain_Axis.txt:215-254` (Table 1, footnote 3), `docs/Pain_Axis.txt:415-424` (footnote 5); grep of the paper for GPU/H100/A100/quantiz/bf16/compute (no hardware statement found). Authors' repo commit `213fb6c89c96004dbe93d1704c636ad8c9d801f3`, `scripts/4.3_selfmed/04_selfmed_two_buttons.py:70-78, 303-331`. `git diff` of `CLAUDE.md`, `AGENTS.md`, `docs/MVP.md`, `docs/PROPOSAL.md` against HEAD `7cf89cb`; `docs/MVP.md:1-40, 106-148`; `docs/PROPOSAL.md:15-18, 71-94, 205-215`; `CLAUDE.md:30-90`. Hugging Face `config.json` and `model.safetensors.index.json` for Qwen3-8B/14B/32B and Qwen2.5-7B-Instruct. PyTorch 2.15 MPS env-var docs. `environment.yml:14-23`.

## Questions
1. Do the parameter counts and bf16 sizes in the edits match each model's `config.json`?
2. Does the diff change any sourced number, any button wording, or anything in `prereg/`, `src/`, `tests/`, `third_party/`?
3. Are 8B = MVP, 14B = stretch, 32B = deferred, RQ6 deferred (not failed), and never-quantize-silently each recorded where the request asks?
4. Are `CLAUDE.md` and `AGENTS.md` still identical?
5. What does the hardware actually allow (MPS memory ceiling), and does the 14B claim hold?
6. Are there internal inconsistencies left in the edited text?

## Findings
- [READ] Qwen3-8B config: hidden 4096, intermediate 12288, layers 36, heads 32, KV heads 8, head_dim 128, vocab 151936, `tie_word_embeddings` false, `torch_dtype` bfloat16, `attention_bias` false. Source: https://huggingface.co/Qwen/Qwen3-8B/resolve/main/config.json
- [READ] Qwen3-14B config: hidden 5120, intermediate 17408, layers 40, heads 40, KV heads 8, head_dim 128, vocab 151936, untied. Source: https://huggingface.co/Qwen/Qwen3-14B/resolve/main/config.json
- [READ] Qwen3-32B config: hidden 5120, intermediate 25600, layers 64, heads 64, KV heads 8, head_dim 128, vocab 151936, untied. Source: https://huggingface.co/Qwen/Qwen3-32B/resolve/main/config.json
- [READ] Recomputed with the doc's formula (L·(h·nh·hd + 2·h·kv·hd + nh·hd·h + 3·h·i) + 2·v·h): 8B 8.1904B / 16.38 GB / 15.26 GiB; 14B 14.7679B / 29.54 GB / 27.51 GiB; 32B 32.7614B / 65.52 GB / 61.02 GiB. All match `docs/MVP.md:20-22` and `docs/PROPOSAL.md:82-84`. Source: configs above, CPU one-liner.
- [READ] Independent check, safetensors `metadata.total_size`: 8B 16,381,470,720 B; 14B 29,536,614,400 B; 32B 65,524,246,528 B (= 8.1907B, 14.7683B, 32.7621B params at 2 B each). The doc formula omits only RMSNorm weights (incl. q_norm/k_norm), < 0.001B. Source: https://huggingface.co/Qwen/Qwen3-{8B,14B,32B}/resolve/main/model.safetensors.index.json
- [READ] Qwen3-8B model card states "Number of Parameters: 8.2B". Source: https://huggingface.co/Qwen/Qwen3-8B/raw/main/README.md line 30
- [READ] Qwen2.5-7B-Instruct safetensors total 15,231,233,024 B (15.23 GB, 14.19 GiB), smaller than the 8B, so `docs/MVP.md:32` ("fits locally") holds for weights. Source: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/model.safetensors.index.json
- [READ] This machine: `hw.memsize` 38,654,705,664 B (36 GiB), `Apple M3 Max`; torch 2.15.0 `torch.backends.mps.is_available()` True; `torch.mps.recommended_max_memory()` = 28.99 GB. Source: `sysctl`, env python one-liner.
- [READ] PyTorch 2.15 MPS allocator: `PYTORCH_MPS_HIGH_WATERMARK_RATIO` default 1.7, `PYTORCH_MPS_LOW_WATERMARK_RATIO` default 1.4 (unified). Source: https://docs.pytorch.org/docs/2.15/mps_environment_variables.html
- [INFERRED] 14B bf16 weights (29.54 GB) exceed the Metal recommended working set (28.99 GB) before any KV cache; the 1.7 high watermark lets allocation proceed, so it will page rather than error. The doc's "Tight ... likely swap" (`docs/MVP.md:21`) is correct and conservative. From: two findings above. Check by: loading Qwen3-14B on MPS and watching `torch.mps.driver_allocated_memory()` and swap.
- [INFERRED] 8B leaves about 12.6 GB of the 28.99 GB working set for activations, KV cache and adapter. From: 28.99 − 16.38. Check by: a Step 3 run on MPS.
- [READ] Paper gives Qwen3 vectors for 8B and 14B only (Table 1 row "Qwen 3 / 8B, 14B / instruct") and names `Qwen/Qwen3-8B` and `Qwen/Qwen3-14B` as post-trained (footnote 3). The paper states no hardware or precision. Source: `docs/Pain_Axis.txt:228-230, 244`
- [READ] Authors' button script loads in `torch.bfloat16`, `device_map="cuda"`, sets `allow_tf32`, and moves vectors to `"cuda"`; batch rows 1024 (7B), 384 (32B), 192 (72B). Source: `third_party/Pain-axis/scripts/4.3_selfmed/04_selfmed_two_buttons.py:73-77, 303-331`
- [READ] Diff touches only `CLAUDE.md`, `AGENTS.md`, `docs/MVP.md`, `docs/PROPOSAL.md` (49+/14−); `git status` shows nothing under `prereg/`, `src/`, `tests/`, `third_party/`. Source: `git diff --stat`, `git status --porcelain`
- [READ] No button description and no quoted pair string appears in any changed hunk. Source: `git diff`
- [READ] Sourced numbers preserved verbatim: 27.0% photos baseline (`docs/PROPOSAL.md:91`), authors' batch 384 and the 80 GB card (`docs/MVP.md:36`), AUC 0.93 to 1.00 (`docs/MVP.md:142`, unchanged), Qwen 2.5 layers 16/38 and coefficient 1.0 (unchanged). Source: `git diff`
- [READ] Removed numbers: "fits on a 24 GB card" (8B and Qwen 2.5 7B) and "needs a 48 GB card" (14B), from HEAD `docs/MVP.md:14`. These were GPU-sizing statements with no paper or log source; their removal changes no result. Source: `git show HEAD:docs/MVP.md`
- [READ] `cmp CLAUDE.md AGENTS.md` reports identical. Source: `cmp`
- [READ] 8B = MVP, 14B = stretch, 32B = deferred is stated in `CLAUDE.md:37-39, 85-86`, `docs/MVP.md:5, 12-32, 139`, `docs/PROPOSAL.md:18, 78-87, 91, 213`.
- [READ] "Stated deferral, not a failed result" for RQ6 is in `docs/MVP.md:31` and `docs/PROPOSAL.md:18, 86`; `CLAUDE.md:39` says only "RQ6 (scale) is deferred with it."
- [READ] "Never quantize silently" is in `docs/MVP.md:30` and `docs/PROPOSAL.md:85`, but not in `CLAUDE.md` (no "quantiz" match). Source: `grep -n quantiz CLAUDE.md`
- [READ] `docs/MVP.md:36` says "See Scope deviation below" but the section is above it, at `docs/MVP.md:12`. Source: `docs/MVP.md:12, 36`
- [READ] `docs/MVP.md:27` gates the 14B on "every part of Step 2"; Step 2 (`docs/MVP.md:106-135`) has no numbered parts, while Step 3 has "five parts" and its own 8B-before-14B rule (`docs/MVP.md:139`). Source: those lines
- [READ] `docs/PROPOSAL.md:71-73` (RQ6, H6 "present at all three sizes") and `:132, :136` ("at all three sizes") are unchanged; the blanket note at `docs/PROPOSAL.md:87` covers them, as the lead directed. Source: `docs/PROPOSAL.md`
- [READ] `CLAUDE.md:86` is a ~110-character line where the rest of the file wraps near 76. Source: `CLAUDE.md:85-87`
- [READ] Thinking off: Qwen3 model card documents `enable_thinking=False` in `apply_chat_template`; default is True. Source: https://huggingface.co/Qwen/Qwen3-8B/raw/main/README.md lines 70, 132-141

## Reuse
none (docs-only feature; no code from the authors is taken).

## Expected values
- Qwen3-8B: 8.19B params, 16.4 GB (15.3 GiB) bf16. Source: config.json and safetensors index above.
- Qwen3-14B: 14.77B params, 29.5 GB (27.5 GiB) bf16. Source: same.
- Qwen3-32B: 32.76B params, 65.5 GB (61.0 GiB) bf16. Source: same.
- Metal recommended max working set on this M3 Max 36 GB: 28.99 GB. Source: `torch.mps.recommended_max_memory()` under torch 2.15.0.
- `cmp CLAUDE.md AGENTS.md` exits 0. Source: `scripts/check.sh:12-15`.
- `git diff --stat` lists only the four doc files and the new brief/plan/report files. Source: feature request.

## New here
- The prior work ran on CUDA in bf16 with no memory constraint stated; this project runs on MPS with a 36 GB unified pool, drops the 32B for now and defers RQ6. No Qwen3 size is run by the paper's button task in either case.

## Traps
- GB vs GiB: the docs quote both (16.4 GB = 15.3 GiB). Metal's 28.99 figure is in GB (10^9); compare like with like.
- The 14B fits the 36 GiB total but not the 28.99 GB Metal working set; "fits in 36 GB" and "runs without paging" are different claims.
- The authors' script hardcodes `"cuda"` and TF32 (`04_selfmed_two_buttons.py:303-331`); a later MPS port must change device only, never the bf16 dtype, or it becomes an unnamed condition change.
- Lowering batch rows from the authors' 1024/384 to fit MPS changes throughput, not the sampling distribution; do not present it as a setting change (temperature 0.7, top-p 0.95, 101 scenarios stay).
- Editing `CLAUDE.md` without `cp CLAUDE.md AGENTS.md` fails `scripts/check.sh`.
- Do not touch `prereg/` or button strings in this feature; the diff currently does not.

## Decisions made
- Treat the edits as correct in substance; the arithmetic is verified to the stated precision by two independent sources.
- Fix `docs/MVP.md:36` "See Scope deviation below" to "above"; it is a pointer error, not content.
- Fix `docs/MVP.md:27` to gate the 14B on Steps 2 and 3 passing on the 8B, matching `docs/MVP.md:133` (8B first for the equivalence test) and `:139` (all five parts of Step 3 on the 8B first).
- Add "never quantize silently; a quantized run is a named, preregistered deviation" and "RQ6 is a stated deferral, not a failed result" to `CLAUDE.md`, then copy to `AGENTS.md`, because the request lists both for every file and `CLAUDE.md` is what every agent reads.
- Optionally rewrap `CLAUDE.md:86` to the file's width; cosmetic.
- Leave `docs/PROPOSAL.md` Experiments 1-3 and RQ6/H6 wording as is, per the lead; the note at `docs/PROPOSAL.md:87` covers them.
- Optionally add the measured 28.99 GB working set to the 14B row of `docs/MVP.md:21`, with its source (`torch.mps.recommended_max_memory()`, torch 2.15.0).

## Open questions
- [LATER] The decision (user, 2026-11-17) is recorded only in these docs; no separate decision log exists.
- [LATER] `@pytest.mark.gpu` and `CLAUDE.md` "Environment" say nothing about MPS; whether "gpu" tests mean CUDA or MPS should be settled before the first real-model test runs.
- [LATER] bf16 op coverage on MPS under torch 2.15.0 is unverified for Qwen3 generation; `PYTORCH_ENABLE_MPS_FALLBACK=1` would silently move ops to CPU and should be logged if used.
- [LATER] Whether MPS bf16 numerics change the equivalence-test tolerance in `docs/MVP.md:135` is unknown until the first run.
