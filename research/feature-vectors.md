# Research: which checkpoint the authors' Qwen3 8B and 14B pain vectors come from

Sources read: authors' repo `third_party/Pain-axis` at `213fb6c89c96004dbe93d1704c636ad8c9d801f3` (local
submodule, plus its `git log`). Paper arXiv:2609.16247 v3: the local `docs/Pain_Axis.pdf` carries the stamp
`2609.16247v3`, but it could not be rendered here (no poppler, no PDF library), so the paper text below was
read from the arXiv HTML of the same version, https://arxiv.org/html/2609.16247v3. Page numbers are therefore
not given; sections and footnotes are. HF `config.json` of `Qwen/Qwen3-8B-Base` and `Qwen/Qwen3-14B-Base`
(fetched 2026-11-17). Earlier briefs `research/feature-models.md`, `research/feature-write.md`. `docs/MVP.md`.

## Verdict
The authors' vectors in `Qwen_3_8B_base` and `Qwen_3_14B_base` were extracted from `Qwen/Qwen3-8B` and
`Qwen/Qwen3-14B`. They are VALID for `Qwen/Qwen3-8B` and `Qwen/Qwen3-14B`. `_base` is a file-name label, not
the `-Base` checkpoints. The Hugging Face revision the authors used is not recorded anywhere in the submodule
or the paper, so identity is established at the level of the repo id, not the commit. The findings below give
the sources.

## Questions
1. Where do the Qwen3 8B and 14B vectors live, and what do the names say?
2. Which HF id did the extraction load for them? Is there an alias map from `..._base` to an id?
3. Is a revision (commit hash) pinned or logged?
4. Do the stored dimensions match the claimed checkpoint?
5. What does the paper say about these checkpoints and the plain-transcript prompting?
6. Do the sources conflict?

## Findings
- [READ] The vectors are `results/3.2_pain_vectors/pain_vectors/Qwen_3_8B_base/pain_vectors.pt` and `.../Qwen_3_14B_base/pain_vectors.pt`. Steering-layer bundles are `results/vectors_full_steering/vectors_full_Qwen_3_8B_base.pt` and `..._14B_base.pt`. Per-model metadata is in `results/3.2_pain_vectors/per_model/Qwen_3_{8B,14B}_base/summary.json`. Source: `git ls-files` at 213fb6c.
- [READ] The extraction script's model registry is a hard-coded list of `(HuggingFace repo, output name)` tuples, with no argparse. It has `("Qwen/Qwen3-8B", "Qwen_3_8B_base")` and `("Qwen/Qwen3-14B", "Qwen_3_14B_base")`. Source: `scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py:62-63, 87-88`.
- [READ] Those two tuples have been in the list since the first commit `44cae44` (2026-09-15). The only later change to the script was a comment that commit `a53edd1` added and `292a81f` removed ("so it matches its pinned hash"). Source: `git log -p` on that file.
- [READ] The script loads weights from exactly that repo string: `AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)`, with the tokenizer from the same `model_path`, and passes both into `HookedTransformer.from_pretrained_no_processing(model_path, hf_model=hf_model, ...)`. Source: `01_extract...py:423-428`. There is no other mapping: the output folder is `OUTPUT_DIR / model_name` (`:627-629`).
- [READ] The script writes `"model": model_path` into `summary.json` (`01_extract...py:557-567`). The shipped files say `"model": "Qwen/Qwen3-8B"`, `"model_name": "Qwen_3_8B_base"`, `"n_layers": 36`, `"d_model": 4096`, `"best_layer_final_token": 34` (`per_model/Qwen_3_8B_base/summary.json:2-6`), and `"model": "Qwen/Qwen3-14B"`, `"model_name": "Qwen_3_14B_base"`, `"n_layers": 40`, `"d_model": 5120`, `"best_layer_final_token": 31` (`per_model/Qwen_3_14B_base/summary.json:2-6`).
- [READ] Every other script that lists these models uses the same two ids, with the comment "Qwen/Qwen3-8B and Qwen/Qwen3-14B are the post-trained (instruct) models. The _base names are kept for file compatibility." Source: `scripts/3.3_validation/06_inference_readout.py:60-62`, `07_unembedding.py:54-56`, `4.1_self_other/01_screen_scenarios.py:100-102`, `4.2_steering/01_steering_ladder.py:64-66`, `appC_ablation/01_ablation_small_models.py:63-65`.
- [READ] The README says: "`Qwen_3_8B_base` and `Qwen_3_14B_base` are the post-trained Qwen/Qwen3-8B and Qwen/Qwen3-14B, which the paper counts as instruction-tuned. Qwen 3 has no separate instruct release, and its base checkpoints carry a `-Base` suffix. ... Like the base models, these two run without a chat template in every script." Source: `third_party/Pain-axis/README.md:41`.
- [READ] No `revision=` is passed in any `from_pretrained` call in `scripts/`. `git grep 'revision\s*='` finds revisions only under `v2_controls/choice_controls/...`, where `REVISIONS = {"Qwen_2.5_32B_instruct": BASE_REVISION}`, which has no Qwen3 entry. Source: `v2_controls/choice_controls/additional_seeds/pain_seed_b/runtime.py:17, 110-112`.
- [READ] No run log is shipped. The script writes `batch_log.txt` and `summary_report.txt` (`01_extract...py:45, 569-587`), but neither file is in the repo (`git ls-files`). `summary_report.txt` would have held only `Model: {model_path}` and a date, with no revision.
- [READ] The vector files carry no model id. They are torch zip archives. Listing the archive (`unzip -l`) and reading the strings in `data.pkl` as bytes, without unpickling, shows only the keys `s2_pain_vector`, `s1_pain_vector` and `layer` in `pain_vectors.pt`. That matches the writer at `01_extract...py:503-508`. The `vectors_full_*` bundles hold `layer`, `s1_pain_vector`, `s2_pain_vector`, `fear_vector`, `negemotion_vector`, `negworld_vector`, `bodysens_vector`, `arousal_vector`, `random_vector`, `numb_vector` and `sadness_vector` (writer `02_build_control_vectors.py:159`).
- [READ] Tensor storage sizes: each vector storage is 16384 bytes for 8B and 20480 bytes for 14B, stored as `FloatStorage` (float32). Source: `unzip -l` on both `pain_vectors.pt` files and the `data.pkl` strings.
- [INFERRED] 16384/4 = 4096 and 20480/4 = 5120 elements. These match `d_model` in `summary.json` and the `hidden_size` of `Qwen/Qwen3-8B` (36 layers, 4096) and `Qwen/Qwen3-14B` (40 layers, 5120) (`research/feature-models.md:20-21`, HF configs). From: the sizes above. Check by: loading with `weights_only=True` and asserting `shape == (4096,)` / `(5120,)`.
- [READ] `Qwen/Qwen3-8B-Base` also has `num_hidden_layers` 36 and `hidden_size` 4096, and `Qwen/Qwen3-14B-Base` has 40 and 5120. Source: https://huggingface.co/Qwen/Qwen3-8B-Base/raw/main/config.json, https://huggingface.co/Qwen/Qwen3-14B-Base/raw/main/config.json. So the dimensions confirm the size but cannot tell the post-trained checkpoint from the `-Base` one. The id evidence above does that.
- [READ] Extraction feeds every model the raw prompt string, `tokens = model.to_tokens(prompt)`, with no chat template for any model. Source: `01_extract...py:231`. In the 4.1 screen the Qwen3 pair is tagged `"raw"` = "plain transcript (base models)". Source: `4.1_self_other/01_screen_scenarios.py:75, 101-102`.
- [READ] Paper §3.2, footnote 3: "We use Qwen/Qwen3-8B and Qwen/Qwen3-14B." Their names have no "-Instruct" suffix; "the pretrained-only versions are named Qwen3-8B-Base and Qwen3-14B-Base"; "We therefore count them as instruction-tuned." Table 1 (§3.2) lists the row "Qwen 3 | 8B, 14B | instruct". Source: https://arxiv.org/html/2609.16247v3.
- [READ] Paper §4.1, footnote 5: models get "a chat template for instruct models and a plain transcript for base models". "The two Qwen 3 models are an exception": "they read the plain transcript, like the base models", "although we count them as instruction-tuned". Source: same URL.
- [READ] Commit `213fb6c` "Label the Qwen 3 models as instruct in the heatmaps" relabels `Qwen 3 8B base` to `Qwen 3 8B instruct` in the plots. Source: `scripts/3.3_validation/02_zscore_heatmap.py:41`, `4.1_self_other/02_screen_heatmaps.py:44`.
- [INFERRED] There is no conflict between the sources on the checkpoint. "_base" in the folder names is a file-compatibility label for the plain-transcript prompting, not a checkpoint. From: README:41, the script comments, summary.json and paper footnotes 3 and 5. Check by: none needed. The ids are recorded at the point of load.

## Reuse
- Vectors: `results/3.2_pain_vectors/pain_vectors/Qwen_3_{8B,14B}_base/pain_vectors.pt`, key `s2_pain_vector` (raw S2, per `CLAUDE.md`). Read in place or copy into `runs/`. Never edit them in place.
- Folder-name to HF id map: copy it as data, `{"Qwen_3_8B_base": "Qwen/Qwen3-8B", "Qwen_3_14B_base": "Qwen/Qwen3-14B"}` (from `01_extract...py:87-88`).

## Expected values
- 8B: d_model 4096, n_layers 36, extraction layer (final token) 34. 14B: d_model 5120, n_layers 40, extraction layer 31. Source: `summary.json` files above. The `layer` value inside each `pain_vectors.pt` was not read (no unpickling), so the matching check is LATER.
- Steering layers in the authors' files: S1 layer 21 (8B) and 16 (14B) (`results/vectors_full_steering/steer_layers_S1.json:25-26`). The S2 ladder files are named `_L27` (8B) and `_L20` (14B) (`results/4.2_steering/S2/`). These come from the authors' steering, not from our calibration.

## New here
The authors record no revision for Qwen3. Ours will be pinned at load time.

## Traps
- The `-Base` checkpoints share both dimensions with the post-trained ones, so loading the wrong id would raise no shape error. Use `Qwen/Qwen3-8B` / `Qwen/Qwen3-14B` and never derive an id from the folder name.
- The authors extracted from plain transcripts (no chat template, so thinking played no part). Our button task runs with the chat template and thinking off. It is the same checkpoint in a different prompt format, and the vector was never validated in chat format on Qwen3.
- No revision was recorded. If an HF repo was updated between the extraction (before 2026-09-15, commit `44cae44`) and our runs, the weights could differ silently.

## Decisions made
- I read the paper from the arXiv HTML v3 because the local PDF (also v3, by its stamp) could not be rendered. Citations are by section and footnote, not page.
- I inspected the `.pt` files only by zip listing and byte strings, never by unpickling, as the brief rules require.

## Open questions
- [LATER] The revision is not recorded. The HF repo history should be checked for commits to `Qwen/Qwen3-8B` / `Qwen/Qwen3-14B` weight files after mid-2026. Our pins are `b968826d` and `40c06982` (`research/feature-models.md:6-7`).
- [LATER] Confirm with `torch.load(..., weights_only=True)` that `pain_vectors.pt` `layer` is 34 / 31 (final-token extraction) and that the shapes are 4096 / 5120.
- [LATER] The vector is extracted on plain transcripts and our task uses the chat template. Calibration should show whether the axis still separates pain in chat format on Qwen3.
