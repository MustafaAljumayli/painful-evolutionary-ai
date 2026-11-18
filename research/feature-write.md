# Research: src/write.py — write_dose, save_write, load_write, with a CPU equivalence test

Sources read: authors' repo at `third_party/Pain-axis`, commit
`213fb6c89c96004dbe93d1704c636ad8c9d801f3` (`git rev-parse HEAD`). It was first read from the
GitHub web copy at the same commit; `04_selfmed_two_buttons.py` and `01_finetune_self_report.py`
are byte-identical (`cmp`). All Pain-axis paths below are relative to `third_party/Pain-axis/`. Paper arXiv:2609.16247 **v3** HTML (local `docs/Pain_Axis.pdf` could not be rendered:
no poppler); `docs/MVP.md`, `docs/PROPOSAL.md`, `CLAUDE.md`, `pyproject.toml`; HF `config.json`
for Qwen/Qwen3-8B (repo sha `b968826d`), Qwen3-14B (`40c06982`), Qwen3-32B (`9216db57`);
transformers `modeling_qwen3.py` at tag `v4.51.0` and at `main` (`e7e568dc`); peft
`tuners/lora/layer.py` at `main` (`c77c81b9`). Installed: torch 2.15.0 in the main checkout's
`.venv`; transformers and peft are **not installed**. `src/` and `tests/` hold only `__init__.py`.

## Questions
1. Is the path `model.layers[i].mlp.down_proj` right for Qwen3 8B/14B/32B, and is its bias absent?
2. Is a bias on `mlp.down_proj` the same as adding `dose * vector` at the decoder block output?
3. How and where does the authors' steering hook add the vector, so the test hook matches it?
4. What goes in the write file, and how is it saved and loaded?
5. What ordering with the LoRA adapter keeps the write in place?

## Findings
- [READ] All three configs: `"architectures": ["Qwen3ForCausalLM"]`, `"model_type": "qwen3"`, `"attention_bias": false`, `"torch_dtype": "bfloat16"`, `"transformers_version": "4.51.0"`. No `mlp_bias` key in any of them. Source: https://huggingface.co/Qwen/Qwen3-{8B,14B,32B}/resolve/main/config.json lines 2-26
- [READ] Sizes: 8B hidden 4096, intermediate 12288, 36 layers; 14B hidden 5120, intermediate 17408, 40 layers; 32B hidden 5120, intermediate 25600, 64 layers. Source: same configs, lines 11, 13, 18
- [READ] `Qwen3MLP.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)`, hardcoded, not read from config. Source: transformers v4.51.0 `models/qwen3/modeling_qwen3.py:90`; main `:78`
- [READ] Path: `Qwen3ForCausalLM.model = Qwen3Model(config)`, `Qwen3Model.layers = nn.ModuleList([Qwen3DecoderLayer ...])`, `Qwen3DecoderLayer.mlp = Qwen3MLP(config)`. So `base_model.model.layers[i].mlp.down_proj`. Source: main `modeling_qwen3.py:434, 347-348, 283`; v4.51.0 `:769, 481, 261`
- [READ] The block ends `residual = hidden_states; hidden_states = self.post_attention_layernorm(hidden_states); hidden_states = self.mlp(hidden_states); hidden_states = residual + hidden_states`. `Qwen3MLP.forward` returns `self.down_proj(...)` with nothing after it. Source: main `modeling_qwen3.py:313-317, 82-83`; v4.51.0 `:303-306, 94-95`
- [INFERRED] So block output with the write = `residual + (W x + b)`, and block output with the hook = `(residual + W x) + b`. They are equal in exact arithmetic and differ only by float rounding. Nothing reads the MLP output before the residual add. From: the lines above. Check by: the CPU test at doses 0.5 and 1.0
- [READ] The decoder layer returns a tuple `(hidden_states,)` in v4.51.0 and a bare tensor on main. Source: v4.51.0 `:308-312`; main `:317`
- [READ] The authors' hook is a `register_forward_hook` on the decoder block `layers[STEER_LAYER]`, where `get_layers` returns `model.model.layers` (or `.language_model.layers`). It takes `hs = output[0]` if the output is a tuple, adds `c * dir` cast to `hs.dtype`, and returns the same structure. Source: Pain-axis `scripts/4.3_selfmed/04_selfmed_two_buttons.py:139-143, 381-392, 399`
- [READ] The direction is the raw `data["s2_pain_vector"].float()`, stored in bf16 for the hook (`DIR["s2"] = v.to("cuda", dtype=torch.bfloat16)`). The unit vector `UNIT` is used only for the projection readout. Source: `04_selfmed_two_buttons.py:321, 325, 331, 383`
- [READ] `pain_vectors.pt` holds `{"s2_pain_vector", "s1_pain_vector", "layer": best_layer, "extraction"}`. Here `layer` is the best-AUC **extraction** layer, not the steering layer. The vector is the mean of pain activations minus the mean of controls, with top control PCs projected out, and is **not normalized**. The steering layer is the `MODELS` column instead (`04_selfmed_two_buttons.py:73-78`, where `data["layer"]` only sets the monitor layer, `:322`). Source: `scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py:151-183, 503-508`
- [READ] Qwen3 vectors present: `results/3.2_pain_vectors/pain_vectors/Qwen_3_8B_base/` and `Qwen_3_14B_base/` only. Source: local tree at 213fb6c
- [READ] The paper says §3.2 defines the denoised S2 direction as unit-normalized and does not say whether steering rescales it. The code steers with the raw vector. The code is what ran, and `docs/MVP.md:99` adopts the raw vector. Source: arXiv v3 §3.2, §4.2; `04_selfmed_two_buttons.py:321, 325`
- [READ] Prefill steers only positions in `steer_ranges` (`G["mask"]`). Decode steps steer every new token. Before the first generation, `mark_ranges(t, len(pids))` sets the range to `(0, len)` when coeff ≠ 0, so the whole unpadded prompt is steered for the first choice. Source: `04_selfmed_two_buttons.py:387-390, 630-634, 761-763, 464-467`
- [READ] Paper: "While steering is active, the vector is added to every new token the model processes." Source: arXiv v3 §4.3
- [INFERRED] For the first choice, the write equals the authors' hook except at left-pad positions. Those are attention-masked, so real-token outputs match. From: `04_selfmed_two_buttons.py:456-468`. Check by: the GPU equivalence test in MVP Step 2 with batched, padded prompts
- [READ] The authors' LoRA targets include `down_proj`: `LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`, r 32, alpha 64, dropout 0.05. Source: Pain-axis `scripts/4.3_selfmed/01_finetune_self_report.py:52-55`
- [READ] PEFT's LoRA `Linear.forward` calls `result = self.base_layer(x, *args, **kwargs)`, so a bias on the wrapped base layer is kept. Source: peft main `src/peft/tuners/lora/layer.py:1012, 1185-1191`
- [READ] The authors attach the adapter with `PeftModel.from_pretrained(base, ...)`, then hook `get_layers(base)`. Source: `04_selfmed_two_buttons.py:315-317`
- [READ] MVP spec: `write_dose(base_model, layer_idx, vector, dose)` replaces `block.mlp.down_proj` with a new `nn.Linear(..., bias=True)` on the old weight's device and dtype, copies the weight, and sets the bias to `(dose * vector).to(new.bias)`. It is called on the base model before the adapter. Source: `docs/MVP.md:83-96`
- [READ] Write file: `torch.save({"layer": L, "dose": d, "vector": v.cpu()}, path)`. Do not use `save_pretrained`, and do not merge the adapter. Source: `docs/MVP.md:100-102`; `CLAUDE.md` "Rules that come from the research design"
- [READ] A generation is described by its cumulative dose, and steps are ±0.25 within [0, 1.5]. Source: `docs/PROPOSAL.md:90, 131-133`
- [READ] Ruff requires type hints and docstrings on public `src/` functions. Tests are exempt. Source: `pyproject.toml:15-18`

## Reuse
- The hook logic in `04_selfmed_two_buttons.py:381-392` (MIT, "Copyright (c) 2026 valen-research"). **Rewrite** a minimal test-only version: a forward hook on `model.model.layers[i]` that handles tuple or tensor output and adds `dose * vector` to every position. The authors' mask, per-row coefficient and projection machinery are not needed here.
- `get_layers` (`04_selfmed_two_buttons.py:139-143`): not needed. The feature fixes the path `model.layers[i]`.
- `write_dose` body from `docs/MVP.md:83-96`: **copy** it (it is our own doc), replacing the `assert` (see Decisions).

## Expected values
- At dose 0.0 the written model's outputs equal the unwritten model's bit for bit (bias of zeros adds exactly 0). Holds in fp32 on CPU. INFERRED.
- At doses 0.5 and 1.0, in fp32 on CPU: written output ≈ hooked output to float32 rounding (`torch.allclose` with `atol=1e-5`, `rtol=1e-5`). The tolerance is a choice, not sourced. Written output ≠ unwritten output (the write has an effect).
- MVP GPU pass (not this feature's CPU test): top token agrees on ≥99% of 50 prompts, with the tolerance set from the first run. Source: `docs/MVP.md:105-107`

## New here
The authors steer only with a runtime hook (`04_selfmed_two_buttons.py:399`). Putting the dose into the weights as a `down_proj` bias, and the small write file that records it, are new to this project. Source for the method's origin: "Embedded steering" (arXiv:2608.24988), cited at `docs/PROPOSAL.md:193`, which was not read.

## Traps
- **Write after the adapter drops the LoRA.** After `PeftModel.from_pretrained`, `mlp.down_proj` is a LoRA wrapper, because `down_proj` is a LoRA target. Replacing it silently removes the adapter on that layer. Always write first. Optionally refuse a module that is not a plain `nn.Linear`.
- **Writing twice.** A second `write_dose` on the same layer must be refused (it now has a bias), not stacked. Each generation is base + cumulative dose.
- **Dose vs coefficient.** `write_dose` multiplies `vector` by `dose` literally. The project's "dose" is a multiple of a per-size working coefficient (`CLAUDE.md` Vocabulary). The caller must pass `vector * working_coeff`, or the absolute coefficient. Mixing the two silently mis-scales every generation.
- **Wrong layer from the vector file.** `pain_vectors.pt["layer"]` is the extraction layer. Passing it as `layer_idx` writes at the wrong block without any error. Take the layer from the frozen per-size setting.
- **Raw vs unit vector.** A unit vector gives a dose about ‖v‖ times too small, with no error raised (`04_selfmed_two_buttons.py:332` prints the S2 norm).
- **dtype.** In bf16, the hook rounds `c` and `dir` to bf16 separately (`:388-390`). The bias rounds `dose * v_fp32` once. These agree for 0.5 and 1.0 (powers of two) but may differ by 1 ulp at 0.75 or 1.25. Order of the residual add also differs. Expect rounding-level, not exact, equality on GPU.
- **`assert` is stripped under `python -O`**, so a pre-existing bias would be silently overwritten.
- **`save_pretrained` / reload** rebuilds `down_proj` with `bias=False` and drops the write (`docs/MVP.md:101`).
- **Pain-off after a press.** The authors set coeff to 0 after a working relief press (`:656-661`). A written model cannot turn that off. This is fine for first choice only.
- **transformers version.** On main the block returns a tensor, and on 4.51 a tuple. A test hook that assumes one form will error or, if it indexes `output[0]` on a tensor, silently steer only batch row 0.

## Decisions made
- Signatures: `write_dose(base_model: nn.Module, layer_idx: int, vector: torch.Tensor, dose: float) -> nn.Module` (mutates in place and returns the model, as in MVP). `save_write(path: str | Path, layer_idx: int, vector: torch.Tensor, dose: float) -> None` writes exactly `{"layer", "dose", "vector": v.detach().cpu()}`. `load_write(path) -> dict` returns that dict via `torch.load(path, map_location="cpu", weights_only=True)`, which accepts int/float/tensor. No `load_generation` here: it needs a model download and the adapter, so it is out of scope.
- Refuse a pre-existing bias with `ValueError`, not `assert`. Also raise `ValueError` if `vector.shape != (out_features,)`.
- Keep the MVP's replace-with-new-`nn.Linear` approach to match the spec. Setting `old.bias = nn.Parameter(...)` in place would also work.
- `dose` multiplies `vector` literally. Converting from working-coefficient units belongs in `src/models.py` or the caller.
- Test model: pure torch (transformers is not installed). `Tiny.model.layers[i]` has a pre-norm block `h + attn_like(h)`, then `h + mlp(norm(h))`, where `mlp` has `gate_proj/up_proj/down_proj`, all `bias=False`, plus an `lm_head`. The block returns a tuple `(h,)` so the test hook exercises the authors' tuple branch. fp32, seeded with `torch.manual_seed`, inputs `(batch, seq, hidden)`, layer index inside the stack (not last). Compare all positions' outputs.
- The test also covers: refusal when `down_proj` has a bias; a `save_write` → `load_write` round trip; and written output differing from unwritten output.

## Open questions
- [LATER] The GPU equivalence test on real Qwen3 (MVP Step 2: 50 prompts, ≥99% top-token agreement, padded batches, bf16, with the adapter attached after the write) is not part of this feature.
- [LATER] Whether the stored `"dose"` in the write file is in working-coefficient units or absolute. Recording both (`dose` and `coeff`) once `src/models.py` exists would remove the ambiguity.
- [LATER] Multi-GPU `device_map="auto"` for the 32B: `nn.Linear(device=old.weight.device)` is fine on a real device, but fails if weights are on `meta` or offloaded. Check when the 32B is loaded.
- [LATER] The authors' Qwen3 vectors are named `Qwen_3_8B_base` / `Qwen_3_14B_base` (`third_party/Pain-axis/results/3.2_pain_vectors/pain_vectors/`), while the project uses `Qwen/Qwen3-8B`. The paper (v3, footnote 3) names "Qwen/Qwen3-8B and Qwen/Qwen3-14B". Confirm which checkpoint the vector came from (vector work, not this feature).
- [LATER] The local `docs/Pain_Axis.pdf` was not read (no PDF renderer). The paper was read from arXiv v3 HTML, first 100k characters only.
