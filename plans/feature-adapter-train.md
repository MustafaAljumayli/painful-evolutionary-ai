# Plan: MPS trainer for the self-denial adapter, with a base-versus-adapter probe subcommand

Brief: research/feature-adapter-train.md

## Shared names
- `src/finetune_adapter.py`: command line with subcommands `train` and `probe`; no other file imports it.
- Adapter output folder: `runs/adapters/Qwen3-<size>/<dataset stem>_<n>p_<timestamp>/` (PEFT adapter, tokenizer, report JSON).
- Probe output: a JSON file holding the 8 base replies and the 8 base-plus-adapter replies.

## Subtasks

### S1: Trainer and probe in `src/finetune_adapter.py`
- Owns: src/finetune_adapter.py, tests/test_finetune_adapter.py
- Depends on: none
- Goal: A flag-driven, non-interactive port of the authors' self-denial trainer that runs on MPS else CPU, trains Qwen3 (8B default) in bf16 on the authors' 1,684 pairs with their default hyperparameters, thinking off, and saves only the unmerged PEFT adapter under `runs/adapters/`. A `probe` subcommand asks the authors' 8 state questions of the base model and of base plus the adapter and saves both reply sets.
- Read: Findings (all), Reuse, Expected values, Traps, Decisions made
- Done when:
  - The MIT notice is at the top of the file, and every copied constant or setting cites the authors' file and line.
  - Hyperparameters, the pair count (1,684), the dataset hash, 318 optimizer steps and 15 warmup steps match Expected values, and no flag changes them.
  - The training text renders with thinking off, the prompt tokens are checked to be a prefix of the full text, and only the answer plus EOS carries labels.
  - Training needs no pain vector file (INFERRED check), and a bf16 failure on MPS or CPU stops with a clear error and never switches dtype.
  - A memory check runs before the model loads and refuses clearly if training will not fit; no quantization, no offload, no merge.
  - The trainable parameter count on Qwen3-8B is 87,293,952 (INFERRED check), and `bf16=True` on CPU builds valid training arguments only with `use_cpu=True` (INFERRED check).
  - Probe replies are greedy, 120 new tokens, thinking off, from an unmerged adapter; an existing non-empty output folder is refused.
  - `./scripts/check.sh` exits 0, with GPU or model-download tests marked `gpu`.
