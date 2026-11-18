# Plan: src/write.py with write_dose, save_write and load_write, proven on CPU against a steering hook

Serves MVP output "one proof" (a dose written into the weights behaves the same as
steering at that dose), pipeline stage "vector + dose -> write (weights)". This feature
is the CPU part only. The GPU proof on real Qwen3 (`tests/test_write_equivalence.py`),
`load_generation`, and device_map/offload handling for 32B are out of scope.

Facts the plan relies on (from `research/feature-write.md`, all READ unless marked):

- Qwen3 8B, 14B, 32B: the module path is `base_model.model.layers[i].mlp.down_proj`.
  `Qwen3MLP.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)` is hardcoded.
  None of the three configs has an `mlp_bias` key. (brief Findings lines 21-24: HF config.json
  for each size; transformers `modeling_qwen3.py` v4.51.0 `:90`, main `:78`)
- The decoder block ends `hidden_states = residual + self.mlp(post_attention_layernorm(h))`, and
  `Qwen3MLP.forward` returns `down_proj(...)` with nothing after it. (brief line 25)
- The decoder layer returns the tuple `(hidden_states,)` in transformers v4.51.0 and a bare tensor
  on main. (brief line 27)
- The authors' hook is a `register_forward_hook` on the decoder block. It takes `output[0]` if the
  output is a tuple, adds `c * dir` cast to `hs.dtype`, and returns the same structure.
  (brief line 28: `third_party/Pain-axis/scripts/4.3_selfmed/04_selfmed_two_buttons.py:381-392`)
- [INFERRED, brief line 26] A bias `b` on `down_proj` gives `residual + (W x + b)`, and the hook gives
  `(residual + W x) + b`. These are equal up to float rounding. The brief checks this with the CPU
  test at doses 0.5 and 1.0, which is S2's "Done when".
- The write file format is `{"layer": L, "dose": d, "vector": v.cpu()}` saved with `torch.save`
  (`docs/MVP.md:100`).

## Interfaces

### `tests/tiny_qwen3.py` (test helper, owned by S1, imported by S2)

Pure torch, because transformers is not installed (brief line 11). It mirrors the Qwen3 module
layout. It is not collected by pytest because the filename does not start with `test_`.
Import it as `from tests.tiny_qwen3 import ...` (`pyproject.toml` sets `pythonpath = ["."]`, and
`tests/__init__.py` exists).

```python
HIDDEN: int = 16        # test fixture sizes, arbitrary (not from any source; tiny model only)
INTERMEDIATE: int = 32
N_LAYERS: int = 3
VOCAB: int = 50

class TinyMLP(nn.Module):        # attrs gate_proj, up_proj, down_proj: all nn.Linear(bias=False)
    # forward(x) = down_proj(silu(gate_proj(x)) * up_proj(x))   (Qwen3MLP shape)
class TinyBlock(nn.Module):      # attrs input_layernorm, self_attn (stand-in: nn.Linear(H, H, bias=False)),
                                 # post_attention_layernorm, mlp: TinyMLP
    # __init__(..., return_tuple: bool = True)
    # forward(h): h = h + self_attn(input_layernorm(h)); h = h + mlp(post_attention_layernorm(h))
    #             return (h,) if return_tuple else h
class TinyModel(nn.Module):      # attrs layers: nn.ModuleList[TinyBlock], norm
class TinyForCausalLM(nn.Module):# attrs model: TinyModel, lm_head: nn.Linear(H, VOCAB, bias=False)
    # forward(hidden: Tensor[batch, seq, HIDDEN]) -> logits Tensor[batch, seq, VOCAB]
    # It runs each block, takes out[0] if out is a tuple, then norm, then lm_head.

def make_tiny_model(seed: int = 0, return_tuple: bool = True) -> TinyForCausalLM:
    """Seeded (torch.manual_seed(seed)), float32, eval mode, on CPU."""

def add_steering_hook(model: nn.Module, layer_idx: int, vector: torch.Tensor,
                      dose: float) -> torch.utils.hooks.RemovableHandle:
    """register_forward_hook on model.model.layers[layer_idx] that adds
    (dose * vector).to(hs.dtype) to every position of the block output.
    Handles tuple output (modifies output[0], returns (hs, *output[1:])) and
    bare-tensor output (returns hs + delta), as the authors' hook does."""
```

Layer norms can be `nn.LayerNorm(HIDDEN)` or a small RMSNorm. Either works for the test.

### `src/write.py` (owned by S2)

```python
def write_dose(base_model: nn.Module, layer_idx: int, vector: torch.Tensor,
               dose: float) -> nn.Module
def save_write(path: str | Path, layer_idx: int, vector: torch.Tensor, dose: float) -> None
def load_write(path: str | Path) -> dict
```

Write file format: exactly the keys `{"layer": int, "dose": float, "vector": Tensor (CPU)}`
(`docs/MVP.md:100`; brief Decisions line 70).

## Subtasks

### S1: Tiny Qwen3-layout test model and steering hook
- Owns: `tests/tiny_qwen3.py`, `tests/test_tiny_qwen3.py`
- Depends on: none
- Do: Implement everything listed for `tests/tiny_qwen3.py` in Interfaces, with exactly those names,
  attributes and signatures. Every `nn.Linear` is `bias=False`, matching Qwen3 (brief line 23).
  The `add_steering_hook` rule (tuple vs tensor, cast delta to `hs.dtype`, keep the structure) is a
  minimal rewrite of the authors' hook (`04_selfmed_two_buttons.py:381-392`). Do not copy their
  mask, per-row coefficient or projection code. If the hook indexed `output[0]` on a bare tensor,
  it would silently steer only batch row 0 (brief Traps line 67), so branch on
  `isinstance(output, tuple)`. Do not edit anything under `third_party/`. Do not import transformers.
- Done when:
  - `tests/test_tiny_qwen3.py` passes on CPU with no GPU marker and checks that:
    (a) `make_tiny_model()` returns logits of shape `(2, 5, VOCAB)` for input `torch.randn(2, 5, HIDDEN)`;
    (b) `model.model.layers[i].mlp.down_proj` is an `nn.Linear` with `bias is None` for every i;
    (c) the same seed gives identical logits twice;
    (d) for both `return_tuple=True` and `return_tuple=False`, a hook at layer 1 with a nonzero vector
        changes the logits for **every** batch row, keeps the block output type
        (tuple stays tuple, tensor stays tensor), and after `handle.remove()` the logits equal the
        unhooked logits exactly.
  - `python -m pytest tests/test_tiny_qwen3.py -q` passes and `./scripts/check.sh` exits 0.

### S2: src/write.py and its CPU equivalence test
- Owns: `src/write.py`, `tests/test_write.py`
- Depends on: S1 (only through the Interfaces section)
- Do:
  - `write_dose`: follow `docs/MVP.md:83-96`, but replace the `assert` with exceptions, because
    `assert` is stripped under `python -O` (brief Traps line 64). Steps:
    1. `block = base_model.model.layers[layer_idx]`, `old = block.mlp.down_proj`.
    2. If `type(old) is not nn.Linear`, raise `TypeError`. This catches a LoRA wrapper after the
       adapter was attached, where writing would silently drop the adapter (brief Traps line 58).
    3. If `old.bias is not None`, raise `ValueError` with a message that says the layer already
       has a bias (it was already written, or it is not Qwen3). Settled answer and brief line 71.
       This also refuses a second write on the same layer (brief Traps line 59).
    4. If `tuple(vector.shape) != (old.out_features,)`, raise `ValueError` (brief line 71).
    5. `new = nn.Linear(old.in_features, old.out_features, bias=True, device=old.weight.device,
       dtype=old.weight.dtype)`. Under `torch.no_grad()`, run `new.weight.copy_(old.weight)` and
       `new.bias.copy_((dose * vector).to(new.bias))`. Set `block.mlp.down_proj = new` and return
       `base_model` (it is mutated in place).
    6. The docstring says: `dose` multiplies `vector` literally, and converting from
       working-coefficient units is the caller's job (settled answer). Pass the raw S2 vector, not
       the unit vector (`docs/MVP.md:99`). Take `layer_idx` from the frozen per-size setting, not
       from `pain_vectors.pt["layer"]` (brief Traps line 61). Call it on the base model before
       attaching the adapter (`docs/MVP.md:85-86`).
  - `save_write(path, layer_idx, vector, dose)`:
    `torch.save({"layer": int(layer_idx), "dose": float(dose), "vector": vector.detach().cpu()}, path)`.
    The docstring warns never to use `save_pretrained` for a written model, because reloading drops
    the write (`docs/MVP.md:101`).
  - `load_write(path)`: `return torch.load(path, map_location="cpu", weights_only=True)` (settled
    answer). Check that the keys are exactly `{"layer", "dose", "vector"}`, and raise `ValueError`
    otherwise.
  - Type hints and docstrings on all three functions (`pyproject.toml` ruff D103/ANN). Use
    `from __future__ import annotations` or `str | Path` directly, since the target is py310.
    No `# noqa`.
  - `load_generation` is out of scope. Do not add it.
- Done when:
  - `tests/test_write.py` uses `make_tiny_model`, `add_steering_hook`, `HIDDEN` from
    `tests.tiny_qwen3`, runs on CPU in float32 with no GPU marker, writes at layer index 1 (inside
    the stack, not the last layer: brief line 74), uses input `torch.randn(2, 5, HIDDEN)` with a
    fixed seed, and compares logits at all positions. It checks:
    1. **Equivalence (the brief's check for the INFERRED line 26).** Parametrized over dose 0.5 and
       1.0 (doses from the feature request and `docs/MVP.md:105`): a fresh unwritten model with
       `add_steering_hook(model, 1, v, dose)` and a fresh same-seed model after
       `write_dose(model, 1, v, dose)` with no hook give
       `torch.allclose(written, hooked, atol=1e-5, rtol=1e-5)`. That tolerance is a chosen value,
       not a sourced one (brief Expected values line 51), and the test comment must say so.
       Use a random vector `v = torch.randn(HIDDEN)`.
    2. Written logits differ from unwritten logits at dose 1.0, so the write has an effect
       (brief line 51).
    3. At dose 0.0, written logits equal unwritten logits with `torch.equal` (brief line 50).
    4. After `write_dose`, `down_proj` has `bias` equal to `dose * v`, and its weight equals the
       original weight.
    5. A layer whose `down_proj` already has a bias raises `ValueError`. Test this both by
       calling `write_dose` twice on the same layer and by swapping in an
       `nn.Linear(..., bias=True)` by hand.
    6. A vector of the wrong shape raises `ValueError`. A `down_proj` that is not a plain
       `nn.Linear` raises `TypeError` (a subclass or wrapper module standing in for LoRA).
    7. `save_write` then `load_write` (in pytest `tmp_path`) gives back `layer == 1`, the
       same `dose` and `torch.equal` on the vector. A file with extra or missing keys raises
       `ValueError` in `load_write`.
    8. The equivalence also holds with `make_tiny_model(return_tuple=False)` at dose 1.0, so both
       transformers output forms are covered (brief line 27).
  - `python -m pytest tests/test_write.py -q` passes and `./scripts/check.sh` exits 0.
