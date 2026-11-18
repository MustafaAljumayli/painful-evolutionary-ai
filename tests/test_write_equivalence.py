"""Proof that a written dose matches steering: tiny model on CPU, real Qwen3-8B on GPU."""

from __future__ import annotations

import pytest
import torch

from src.models import UncalibratedError, injection_layer, working_coefficient
from src.run_choice import (
    DATASET_PATH,
    DTYPES,
    PAIRS_PATH,
    ROOT,
    Steering,
    default_vector_path,
    first_choice_prompt_ids,
    get_layers,
    load_model,
    load_pairs,
    load_scenarios,
    load_vector,
    select_device,
)
from src.write import load_generation, write_dose
from tests.tiny_qwen3 import HIDDEN, make_tiny_model

# Arbitrary test values for the tiny model.
TINY_LAYER = 1
TINY_DOSES = [0.5, 1.0]
# Sourced: the plan and brief (research/feature-run-choice.md, "GPU proof test").
GPU_DOSES = [0.5, 1.0]
N_PROMPTS = 50
MIN_AGREEMENT = 0.99
SIZE = "8B"
FIXTURE_PAIRS = ROOT / "tests" / "fixtures" / "pairs_small.json"


def _steer(model: torch.nn.Module, layer: int, v: torch.Tensor, coeff: float) -> Steering:
    """Attach the runner's steering hook with one row at `coeff` times the raw vector."""
    st = Steering((v / v.norm()).float())
    st.G["coeff"] = torch.tensor([coeff], device=v.device)
    st.G["dirs"] = v[None, :]
    model.model.layers[layer].register_forward_hook(st.steer_hook)
    return st


@pytest.mark.parametrize("return_tuple", [True, False])
@pytest.mark.parametrize("dose", TINY_DOSES)
def test_tiny_model_hook_equals_write(dose: float, return_tuple: bool) -> None:
    """The runner's hook and write_dose give the same top token and near-equal logits."""
    torch.manual_seed(5)
    x = torch.randn(4, 6, HIDDEN)
    v = torch.randn(HIDDEN)
    hooked = make_tiny_model(return_tuple=return_tuple)
    _steer(hooked, TINY_LAYER, v, dose)
    written = write_dose(make_tiny_model(return_tuple=return_tuple), TINY_LAYER, v, dose)
    plain = make_tiny_model(return_tuple=return_tuple)
    with torch.no_grad():
        a, b, c = hooked(x)[:, -1], written(x)[:, -1], plain(x)[:, -1]
    max_diff = (a - b).abs().max().item()
    print(f"tiny dose {dose}: max logit difference hook vs write = {max_diff:.3e}")
    assert torch.equal(a.argmax(-1), b.argmax(-1))
    # atol is a chosen value for float32 on CPU, not a sourced one.
    assert max_diff < 1e-5
    assert (a - c).abs().max().item() > max_diff  # the steering does something


def _gpu_prompts(tok: object) -> list[list[int]]:
    path = PAIRS_PATH if PAIRS_PATH.exists() else FIXTURE_PAIRS
    pairs = load_pairs(path)
    pair = pairs[sorted(pairs)[0]]
    scen = load_scenarios(DATASET_PATH)
    turns = [s[0] for kind in sorted(scen) for s in scen[kind]]
    turns = turns[:N_PROMPTS]
    return [
        first_choice_prompt_ids(tok, pair, "pain_on_button_works", i % 2, t)
        for i, t in enumerate(turns)
    ]


@pytest.mark.gpu
def test_real_model_hook_equals_write() -> None:
    """On Qwen3-8B, hook and write agree on the top token for >=99% of 50 first-choice prompts."""
    try:
        layer = injection_layer(SIZE)
        coefficient = working_coefficient(SIZE)
    except UncalibratedError as e:
        pytest.skip(f"{SIZE} is not calibrated: {e}")
    try:
        v, _ = load_vector(default_vector_path(SIZE))
    except SystemExit as e:
        pytest.skip(f"vector cannot be loaded: {e}")
    device = select_device()
    dtype = "bfloat16"
    tok, model = load_model("Qwen/Qwen3-8B", device, dtype)
    prompts = _gpu_prompts(tok)
    assert len(prompts) == N_PROMPTS
    vec = v.to(device, dtype=DTYPES[dtype])

    @torch.inference_mode()
    def last_logits() -> torch.Tensor:
        rows = [
            model(input_ids=torch.tensor([p], device=device)).logits[0, -1].float().cpu()
            for p in prompts
        ]
        return torch.stack(rows)

    layer_mod = get_layers(model)[layer].mlp
    original = layer_mod.down_proj
    steered = {}
    for dose in GPU_DOSES:
        _steer(model, layer, vec, dose * coefficient)
        steered[dose] = last_logits()
        get_layers(model)[layer]._forward_hooks.clear()
    for dose in GPU_DOSES:
        load_generation(model, SIZE, v.to(device, dtype=DTYPES[dtype]), dose)
        written = last_logits()
        layer_mod.down_proj = original
        agree = (steered[dose].argmax(-1) == written.argmax(-1)).float().mean().item()
        max_diff = (steered[dose] - written).abs().max().item()
        print(f"{SIZE} dose {dose}: top-token agreement {agree:.3f}, max logit diff {max_diff:.3f}")
        assert agree >= MIN_AGREEMENT
