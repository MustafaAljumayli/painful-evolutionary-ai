"""Tests for the tiny Qwen3-layout test model and its steering hook."""

import pytest
import torch
from torch import nn

from tests.tiny_qwen3 import (
    HIDDEN,
    N_LAYERS,
    VOCAB,
    add_steering_hook,
    make_tiny_model,
)


def _x() -> torch.Tensor:
    torch.manual_seed(123)
    return torch.randn(2, 5, HIDDEN)


def test_logits_shape() -> None:
    """Logits have shape (batch, seq, VOCAB)."""
    assert make_tiny_model()(_x()).shape == (2, 5, VOCAB)


def test_down_proj_has_no_bias() -> None:
    """Every down_proj is a plain nn.Linear without bias."""
    model = make_tiny_model()
    for i in range(N_LAYERS):
        dp = model.model.layers[i].mlp.down_proj
        assert type(dp) is nn.Linear
        assert dp.bias is None


def test_same_seed_same_logits() -> None:
    """The same seed gives identical logits."""
    x = _x()
    assert torch.equal(make_tiny_model(3)(x), make_tiny_model(3)(x))


@pytest.mark.parametrize("return_tuple", [True, False])
def test_hook_steers_every_row_and_removes_cleanly(return_tuple: bool) -> None:
    """The hook changes all batch rows, keeps the output type, and is removable."""
    model = make_tiny_model(return_tuple=return_tuple)
    x = _x()
    base = model(x)
    types: list[type] = []
    handle = add_steering_hook(model, 1, torch.randn(HIDDEN), 1.0)
    # Registered after the steering hook, so it sees the steering hook's output.
    probe = model.model.layers[1].register_forward_hook(lambda _m, _i, out: types.append(type(out)))
    steered = model(x)
    probe.remove()
    for row in range(x.shape[0]):
        assert not torch.equal(steered[row], base[row])
    assert types[0] is (tuple if return_tuple else torch.Tensor)
    handle.remove()
    assert torch.equal(model(x), base)
