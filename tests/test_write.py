"""CPU tests: a written dose matches a steering hook on the tiny model."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.write import load_write, save_write, write_dose
from tests.tiny_qwen3 import HIDDEN, add_steering_hook, make_tiny_model

LAYER = 1


def _inputs() -> torch.Tensor:
    torch.manual_seed(123)
    return torch.randn(2, 5, HIDDEN)


def _vector() -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(HIDDEN)


@pytest.mark.parametrize("dose", [0.5, 1.0])
def test_equivalence(dose: float) -> None:
    """Written model matches a hooked model at all positions."""
    x, v = _inputs(), _vector()
    hooked_model = make_tiny_model()
    add_steering_hook(hooked_model, LAYER, v, dose)
    written_model = write_dose(make_tiny_model(), LAYER, v, dose)
    with torch.no_grad():
        hooked = hooked_model(x)
        written = written_model(x)
    # atol and rtol are chosen values, not sourced ones.
    assert torch.allclose(written, hooked, atol=1e-5, rtol=1e-5)


def test_equivalence_bare_tensor_output() -> None:
    """Equivalence also holds when blocks return a bare tensor."""
    x, v = _inputs(), _vector()
    hooked_model = make_tiny_model(return_tuple=False)
    add_steering_hook(hooked_model, LAYER, v, 1.0)
    written_model = write_dose(make_tiny_model(return_tuple=False), LAYER, v, 1.0)
    with torch.no_grad():
        assert torch.allclose(written_model(x), hooked_model(x), atol=1e-5, rtol=1e-5)


def test_write_has_effect() -> None:
    """Written logits differ from unwritten logits at dose 1.0."""
    x, v = _inputs(), _vector()
    written_model = write_dose(make_tiny_model(), LAYER, v, 1.0)
    with torch.no_grad():
        assert not torch.allclose(written_model(x), make_tiny_model()(x))


def test_dose_zero_is_identity() -> None:
    """Dose 0.0 leaves logits exactly unchanged."""
    x, v = _inputs(), _vector()
    written_model = write_dose(make_tiny_model(), LAYER, v, 0.0)
    with torch.no_grad():
        assert torch.equal(written_model(x), make_tiny_model()(x))


def test_bias_and_weight() -> None:
    """Bias equals dose * vector and the weight is untouched."""
    v = _vector()
    original = make_tiny_model().model.layers[LAYER].mlp.down_proj.weight
    model = write_dose(make_tiny_model(), LAYER, v, 0.5)
    proj = model.model.layers[LAYER].mlp.down_proj
    assert torch.equal(proj.bias, 0.5 * v)
    assert torch.equal(proj.weight, original)


def test_existing_bias_raises() -> None:
    """A layer that already has a bias is refused."""
    v = _vector()
    model = write_dose(make_tiny_model(), LAYER, v, 1.0)
    with pytest.raises(ValueError, match="bias"):
        write_dose(model, LAYER, v, 1.0)
    other = make_tiny_model()
    proj = other.model.layers[LAYER].mlp.down_proj
    other.model.layers[LAYER].mlp.down_proj = nn.Linear(
        proj.in_features, proj.out_features, bias=True
    )
    with pytest.raises(ValueError, match="bias"):
        write_dose(other, LAYER, v, 1.0)


def test_wrong_shape_raises() -> None:
    """A vector of the wrong shape is refused."""
    with pytest.raises(ValueError, match="shape"):
        write_dose(make_tiny_model(), LAYER, torch.randn(HIDDEN + 1), 1.0)


def test_non_linear_raises() -> None:
    """A subclass standing in for a LoRA wrapper is refused."""

    class Wrapped(nn.Linear):
        pass

    model = make_tiny_model()
    proj = model.model.layers[LAYER].mlp.down_proj
    model.model.layers[LAYER].mlp.down_proj = Wrapped(
        proj.in_features, proj.out_features, bias=False
    )
    with pytest.raises(TypeError):
        write_dose(model, LAYER, _vector(), 1.0)


def test_save_load_roundtrip(tmp_path) -> None:
    """save_write then load_write returns the same layer, dose and vector."""
    v = _vector()
    path = tmp_path / "w.pt"
    save_write(path, LAYER, v, 0.5)
    data = load_write(path)
    assert data["layer"] == LAYER
    assert data["dose"] == 0.5
    assert torch.equal(data["vector"], v)


def test_load_write_bad_keys(tmp_path) -> None:
    """Extra or missing keys raise ValueError."""
    extra = tmp_path / "extra.pt"
    torch.save({"layer": 1, "dose": 1.0, "vector": _vector(), "x": 1}, extra)
    with pytest.raises(ValueError):
        load_write(extra)
    missing = tmp_path / "missing.pt"
    torch.save({"layer": 1, "dose": 1.0}, missing)
    with pytest.raises(ValueError):
        load_write(missing)
