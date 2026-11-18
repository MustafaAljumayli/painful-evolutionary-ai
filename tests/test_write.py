"""CPU tests: a written dose matches a steering hook on the tiny model."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models import UncalibratedError
from src.write import load_generation, load_write, save_write, write_dose
from tests.prereg_tmp import FIXTURE_COEFFICIENT, FIXTURE_LAYER, write_prereg
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


def _bias(model: nn.Module) -> torch.Tensor | None:
    return model.model.layers[FIXTURE_LAYER].mlp.down_proj.bias


def _prereg(tmp_path):
    return write_prereg(
        tmp_path,
        {
            "8B": {
                "injection_layer": FIXTURE_LAYER,
                "working_coefficient": FIXTURE_COEFFICIENT,
            }
        },
    )


def test_load_generation_dose_one(tmp_path) -> None:
    """Dose 1.0 writes exactly the working coefficient."""
    v, x = _vector(), _inputs()
    got = load_generation(make_tiny_model(), "8B", v, 1.0, prereg_path=_prereg(tmp_path))
    ref = write_dose(make_tiny_model(), FIXTURE_LAYER, v, FIXTURE_COEFFICIENT)
    assert torch.equal(_bias(got), _bias(ref))
    with torch.no_grad():
        assert torch.equal(got(x), ref(x))


def test_load_generation_dose_half(tmp_path) -> None:
    """Dose 0.5 writes half the working coefficient, exactly."""
    v, x = _vector(), _inputs()
    p = _prereg(tmp_path)
    half = load_generation(make_tiny_model(), "8B", v, 0.5, prereg_path=p)
    one = load_generation(make_tiny_model(), "8B", v, 1.0, prereg_path=p)
    ref = write_dose(make_tiny_model(), FIXTURE_LAYER, v, 0.5 * FIXTURE_COEFFICIENT)
    assert torch.equal(_bias(half), _bias(ref))
    assert torch.equal(_bias(half), 0.5 * _bias(one))
    with torch.no_grad():
        assert torch.equal(half(x), ref(x))


def test_load_generation_uncalibrated_coefficient(tmp_path) -> None:
    """An unset coefficient raises UncalibratedError and leaves the model alone."""
    p = write_prereg(
        tmp_path,
        {
            "8B": {
                "injection_layer": FIXTURE_LAYER,
                "working_coefficient": FIXTURE_COEFFICIENT,
            },
            "14B": {"injection_layer": FIXTURE_LAYER, "working_coefficient": None},
        },
    )
    model = make_tiny_model()
    with pytest.raises(UncalibratedError, match="14B working_coefficient is not set") as ei:
        load_generation(model, "14B", _vector(), 1.0, prereg_path=p)
    assert ei.type is UncalibratedError
    assert _bias(model) is None


def test_load_generation_uncalibrated_layer(tmp_path) -> None:
    """An unset layer raises UncalibratedError."""
    p = write_prereg(
        tmp_path,
        {"14B": {"injection_layer": None, "working_coefficient": FIXTURE_COEFFICIENT}},
    )
    with pytest.raises(UncalibratedError, match="14B injection_layer is not set") as ei:
        load_generation(make_tiny_model(), "14B", _vector(), 1.0, prereg_path=p)
    assert ei.type is UncalibratedError


def test_load_generation_saves_write(tmp_path) -> None:
    """write_path gets a file with dose and the absolute coefficient."""
    v = _vector()
    out = tmp_path / "w.pt"
    load_generation(make_tiny_model(), "8B", v, 0.5, prereg_path=_prereg(tmp_path), write_path=out)
    data = load_write(out)
    assert data["size"] == "8B"
    assert data["layer"] == FIXTURE_LAYER
    assert data["dose"] == 0.5
    assert data["coefficient"] == 0.5 * FIXTURE_COEFFICIENT
    assert torch.equal(data["vector"], v)


def test_save_load_roundtrip(tmp_path) -> None:
    """save_write then load_write returns all five values."""
    v = _vector()
    path = tmp_path / "w.pt"
    save_write(path, "8B", FIXTURE_LAYER, v, 0.5, 0.5 * FIXTURE_COEFFICIENT)
    data = load_write(path)
    assert set(data) == {"size", "layer", "dose", "coefficient", "vector"}
    assert data["size"] == "8B"
    assert data["layer"] == FIXTURE_LAYER
    assert data["dose"] == 0.5
    assert data["coefficient"] == 0.5 * FIXTURE_COEFFICIENT
    assert torch.equal(data["vector"], v)


def _valid() -> dict:
    return {
        "size": "8B",
        "layer": FIXTURE_LAYER,
        "dose": 0.5,
        "coefficient": 0.5 * FIXTURE_COEFFICIENT,
        "vector": _vector(),
    }


def test_load_write_bad_keys(tmp_path) -> None:
    """Extra or missing keys raise ValueError."""
    extra = _valid()
    extra["x"] = 1
    cases = {"extra": extra}
    for key in ("dose", "coefficient", "size"):
        d = _valid()
        del d[key]
        cases[f"no_{key}"] = d
    for name, d in cases.items():
        path = tmp_path / f"{name}.pt"
        torch.save(d, path)
        with pytest.raises(ValueError):
            load_write(path)


@pytest.mark.parametrize(
    "key,value", [("dose", None), ("coefficient", None), ("dose", True), ("coefficient", "2.5")]
)
def test_load_write_bad_numbers(tmp_path, key: str, value: object) -> None:
    """None, bool and str numbers raise ValueError."""
    d = _valid()
    d[key] = value
    path = tmp_path / "bad.pt"
    torch.save(d, path)
    with pytest.raises(ValueError, match=key):
        load_write(path)
