"""Tests for src/models.py. Prereg files are written under tmp_path."""

import dataclasses
import inspect
import json
from pathlib import Path

import pytest

from src import models
from src.models import (
    MODELS,
    PREREG_PATH,
    ModelSpec,
    UncalibratedError,
    get_spec,
    injection_layer,
    working_coefficient,
)

SIZES = ["8B", "14B", "32B"]
LAYERS = {"8B": 36, "14B": 40, "32B": 64}


def write(tmp_path: Path, data: object) -> Path:
    p = tmp_path / "models.json"
    p.write_text(json.dumps(data))
    return p


def full(layer: object, coef: object) -> dict:
    return {s: {"injection_layer": layer, "working_coefficient": coef} for s in SIZES}


def test_models_table() -> None:
    assert set(MODELS) == set(SIZES)
    assert {s: MODELS[s].hf_id for s in SIZES} == {
        "8B": "Qwen/Qwen3-8B",
        "14B": "Qwen/Qwen3-14B",
        "32B": "Qwen/Qwen3-32B",
    }
    assert {s: MODELS[s].num_layers for s in SIZES} == LAYERS
    assert all(MODELS[s].thinking is False for s in SIZES)
    assert all(get_spec(s) is MODELS[s] for s in SIZES)


def test_spec_frozen_and_fields() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        MODELS["8B"].num_layers = 1  # type: ignore[misc]
    assert [f.name for f in dataclasses.fields(ModelSpec)] == [
        "size",
        "hf_id",
        "num_layers",
        "thinking",
    ]


def test_uncalibrated_is_lookup_error() -> None:
    assert issubclass(UncalibratedError, LookupError)


def uncalibrated_files(tmp_path: Path) -> dict[str, Path]:
    return {
        "missing": tmp_path / "nope.json",
        "empty": write(tmp_path, {}),
    }


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("case", ["missing", "empty", "size_key", "field", "null"])
def test_uncalibrated(tmp_path: Path, size: str, case: str) -> None:
    if case == "missing":
        path = tmp_path / "nope.json"
    elif case == "empty":
        path = write(tmp_path, {})
    elif case == "size_key":
        path = write(tmp_path, {"other": {"injection_layer": 1, "working_coefficient": 1}})
    elif case == "field":
        path = write(tmp_path, {s: {} for s in SIZES})
    else:
        path = write(tmp_path, full(None, None))
    for fn, field in (
        (injection_layer, "injection_layer"),
        (working_coefficient, "working_coefficient"),
    ):
        with pytest.raises(UncalibratedError) as exc:
            fn(size, path)
        msg = str(exc.value)
        assert size in msg and field in msg and str(path) in msg


@pytest.mark.parametrize("size", SIZES)
def test_layer_bounds(tmp_path: Path, size: str) -> None:
    last = LAYERS[size] - 1
    assert injection_layer(size, write(tmp_path, full(last, 1.0))) == last
    assert injection_layer(size, write(tmp_path, full(0, 1.0))) == 0
    for bad in (last + 1, -1):
        with pytest.raises(ValueError, match=rf"0\.\.{last}"):
            injection_layer(size, write(tmp_path, full(bad, 1.0)))


@pytest.mark.parametrize("bad", [True, 16.0, "16"])
def test_layer_bad_type(tmp_path: Path, bad: object) -> None:
    with pytest.raises(ValueError):
        injection_layer("8B", write(tmp_path, full(bad, 1.0)))


@pytest.mark.parametrize("bad", [0, -1.0, True, "1"])
def test_coefficient_rejected(tmp_path: Path, bad: object) -> None:
    with pytest.raises(ValueError):
        working_coefficient("8B", write(tmp_path, full(20, bad)))


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_coefficient_nonfinite(tmp_path: Path, token: str) -> None:
    p = tmp_path / "models.json"
    value = {"nan": float("nan")}  # keep json.dumps(float("nan")) form
    assert json.dumps(value["nan"]) == "NaN"
    p.write_text(
        "{"
        + ", ".join(
            f'"{s}": {{"injection_layer": 1, "working_coefficient": {token}}}' for s in SIZES
        )
        + "}"
    )
    with pytest.raises(ValueError):
        working_coefficient("8B", p)


def test_coefficient_int_returned_as_float(tmp_path: Path) -> None:
    out = working_coefficient("8B", write(tmp_path, full(20, 3)))
    assert out == 3.0 and isinstance(out, float)


@pytest.mark.parametrize("size", SIZES)
def test_calibrated_roundtrip(tmp_path: Path, size: str) -> None:
    # Test fixtures, not findings.
    p = write(tmp_path, full(20, 2.5))
    assert injection_layer(size, p) == 20
    assert working_coefficient(size, p) == 2.5


def test_unknown_size(tmp_path: Path) -> None:
    p = write(tmp_path, full(20, 2.5))
    with pytest.raises(KeyError, match="8B, 14B, 32B"):
        get_spec("7B")
    with pytest.raises(KeyError):
        injection_layer("7B", p)
    with pytest.raises(KeyError):
        working_coefficient("7B", tmp_path / "nope.json")
    with pytest.raises(KeyError):
        injection_layer("7B", tmp_path / "nope.json")


def test_signatures() -> None:
    for fn in (injection_layer, working_coefficient):
        params = list(inspect.signature(fn).parameters.values())
        assert [p.name for p in params] == ["size", "prereg_path"]
        assert params[0].default is inspect.Parameter.empty
        assert params[1].default == PREREG_PATH
    assert PREREG_PATH.parts[-2:] == ("prereg", "models.json")
    assert models.PREREG_PATH == PREREG_PATH


@pytest.mark.gpu
@pytest.mark.parametrize("size", SIZES)
def test_layer_counts_match_hub(size: str) -> None:
    transformers = pytest.importorskip("transformers")
    spec = get_spec(size)
    config = transformers.AutoConfig.from_pretrained(spec.hf_id)
    assert config.num_hidden_layers == spec.num_layers
    pytest.importorskip("accelerate")
    from accelerate import init_empty_weights

    with init_empty_weights():
        model = transformers.AutoModelForCausalLM.from_config(config)
    assert len(model.model.layers) == spec.num_layers
