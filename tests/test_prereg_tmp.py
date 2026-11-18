"""Tests for the temporary prereg helper."""

from __future__ import annotations

from pathlib import Path

from src.models import injection_layer, working_coefficient
from tests.prereg_tmp import FIXTURE_COEFFICIENT, FIXTURE_LAYER, write_prereg
from tests.tiny_qwen3 import N_LAYERS


def test_write_prereg_readable_by_models(tmp_path: Path) -> None:
    entries = {
        "8B": {
            "injection_layer": FIXTURE_LAYER,
            "working_coefficient": FIXTURE_COEFFICIENT,
        }
    }
    path = write_prereg(tmp_path, entries)
    assert working_coefficient("8B", path) == FIXTURE_COEFFICIENT
    assert injection_layer("8B", path) == FIXTURE_LAYER


def test_fixture_layer_fits_tiny_model() -> None:
    assert FIXTURE_LAYER < N_LAYERS
