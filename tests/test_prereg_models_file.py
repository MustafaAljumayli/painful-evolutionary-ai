"""Checks the committed prereg/models.json against the models.py interface."""

import json

import pytest

from src import models
from src.models import MODELS, PREREG_PATH, UncalibratedError

FIELDS = {"injection_layer", "working_coefficient"}


def _load() -> dict:
    with open(PREREG_PATH) as f:
        return json.load(f)


def test_prereg_file_schema() -> None:
    """Top-level keys match MODELS and each entry has exactly the two fields."""
    data = _load()
    assert set(data) == set(MODELS)
    for size, entry in data.items():
        assert set(entry) == FIELDS, size


@pytest.mark.parametrize("size", sorted(MODELS))
def test_uncalibrated_template_raises(size: str) -> None:
    """While every value is null, both readers raise UncalibratedError."""
    data = _load()
    if any(v is not None for entry in data.values() for v in entry.values()):
        pytest.skip("prereg calibrated")
    with pytest.raises(UncalibratedError):
        models.injection_layer(size)
    with pytest.raises(UncalibratedError):
        models.working_coefficient(size)
