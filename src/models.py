"""Per-size Qwen3 settings.

Fixed facts about each model (id, layer count) live here. The calibrated
injection layer and working coefficient live in ``prereg/models.json`` and are
read on every call. Until they are frozen there, the readers raise
``UncalibratedError``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

PREREG_PATH: Path = Path(__file__).resolve().parent.parent / "prereg" / "models.json"


class UncalibratedError(LookupError):
    """Raised when a size's injection layer or working coefficient is not set yet."""


@dataclass(frozen=True)
class ModelSpec:
    """Fixed facts about one Qwen3 size."""

    size: str
    hf_id: str
    num_layers: int
    thinking: bool = False


# thinking=False everywhere. Callers must pass it as enable_thinking=False to
# tokenizer.apply_chat_template, because the Qwen3 template defaults to True
# (Qwen3-8B model card README line 70).
MODELS: dict[str, ModelSpec] = {
    "8B": ModelSpec(
        size="8B",
        hf_id="Qwen/Qwen3-8B",
        # https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json
        # num_hidden_layers, line 18, sha b968826d9c46dd6066d109eabc6255188de91218
        num_layers=36,
    ),
    "14B": ModelSpec(
        size="14B",
        hf_id="Qwen/Qwen3-14B",
        # https://huggingface.co/Qwen/Qwen3-14B/blob/main/config.json
        # num_hidden_layers, line 18, sha 40c069824f4251a91eefaf281ebe4c544efd3e18
        num_layers=40,
    ),
    "32B": ModelSpec(
        size="32B",
        hf_id="Qwen/Qwen3-32B",
        # https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json
        # num_hidden_layers, line 18, sha 9216db5781bf21249d130ec9da846c4624c16137
        num_layers=64,
    ),
}


def get_spec(size: str) -> ModelSpec:
    """Return the fixed spec for a size ("8B", "14B" or "32B")."""
    if size not in MODELS:
        raise KeyError(f"unknown size {size!r}; valid sizes: {', '.join(MODELS)}")
    return MODELS[size]


def _read_field(size: str, field: str, prereg_path: Path) -> object:
    """Read one prereg field; raise UncalibratedError if it is absent or null."""
    try:
        with open(prereg_path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    entry = data.get(size) if isinstance(data, dict) else None
    value = entry.get(field) if isinstance(entry, dict) else None
    if value is None:
        raise UncalibratedError(f"{size} {field} is not set in {prereg_path}")
    return value


def injection_layer(size: str, prereg_path: Path = PREREG_PATH) -> int:
    """Return the calibrated injection layer for a size.

    The index is 0-based into ``model.model.layers``, following the authors'
    ``04_selfmed_two_buttons.py`` at commit 213fb6c, lines 139-143 and 399.
    Valid range is 0 <= L <= num_layers - 1.
    """
    spec = get_spec(size)
    value = _read_field(size, "injection_layer", prereg_path)
    last = spec.num_layers - 1
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= last:
        raise ValueError(f"{size} injection_layer {value!r} must be an int in 0..{last}")
    return value


def working_coefficient(size: str, prereg_path: Path = PREREG_PATH) -> float:
    """Return the calibrated working coefficient for a size, as a positive finite float."""
    get_spec(size)
    value = _read_field(size, "working_coefficient", prereg_path)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{size} working_coefficient {value!r} must be a finite number > 0")
    return float(value)
