"""Helper that writes a temporary prereg models file for CPU tests."""

from __future__ import annotations

import json
from pathlib import Path

# Test fixture values, arbitrary (not sourced numbers; tiny model only).
FIXTURE_LAYER: int = 1  # must be < tests.tiny_qwen3.N_LAYERS (3)
FIXTURE_COEFFICIENT: float = 2.5  # non-power-of-two so scaling is exercised


def write_prereg(tmp_dir: Path, entries: dict[str, dict[str, object]]) -> Path:
    """Write entries as JSON to tmp_dir / "models.json" and return that path."""
    path = Path(tmp_dir) / "models.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path
