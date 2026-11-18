"""Pin the provenance evidence from research/feature-vectors.md.

The authors' Qwen3 8B and 14B pain vectors carry the folder suffix "_base".
These tests check, from the submodule's own files, that the suffix is only a
label: the extraction ids are the post-trained Qwen3 checkpoints.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "third_party" / "Pain-axis"
SCRIPT = "scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py"

# Brief, Reuse, from 01_extract...py:87-88.
HF_IDS = {"Qwen_3_8B_base": "Qwen/Qwen3-8B", "Qwen_3_14B_base": "Qwen/Qwen3-14B"}

# Brief, Findings, from per_model/<name>/summary.json:2-6.
EXPECTED = {
    "Qwen_3_8B_base": {"n_layers": 36, "d_model": 4096, "best_layer_final_token": 34},
    "Qwen_3_14B_base": {"n_layers": 40, "d_model": 5120, "best_layer_final_token": 31},
}

NAMES = list(HF_IDS)


def _path(*parts: str) -> Path:
    """Return a path under the submodule root, skipping the test if it is missing."""
    p = ROOT.joinpath(*parts)
    if not p.is_file():
        pytest.skip("third_party/Pain-axis not initialized; run git submodule update --init")
    return p


def _script_text() -> str:
    """Read the extraction script as text without importing it."""
    return _path(*SCRIPT.split("/")).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", NAMES)
def test_summary_json(name: str) -> None:
    """The stored summary names the post-trained checkpoint and expected shape."""
    path = _path("results", "3.2_pain_vectors", "per_model", name, "summary.json")
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["model"] == HF_IDS[name]
    assert summary["model_name"] == name
    for key, value in EXPECTED[name].items():
        assert summary[key] == value, key


def test_extraction_script_registry() -> None:
    """The registry maps the _base folder names to the non-Base HF ids (:87-88)."""
    text = _script_text()
    assert '("Qwen/Qwen3-8B", "Qwen_3_8B_base")' in text
    assert '("Qwen/Qwen3-14B", "Qwen_3_14B_base")' in text
    assert '"Qwen/Qwen3-8B-Base"' not in text
    assert '"Qwen/Qwen3-14B-Base"' not in text


def test_weights_load_from_registry_string() -> None:
    """Weights load straight from the registry string, with no alias map (:423-428)."""
    assert "AutoModelForCausalLM.from_pretrained(model_path" in _script_text()
