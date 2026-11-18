# Plan: src/models.py, per-size Qwen3 records with injection layer and working coefficient read from prereg/models.json

## Interfaces
Fixed for every subtask. Import path is `src.models` (pytest `pythonpath = ["."]`, `pyproject.toml:22`).

```python
# src/models.py
from dataclasses import dataclass
from pathlib import Path

PREREG_PATH: Path  # = Path(__file__).resolve().parent.parent / "prereg" / "models.json"

class UncalibratedError(LookupError):
    """Raised when a size's injection layer or working coefficient is not set yet."""

@dataclass(frozen=True)
class ModelSpec:
    size: str          # "8B" | "14B" | "32B"
    hf_id: str         # "Qwen/Qwen3-8B" | "Qwen/Qwen3-14B" | "Qwen/Qwen3-32B"
    num_layers: int    # 36 | 40 | 64
    thinking: bool = False

MODELS: dict[str, ModelSpec]  # keys exactly "8B", "14B", "32B"

def get_spec(size: str) -> ModelSpec: ...
def injection_layer(size: str, prereg_path: Path = PREREG_PATH) -> int: ...
def working_coefficient(size: str, prereg_path: Path = PREREG_PATH) -> float: ...
```

`ModelSpec` has no layer or coefficient field and no field defaults other than
`thinking=False`. (Brief, "Decisions made".)

`prereg/models.json` schema (brief, "Decisions made"):

```json
{
  "8B":  {"injection_layer": null, "working_coefficient": null},
  "14B": {"injection_layer": null, "working_coefficient": null},
  "32B": {"injection_layer": null, "working_coefficient": null}
}
```

`injection_layer` is `int | null`, `working_coefficient` is `number | null`.
Uncalibrated = file missing, size key missing, field missing, or field `null`.

Errors:
- Unknown size: `KeyError`, message lists the valid sizes `8B, 14B, 32B`.
- Uncalibrated: `UncalibratedError`, message names the size, the field
  (`injection_layer` or `working_coefficient`) and the prereg path.
- Bad layer: `ValueError`, message names the size, the value and the range
  `0..num_layers-1` (e.g. `0..35`).
- Bad coefficient: `ValueError`, message names the size and the value.

## Subtasks

### S1: src/models.py and its tests
- Owns: `src/models.py`, `tests/test_models.py`
- Depends on: none
- Do:
  - Implement exactly the Interfaces section. Module docstring says what the
    module is for (per-size Qwen3 settings; calibrated values live in
    `prereg/models.json`).
  - `MODELS` entries, hard-coded, each with the config URL and repo sha in a
    comment beside the layer count:
    - `"8B"`: `Qwen/Qwen3-8B`, `num_layers=36`. Source:
      https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json
      (`num_hidden_layers`, line 18, sha `b968826d9c46dd6066d109eabc6255188de91218`).
    - `"14B"`: `Qwen/Qwen3-14B`, `num_layers=40`. Source:
      https://huggingface.co/Qwen/Qwen3-14B/blob/main/config.json
      (line 18, sha `40c069824f4251a91eefaf281ebe4c544efd3e18`).
    - `"32B"`: `Qwen/Qwen3-32B`, `num_layers=64`. Source:
      https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json
      (line 18, sha `9216db5781bf21249d130ec9da846c4624c16137`).
    - `thinking=False` for all three. A comment says callers must pass it as
      `enable_thinking=False` to `tokenizer.apply_chat_template`, because the
      Qwen3 template defaults to True (Qwen3-8B model card README line 70).
  - Layer index is 0-based into `model.model.layers`, following the authors'
    `04_selfmed_two_buttons.py` at commit 213fb6c, lines 139-143 and 399.
    Valid range `0 <= L <= num_layers - 1`. Say so in a comment.
  - `get_spec(size)`: return `MODELS[size]`; unknown size raises `KeyError`
    listing valid sizes.
  - Private helper reads the prereg file with `json.load` each call (no
    caching, so tests with different tmp files do not interfere). Missing file
    means uncalibrated, not `FileNotFoundError`.
  - `injection_layer`: call `get_spec` first (unknown size wins over missing
    file). Value `None`/missing -> `UncalibratedError`. Use `is None`, never a
    truthiness check: layer 0 is a valid calibrated value. Reject `bool`
    explicitly (`True` is an `int`), reject floats including `16.0` (no
    `int()` cast), reject `< 0` and `>= num_layers` with `ValueError`.
  - `working_coefficient`: `None`/missing -> `UncalibratedError`. Must be an
    `int` or `float` but not `bool`, finite (`math.isfinite`) and `> 0`, else
    `ValueError`. Return as `float`.
  - No Qwen 2.5 value as a default anywhere: no `16`, `38`, `1.0` as a default
    argument, fallback or `dict.get(..., default)` value (CLAUDE.md Rules;
    authors' Qwen 2.5 rows are `16, 1.0` and `38, 1.0`, `s04.py:73-78`). Do not
    copy the authors' `MODELS` table. Do not reject 16 or 38 as calibrated
    values: they are legal Qwen3 layers.
  - Type hints and docstrings on all public functions (ruff D103, ANN001,
    ANN201). No `transformers`, `torch` or network imports.
- Done when:
  - `./scripts/check.sh` exits 0.
  - `python -m pytest tests/test_models.py -q` passes on CPU, no network.
    Tests write their own prereg files under pytest `tmp_path` and pass the
    path explicitly; none reads the repo's `prereg/models.json`. Tests cover:
    - `MODELS` keys are exactly `{"8B", "14B", "32B"}`; hf ids are
      `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B`, `Qwen/Qwen3-32B`; `num_layers` are
      36, 40, 64; `thinking is False` for all three.
    - `ModelSpec` is frozen (assigning a field raises
      `dataclasses.FrozenInstanceError`) and has no layer or coefficient field
      (`dataclasses.fields` names are exactly size, hf_id, num_layers, thinking).
    - Missing file, empty `{}` file, size key missing, field missing and
      field `null`: both readers raise `UncalibratedError` for every size, and
      the message contains the size, the field name and the path.
    - `UncalibratedError` is a subclass of `LookupError`.
    - Layer bounds: 8B accepts 35, rejects 36; 14B accepts 39, rejects 40;
      32B accepts 63, rejects 64; every size rejects -1 and accepts 0
      (brief, "Expected values"). Rejection is `ValueError` whose message
      contains the range, e.g. `0..35`.
    - Layer `true`, `16.0` and `"16"` are rejected with `ValueError`.
    - Coefficient `0`, `-1.0`, `true`, `"1"` are rejected with `ValueError`;
      `NaN`/`Infinity` (write with `json.dumps(float("nan"))`) rejected; a
      positive int is returned as `float`.
    - A calibrated file returns the stored layer and coefficient unchanged
      (use test values that are not Qwen 2.5 values, e.g. layer 20,
      coefficient 2.5, marked in a comment as test fixtures, not findings).
    - Unknown size `"7B"` raises `KeyError` from `get_spec`,
      `injection_layer` and `working_coefficient`.
    - A source check: `inspect.signature` of `injection_layer` and
      `working_coefficient` has only `prereg_path` with a default (equal to
      `PREREG_PATH`), and `PREREG_PATH` ends with `prereg/models.json`.
  - Reader confirms each layer count in the code carries the config URL in a
    comment. The brief's INFERRED finding (0-based range equals
    `num_hidden_layers`) has a check: a `@pytest.mark.gpu` test in
    `tests/test_models.py` that, for each size, loads
    `AutoConfig.from_pretrained(spec.hf_id)` and asserts
    `config.num_hidden_layers == spec.num_layers`, and loads the model on
    meta device (or skips with a reason if `accelerate` is unavailable) to
    assert `len(model.model.layers) == spec.num_layers`. It imports
    `transformers` inside the test function so CPU collection does not need it.
    `scripts/check.sh` skips it.

### S2: Uncalibrated prereg/models.json template
- Owns: `prereg/models.json`, `tests/test_prereg_models_file.py`
- Depends on: S1 (Interfaces only)
- Do:
  - Create `prereg/models.json` with exactly the schema in Interfaces: all
    three sizes, every `injection_layer` and `working_coefficient` set to
    `null`. No numbers in it: values are frozen only after calibration
    (`docs/MVP.md:117`), and inventing any is forbidden (CLAUDE.md Rules).
  - Test file checks the committed file against the Interfaces only.
- Done when:
  - `./scripts/check.sh` exits 0.
  - `python -m pytest tests/test_prereg_models_file.py -q` passes, with tests
    that:
    - load the file at `src.models.PREREG_PATH` with `json.load`; its top-level
      keys equal `set(src.models.MODELS)`; each entry has exactly the keys
      `injection_layer` and `working_coefficient`.
    - while the file is uncalibrated (all values `null`), `injection_layer`
      and `working_coefficient` with the default path raise
      `UncalibratedError` for every size. Guard this test with
      `pytest.skip("prereg calibrated")` if any value is non-null, so freezing
      real values later does not break the suite.
