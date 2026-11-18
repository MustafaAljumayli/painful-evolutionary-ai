# Plan: one dose-to-coefficient conversion in load_generation, and write files that record both dose and coefficient

Serves MVP output 2 (the write proof) and every later run that turns a dose
into weights. Stage: `vector + dose -> write (weights)`. Brief:
`research/feature-dose-units.md`.

## Interfaces

### `tests/prereg_tmp.py` (made by S1, used by S2)

```python
# Arbitrary fixture values for CPU tests, not sourced numbers.
FIXTURE_LAYER: int = 1            # must be < tests.tiny_qwen3.N_LAYERS (3)
FIXTURE_COEFFICIENT: float = 2.5  # non-power-of-two so scaling is exercised

def write_prereg(tmp_dir: Path, entries: dict[str, dict[str, object]]) -> Path:
    """Write entries as JSON to tmp_dir / "models.json" and return that path."""
```

The JSON shape is the one `src/models.py` reads (`src/models.py:68-79`):
`{"<size>": {"injection_layer": <int or null>, "working_coefficient": <number or null>}}`,
sizes `"8B"`, `"14B"`, `"32B"`.

### `src/write.py` (made by S2)

```python
def write_dose(base_model: nn.Module, layer_idx: int, vector: torch.Tensor, dose: float) -> nn.Module
    # UNCHANGED behaviour: bias = dose * vector, literal multiplier.

def save_write(path: str | Path, size: str, layer_idx: int, vector: torch.Tensor,
               dose: float, coefficient: float) -> None

def load_write(path: str | Path) -> dict

def load_generation(base_model: nn.Module, size: str, vector: torch.Tensor, dose: float,
                    prereg_path: Path = PREREG_PATH,
                    write_path: str | Path | None = None) -> nn.Module
```

`PREREG_PATH`, `working_coefficient`, `injection_layer` and
`UncalibratedError` are imported from `src.models` (`src/models.py:16, 19,
82, 97`). `src/models.py` is not changed.

### Write file format

`torch.save` of a dict with exactly these keys
(brief, "Decisions made"):

| key | type | meaning |
|---|---|---|
| `"size"` | `str` | `"8B"`, `"14B"` or `"32B"` |
| `"layer"` | `int` | injection layer the write went into |
| `"dose"` | `float` | dose in working-coefficient units (1.0 = working dose) |
| `"coefficient"` | `float` | absolute multiplier applied to the raw vector, `dose * working_coefficient(size)` |
| `"vector"` | CPU `torch.Tensor` | raw vector |

Loaded with `torch.load(path, map_location="cpu", weights_only=True)`.

## Subtasks

### S1: temporary prereg helper for tests
- Owns: `tests/prereg_tmp.py`, `tests/test_prereg_tmp.py`
- Depends on: none
- Do:
  - Create `tests/prereg_tmp.py` exactly matching the Interfaces section.
    `write_prereg` does `path = Path(tmp_dir) / "models.json"`,
    `path.write_text(json.dumps(entries), encoding="utf-8")`, returns `path`.
    Put a comment on the two constants saying they are arbitrary fixture
    values, not sourced numbers (same style as `tests/tiny_qwen3.py:8`).
    Module docstring and type hints on the function.
  - This mirrors the `write`/`full` helpers in `tests/test_models.py:25-32`;
    do not edit `tests/test_models.py`.
  - Create `tests/test_prereg_tmp.py` with CPU tests only.
- Done when:
  - `tests/test_prereg_tmp.py` has a test that writes
    `{"8B": {"injection_layer": FIXTURE_LAYER, "working_coefficient": FIXTURE_COEFFICIENT}}`
    via `write_prereg(tmp_path, ...)` and asserts
    `src.models.working_coefficient("8B", path) == FIXTURE_COEFFICIENT` and
    `src.models.injection_layer("8B", path) == FIXTURE_LAYER`.
  - A second test asserts `FIXTURE_LAYER < tests.tiny_qwen3.N_LAYERS`
    (brief trap "Layer from the file": tiny model has 3 layers,
    `tests/tiny_qwen3.py:11`).
  - `python -m pytest tests/test_prereg_tmp.py -q` passes and
    `./scripts/check.sh` exits 0.

### S2: dose-to-coefficient conversion in load_generation, five-key write file
- Owns: `src/write.py`, `tests/test_write.py`
- Depends on: S1 (uses only the Interfaces of `tests/prereg_tmp.py`)
- Do:
  - `write_dose`: leave the code unchanged (request). You may add one
    docstring sentence that `load_generation` is the only place that
    converts a dose to a coefficient.
  - `_WRITE_KEYS = {"size", "layer", "dose", "coefficient", "vector"}`.
  - `save_write(path, size, layer_idx, vector, dose, coefficient)`: save
    `{"size": str(size), "layer": int(layer_idx), "dose": float(dose),
    "coefficient": float(coefficient), "vector": vector.detach().cpu()}`.
    It must NOT call `working_coefficient` or check the coefficient
    (that would be a second conversion point; brief "Decisions made").
    Docstring: `dose` is in working-coefficient units, `coefficient` is the
    absolute multiplier applied; never use `save_pretrained` for a written
    model.
  - `load_write(path)`: load with `weights_only=True`. Raise `ValueError`
    unless the result is a dict whose key set equals `_WRITE_KEYS`. Then,
    for each of `"dose"` and `"coefficient"`, raise `ValueError` (message
    naming the key) if the value is `None`, a `bool`, or not an
    `int`/`float`. Return the dict. No range checks, no prereg reads.
  - `load_generation(base_model, size, vector, dose, prereg_path=PREREG_PATH, write_path=None)`:
    1. `c = working_coefficient(size, prereg_path)` (read the coefficient
       FIRST; brief "Decisions made").
    2. `layer = injection_layer(size, prereg_path)`.
    3. `coefficient = dose * c`; `write_dose(base_model, layer, vector, coefficient)`.
    4. If `write_path is not None`: `save_write(write_path, size, layer, vector, dose, coefficient)`.
    5. Return `base_model`.
    Do not catch or wrap `UncalibratedError`, `KeyError` or `ValueError`
    from `src.models`. Do not load an HF model, attach an adapter, or
    check a dose floor or cap (out of scope). Docstring: takes an
    already-loaded base model; `dose` is in working-coefficient units
    (dose 1.0 = the working coefficient, `docs/MVP.md:118-120`); this is
    the only place in the repo that converts dose to coefficient; attach
    the adapter after calling it.
  - `tests/test_write.py`: keep every existing `write_dose` test as is.
    Rewrite `test_save_load_roundtrip` and `test_load_write_bad_keys` for
    the new format (brief trap "Old test": the old three-key cases would
    pass silently). Every new test passes `prereg_path` built with
    `write_prereg(tmp_path, ...)`; never rely on the committed
    `prereg/models.json` (all null, `prereg/models.json:1-5`; brief trap
    "Default prereg path in tests"). Use `make_tiny_model`, `_vector()`
    and `FIXTURE_LAYER`, `FIXTURE_COEFFICIENT`.
- Done when (tests to write, all CPU, no `gpu` mark):
  - `test_load_generation_dose_one`: prereg 8B =
    `{"injection_layer": FIXTURE_LAYER, "working_coefficient": FIXTURE_COEFFICIENT}`.
    `load_generation(make_tiny_model(), "8B", v, 1.0, prereg_path=p)` has
    `down_proj.bias` at `FIXTURE_LAYER` `torch.equal` to that of
    `write_dose(make_tiny_model(), FIXTURE_LAYER, v, FIXTURE_COEFFICIENT)`,
    and forward outputs on `_inputs()` are `torch.equal`. This also checks
    brief INFERRED finding 31 (layer 1 from a temp prereg passes the 8B
    range check and indexes the 3-layer tiny model).
  - `test_load_generation_dose_half`: at dose 0.5 the bias is `torch.equal`
    to `write_dose(..., 0.5 * FIXTURE_COEFFICIENT)` bias AND to `0.5 *` the
    dose-1.0 bias from `load_generation`; forward outputs `torch.equal` to
    the `write_dose(..., 0.5 * FIXTURE_COEFFICIENT)` model. This is the
    brief's check for INFERRED finding 32 (`torch.equal`, exact).
  - `test_load_generation_uncalibrated_coefficient`: prereg with 8B set and
    14B `{"injection_layer": FIXTURE_LAYER, "working_coefficient": None}`.
    `pytest.raises(UncalibratedError, match="14B working_coefficient is not set")`
    and `excinfo.type is UncalibratedError` (unchanged, not wrapped); the
    model's `down_proj.bias` at `FIXTURE_LAYER` is still `None`.
  - `test_load_generation_uncalibrated_layer`: 14B with
    `injection_layer: None` and coefficient set raises `UncalibratedError`
    matching `"14B injection_layer is not set"`, type exactly
    `UncalibratedError`.
  - `test_load_generation_saves_write`: with `write_path=tmp_path / "w.pt"`
    at dose 0.5, `load_write` returns `size == "8B"`,
    `layer == FIXTURE_LAYER`, `dose == 0.5`,
    `coefficient == 0.5 * FIXTURE_COEFFICIENT`, vector `torch.equal` to `v`.
  - `test_save_load_roundtrip` (rewritten): `save_write(path, "8B",
    FIXTURE_LAYER, v, 0.5, 0.5 * FIXTURE_COEFFICIENT)` then `load_write`
    returns all five values equal to what was saved, with key set exactly
    `{"size", "layer", "dose", "coefficient", "vector"}`. This is the
    brief's check for INFERRED finding 33 (`weights_only=True` accepts str
    and float entries).
  - `test_load_write_bad_keys` (rewritten): starting from a valid five-key
    dict saved with `torch.save`, each of these raises `ValueError`: an
    extra key; `"dose"` removed; `"coefficient"` removed; `"size"` removed.
  - `test_load_write_bad_numbers`: five-key files with `dose=None`,
    `coefficient=None`, `dose=True`, `coefficient="2.5"` each raise
    `ValueError`.
  - Reviewer reads: `grep -rn "working_coefficient" src/` hits only
    `src/models.py` and the one call inside `load_generation`;
    `write_dose`'s body is unchanged from `dc0c1b1`; `src/models.py` and
    `prereg/models.json` are unchanged.
  - `python -m pytest tests/test_write.py -q` passes and
    `./scripts/check.sh` exits 0.
