# Research: one dose-to-coefficient conversion in load_generation, and write files that record both

Sources read: this repo at `dc0c1b1` (`CLAUDE.md`, `docs/MVP.md`, `docs/PROPOSAL.md`, `research/feature-write.md`, `src/write.py`, `src/models.py`, `prereg/models.json`, `tests/tiny_qwen3.py`, `tests/test_write.py`, `tests/test_models.py`, `tests/test_prereg_models_file.py`, `scripts/check.sh`). Authors' repo `third_party/Pain-axis` at `213fb6c89c96004dbe93d1704c636ad8c9d801f3` (not needed beyond the prior brief: the feature is internal plumbing). Paper not needed. torch 2.15.0 in the main checkout's `.venv` (the worktree has none; `scripts/check.sh:17-25` falls back to it).

Serves: MVP output 2 (the write proof) and every later run that turns a dose into weights. Stage: `vector + dose -> write (weights)`.

## Questions
1. What do `src/write.py` and `src/models.py` do now, and what does models.py raise for an uncalibrated size?
2. How can a test point models.py at a temporary `prereg/models.json`?
3. What is the current write-file format, and who reads it?
4. What does `docs/MVP.md` Step 2 say `load_generation` does?
5. What do `tests/tiny_qwen3.py` and `tests/test_write.py` provide?
6. Which signature for `load_generation` and `save_write` satisfies the request with the least, and keeps exactly one conversion point?

## Findings
- [READ] `write_dose(base_model: nn.Module, layer_idx: int, vector: torch.Tensor, dose: float) -> nn.Module` replaces `model.model.layers[layer_idx].mlp.down_proj` with a biased `nn.Linear`, bias = `(dose * vector).to(new.bias)`, in place, returns `base_model`. Raises TypeError if down_proj is not exactly `nn.Linear`, ValueError if it already has a bias or the vector shape is not `(out_features,)`. Docstring already says "`dose` multiplies `vector` literally. Converting from working-coefficient units is the caller's job." Source: `src/write.py:13-52`
- [READ] `save_write(path: str | Path, layer_idx: int, vector: torch.Tensor, dose: float) -> None` does `torch.save({"layer": int(layer_idx), "dose": float(dose), "vector": vector.detach().cpu()}, path)`. Source: `src/write.py:55-67`
- [READ] `load_write(path: str | Path) -> dict` does `torch.load(path, map_location="cpu", weights_only=True)` and raises ValueError unless the result is a dict whose key set equals `_WRITE_KEYS = {"layer", "dose", "vector"}`. Source: `src/write.py:10, 70-75`
- [READ] No `load_generation` exists anywhere yet; nothing outside `src/write.py` and `tests/test_write.py` calls `write_dose`, `save_write` or `load_write`. Source: `git grep` over `src tests scripts .claude` (no hits). There is no `runs/` directory, so no existing write files would be invalidated by a format change. Source: `ls runs` (absent)
- [READ] `src/models.py`: `PREREG_PATH = Path(__file__).resolve().parent.parent / "prereg" / "models.json"` (module constant, line 16). `injection_layer(size: str, prereg_path: Path = PREREG_PATH) -> int` and `working_coefficient(size: str, prereg_path: Path = PREREG_PATH) -> float` re-read the file on every call. Source: `src/models.py:16, 82-108`
- [READ] Uncalibrated error: `class UncalibratedError(LookupError)`. `_read_field` raises `UncalibratedError(f"{size} {field} is not set in {prereg_path}")` when the file is missing, the size entry is missing, or the field is absent or null. Source: `src/models.py:19-20, 68-79`
- [READ] Other models.py errors: unknown size -> `KeyError(f"unknown size {size!r}; valid sizes: ...")` from `get_spec` (called first by both readers). Bad coefficient (bool, non-number, non-finite, <= 0) -> ValueError. Layer outside `0..num_layers-1` of the real size (8B: 36, 14B: 40, 32B: 64) or non-int -> ValueError. Source: `src/models.py:36-65, 89-108`
- [READ] Tests point models.py at a temp file by passing `prereg_path` explicitly; `tests/test_models.py` writes `tmp_path / "models.json"` with helpers `write(tmp_path, data)` and `full(layer, coef)`. No monkeypatching is used. Source: `tests/test_models.py:25-32, 62-81`
- [READ] `prereg/models.json` currently holds `null` for both `injection_layer` and `working_coefficient` at `8B`, `14B` and `32B`. Source: `prereg/models.json:1-5`. `tests/test_prereg_models_file.py` asserts that schema and that both readers raise UncalibratedError while all are null. Source: `tests/test_prereg_models_file.py:18-35`
- [READ] MVP Step 2: write file is `torch.save({"layer": L, "dose": d, "vector": v.cpu()}, path)`. "To load a generation, load the base model, call `write_dose`, then attach the adapter." Source: `docs/MVP.md:101`. Layout lists `src/write.py  write_dose, load_generation`. Source: `docs/MVP.md:26`
- [READ] "From here on, **dose** means a multiple of that size's working coefficient: dose 1.0 is the working coefficient." Source: `docs/MVP.md:118-120`. Same in `CLAUDE.md` Vocabulary ("Dose").
- [READ] Doses used later: 0.25 to 1.5 in steps, lineage steps ±0.25, floor 0, cap 1.5. Source: `docs/PROPOSAL.md:125, 130-132, 182`
- [READ] `tests/tiny_qwen3.py` provides `HIDDEN=16`, `INTERMEDIATE=32`, `N_LAYERS=3`, `VOCAB=50` (fixture values, unsourced by design), `make_tiny_model(seed=0, return_tuple=True)` (seeded fp32 CPU eval model with `model.model.layers[i].mlp.down_proj` bias-free), and `add_steering_hook(model, layer_idx, vector, dose)`. Source: `tests/tiny_qwen3.py:8-97`
- [READ] `tests/test_write.py` uses `LAYER = 1`, `_inputs()` (seed 123, shape `(2, 5, HIDDEN)`), `_vector()` (seed 7, shape `(HIDDEN,)`), tolerance `atol=1e-5, rtol=1e-5` (marked as chosen). It has `test_save_load_roundtrip` and `test_load_write_bad_keys`, which assume the old three-key format and must change. Source: `tests/test_write.py:12-22, 111-131`
- [READ] The prior brief left as LATER "Whether the stored "dose" ... is in working-coefficient units or absolute. Recording both (`dose` and `coeff`) ... would remove the ambiguity." Source: `research/feature-write.md:79`
- [INFERRED] The tiny model has 3 layers but `injection_layer` validates against the real size (8B allows 0..35), so a temp prereg with `injection_layer: 1` passes validation and indexes the tiny model fine. From: `src/models.py:89-94`, `tests/tiny_qwen3.py:11`. Check by: the load_generation tests.
- [INFERRED] `load_generation` at dose 0.5 with coefficient c yields a bias bit-equal to `write_dose(..., 0.5 * c)`, and to half the dose-1.0 bias, because multiplying by 0.5 is exact in binary float (no overflow/underflow at these magnitudes). From: `src/write.py:50`. Check by: `torch.equal` on the biases.
- [INFERRED] `torch.load(..., weights_only=True)` accepts a dict of str, int, float and tensor, so adding a `"size"` string and `"coefficient"` float keeps the safe loader. From: torch 2.15.0 installed; not checked in docs. Check by: the round-trip test.

## Reuse
- From this repo: `write_dose` unchanged (request says keep it). `models.working_coefficient` and `models.injection_layer` imported, with `prereg_path` passed through. `UncalibratedError` propagated, never caught or wrapped.
- Test helpers: `make_tiny_model`, `HIDDEN` from `tests/tiny_qwen3.py`; mirror the `write`/`full` temp-prereg pattern of `tests/test_models.py:25-32` (rewrite locally or import; both are trivial).
- Nothing from the authors' code.

## Expected values
- dose 1.0: `load_generation` bias == `write_dose(..., c)` bias exactly, c = the temp file's coefficient (fixture value).
- dose 0.5: bias == `0.5 * c * vector` exactly; forward output equal to `write_dose(..., 0.5 * c)` model output (`torch.equal`, same seed, same ops).
- Uncalibrated size (coefficient or layer null in temp file): `pytest.raises(UncalibratedError)` with the message `"<size> working_coefficient is not set in <path>"` (or `injection_layer`, depending on read order).
- Round trip: `load_write(path)` returns `size`, `layer`, `dose`, `coefficient`, `vector` equal to what was saved.

## New here
The authors steer with one absolute coefficient (1.0 for Qwen 2.5, per `docs/MVP.md:18`). Per-size working-coefficient units, and a write file that records both units, are this project's.

## Traps
- **Double conversion.** If any caller multiplies by the coefficient before calling `load_generation`, or passes an absolute value as `dose`, every generation is mis-scaled with no error. Only `load_generation` may call `working_coefficient`; `write_dose` and `save_write` must not.
- **File inconsistency.** If `save_write` takes `dose` and `coefficient` as two free arguments, a caller can record a pair that does not match what was written. Have `load_generation` save the file itself (see Decisions).
- **Catching the error.** Wrapping `UncalibratedError` in another type breaks the "raises the models.py error unchanged" test and hides the prereg path in the message.
- **Default prereg path in tests.** Forgetting to pass `prereg_path` makes the test read the committed file, which is all null, so it raises UncalibratedError and a "raises" test would pass for the wrong reason. Use a temp file in every test, including the uncalibrated one (null for one size, set for another).
- **Layer from the file.** Tiny-model layer must be < 3; a temp `injection_layer` of 3 or more passes models.py validation but crashes on indexing.
- **Old test.** `test_load_write_bad_keys` builds three-key files; with the new format both of its cases still fail, so it passes silently unless rewritten for the new keys.

## Decisions made
- `load_generation(base_model: nn.Module, size: str, vector: torch.Tensor, dose: float, prereg_path: Path = PREREG_PATH, write_path: str | Path | None = None) -> nn.Module`. It takes an already-loaded base model (so it runs on CPU with the tiny model; HF loading and adapter attachment from `docs/MVP.md:101` are left to the caller and out of scope). It reads `layer = injection_layer(size, prereg_path)` and `c = working_coefficient(size, prereg_path)`, calls `write_dose(base_model, layer, vector, dose * c)`, and, if `write_path` is given, calls `save_write` with size, layer, dose, `dose * c` and vector. Why: the layer is also per-size frozen (`CLAUDE.md` Rules), and saving inside the one converter keeps the two recorded numbers consistent.
- Read the coefficient before the layer, so an uncalibrated-coefficient test sees `working_coefficient is not set`. Either order raises UncalibratedError.
- `save_write(path, size: str, layer_idx: int, vector: torch.Tensor, dose: float, coefficient: float) -> None`, keys exactly `{"size", "layer", "dose", "coefficient", "vector"}`. It does not compute or check `coefficient` against prereg (that would be a second conversion point). Docstring: `dose` in working-coefficient units, `coefficient` the absolute multiplier applied.
- `load_write` keeps the exact-key-set check against the new five keys, and additionally refuses `dose` or `coefficient` that is None, a bool, or not an int/float, so "missing either number" covers both an absent key and a null. Raises ValueError (as now).
- No range check on dose (0 to 1.5) here; not requested.
- Tests: new cases go in `tests/test_write.py` (the module under test is `src/write.py`; `CLAUDE.md` Standards). The fixture coefficient in the temp prereg is an arbitrary test value, commented as such (like `tests/tiny_qwen3.py:8`); a non-power-of-two such as 2.5 exercises real scaling.
- Leave `prereg/models.json` untouched.

## Open questions
- [LATER] `docs/MVP.md:101` still shows the three-key write file; update it to the five-key format.
- [LATER] Should `load_write` also check `coefficient == dose * working_coefficient(size)` when re-loading a generation? That needs the prereg and would be a second reader, so it is left out.
- [LATER] Whether `load_generation` should later load the HF model and attach the adapter itself (MVP wording), once adapters exist.
- [LATER] Whether `load_generation` should enforce the dose cap 1.5 and floor 0 (`docs/PROPOSAL.md:132`).
