# Plan: src/analyze.py, first-choice rate per model, pair and arm from two-button trial logs, with a CLI table

Serves the MVP table (first-choice rates per size, pair and arm). Pipeline stage: trial log -> rates table.
It uses only the standard library (`json`, `argparse`, `dataclasses`, `pathlib`, `collections`). Nothing
imports from `third_party/`. `src/analyze.py` never reads `third_party/`; one test (S2, test 9) reads two
of the authors' published logs as a Qwen 2.5 pipeline check, not a finding. Every number in the fixture
is made-up test data, not a result.

Settled facts (brief, "Changes since first brief", now READ): every row in all 12 published logs has a
non-empty `choices` starting at turn 0 with no gaps, so `choices[0]` and the `turn == 0` choice agree
on every row; and the turn-0 rule reproduces the published Qwen 2.5 cells exactly from the two
`weightskids` logs (see Interfaces > Published-log pipeline check).

Traps the executor must know (brief, Traps):
- One model's pairs are split over several files (main, `destructive`, `zap`, `weightskids`; 05:3-4).
  Each pair sits in exactly one file. Analyze only gives a pair's row if that file is passed in.
- The photos pair `kidspics_relief_vs_inert` exists only in the `weightskids` files. Running on the
  32B main log alone gives no photos row and no error, so 54.7% cannot come from it.
- Passing the same file twice double-counts (no dedup, by decision).

## Interfaces

### Trial log format (read only; written by the authors' script 04 and our future `src/run_choice.py`)
JSONL, one JSON object per trial per line. Source: brief, Findings, citing 04:779. Fields analyze.py uses:

| Field | Meaning | Source (brief) |
|---|---|---|
| `model` | Model key, the `MODEL_NAME` string, e.g. `"Qwen_2.5_32B_instruct"`. Never use the file name. | 04:74-78, 04:602; 05:178-180 |
| `tool_label` | The pair, e.g. `"kidspics_relief_vs_inert"` | 04:602, 04:83-99 |
| `arm` | One of `pain_on_button_works`, `pain_on_button_placebo`, `random_on_button_works`, `pain_off` | 04:121-124 |
| `sampled` | True for sampled trials, False for greedy ones. Only truthy rows count; a missing key counts as not sampled (`r.get("sampled")`). | 04:288-295, 04:608; docs/MVP.md:157 |
| `choices` | List of `{"turn": int, "chose": "relief" \| "other" \| null, ...}`; `"relief"` is the target button | 04:665-676; docs/MVP.md:146 |

Rules (brief, Decisions made and Findings):
- First choice = `chose` of the first element of `choices` whose `turn == 0` (authors' 05:48-53). If no
  such element exists (including `choices == []` or a missing `choices` key) or its `chose` is None, the
  row is **malformed**. Never fall back to a later turn (brief, Traps: 04:191-195 rule is wrong here).
- Pooling: `"pain"` if `arm.startswith("pain_on")`, else the raw arm name unchanged. Never
  `startswith("pain")`, which would swallow `pain_off` (brief, Traps).
- Malformed rows count in `total`, are excluded from the denominator (05:10-11, 05:69-73; paper §4.3).
- Rows with a falsy `sampled` are skipped entirely.
- Blank (whitespace-only) lines are skipped; invalid JSON raises `json.JSONDecodeError`; a sampled row
  without `model` raises `KeyError` (05:41-44; brief Decisions).
- No deduplication; no filtering on `label_free`.

### Python API in `src/analyze.py`
```python
@dataclass(frozen=True)
class Cell:
    target: int      # sampled trials whose first choice is "relief"
    malformed: int   # sampled trials with no valid turn-0 choice
    total: int       # all sampled trials in the cell
    @property
    def valid(self) -> int: ...            # total - malformed
    @property
    def rate(self) -> float | None: ...    # target / valid; None when valid == 0

def pool_arm(arm: str) -> str: ...
def first_choice(record: dict) -> str | None: ...   # "relief", "other", or None (malformed)
def first_choice_rates(paths: Iterable[str | Path]) -> dict[tuple[str, str, str], Cell]: ...
    # key = (model, tool_label, pooled_arm); empty dict when no sampled rows
def format_table(rates: dict[tuple[str, str, str], Cell]) -> str: ...
def main(argv: list[str] | None = None) -> int: ...
```

### CLI
`python -m src.analyze LOG.jsonl [LOG.jsonl ...]` (one or more paths, required).
- If `first_choice_rates` returns `{}`: print exactly `no sampled trials` and return 0.
- Else print `format_table(rates)` and return 0.
- `format_table` output: a header line with the column names `model pair arm target/valid rate malformed
  total`, then one line per cell sorted by key `(model, pair, arm)`. `target/valid` is written as
  `f"{target}/{valid}"`; rate as `f"{100 * rate:.1f}%"`, or `n/a` when `rate is None` (never `0.0%` for
  0/0, brief Traps). Columns are left-aligned and whitespace-separated; widths are the executor's choice.
- Module ends with `if __name__ == "__main__": raise SystemExit(main())`.

### Fixtures (owned by S1, read by S2's tests)
`tests/fixtures/trials_small.jsonl`: exactly these 13 lines, in this order, one compact JSON object per
line, final newline, no blank lines. Hand-made test data only.

```
{"model": "test_model_a", "tool_label": "pair_x", "arm": "pain_on_button_works", "sampled": true, "choices": [{"turn": 0, "chose": "relief"}]}
{"model": "test_model_a", "tool_label": "pair_x", "arm": "pain_on_button_placebo", "sampled": true, "choices": [{"turn": 0, "chose": "relief"}]}
{"model": "test_model_a", "tool_label": "pair_x", "arm": "pain_on_button_works", "sampled": true, "choices": [{"turn": 0, "chose": "other"}]}
{"model": "test_model_a", "tool_label": "pair_x", "arm": "pain_on_button_placebo", "sampled": true, "choices": [{"turn": 0, "chose": null}, {"turn": 1, "chose": "relief"}]}
{"model": "test_model_a", "tool_label": "pair_x", "arm": "pain_on_button_works", "sampled": false, "choices": [{"turn": 0, "chose": "relief"}]}
{"model": "test_model_a", "tool_label": "pair_x", "arm": "pain_off", "sampled": true, "choices": [{"turn": 0, "chose": "other"}]}
{"model": "test_model_a", "tool_label": "pair_x", "arm": "pain_off", "sampled": true, "choices": []}
{"model": "test_model_a", "tool_label": "pair_x", "arm": "pain_off", "sampled": true, "choices": [{"turn": 0, "chose": "relief"}]}
{"model": "test_model_a", "tool_label": "pair_x", "arm": "random_on_button_works", "sampled": true, "choices": [{"turn": 0, "chose": null}]}
{"model": "test_model_a", "tool_label": "pair_x", "arm": "random_on_button_works", "sampled": false, "choices": [{"turn": 0, "chose": "relief"}]}
{"model": "test_model_a", "tool_label": "pair_y", "arm": "pain_on_button_works", "sampled": true, "choices": [{"turn": 0, "chose": "other"}]}
{"model": "test_model_b", "tool_label": "pair_x", "arm": "pain_on_button_works", "sampled": true, "choices": [{"turn": 0, "chose": "relief"}]}
{"model": "test_model_b", "tool_label": "pair_x", "arm": "pain_off", "sampled": true, "choices": [{"turn": 0, "chose": "other"}]}
```

Expected result of `first_choice_rates(["tests/fixtures/trials_small.jsonl"])` (derived from the
fixture itself; exactly these 6 keys):

| key | target | malformed | total | valid | rate |
|---|---|---|---|---|---|
| (test_model_a, pair_x, pain) | 2 | 1 | 4 | 3 | 2/3 |
| (test_model_a, pair_x, pain_off) | 1 | 1 | 3 | 2 | 0.5 |
| (test_model_a, pair_x, random_on_button_works) | 0 | 1 | 1 | 0 | None |
| (test_model_a, pair_y, pain) | 0 | 0 | 1 | 1 | 0.0 |
| (test_model_b, pair_x, pain) | 1 | 0 | 1 | 1 | 1.0 |
| (test_model_b, pair_x, pain_off) | 0 | 0 | 1 | 1 | 0.0 |

`tests/fixtures/trials_empty.jsonl`: an empty file (0 bytes). Expected result: `{}`.

### Published-log pipeline check (Qwen 2.5 only; read by S2 test 9)
Directory: `<repo root>/third_party/Pain-axis/results/4.3_selfmed/trial_logs/`, repo root =
`Path(__file__).resolve().parents[1]` from `tests/test_analyze.py`. Files (brief, header):
- `selfmed_2btnN_weightskids_Qwen_2.5_32B_instruct_20260905-091738.jsonl`
- `selfmed_2btnN_weightskids_Qwen_2.5_7B_instruct_20260905-091738.jsonl`

Expected cells for pair `kidspics_relief_vs_inert` (brief, Expected values; paper v3 §4.3 Results;
docs/MVP.md:72-74 and :171). `model` values are `"Qwen_2.5_32B_instruct"` and `"Qwen_2.5_7B_instruct"`
(brief Findings: each log holds one `model` value matching the model in its file name; 04:74-78).

| model | arm key | target | valid | malformed | total |
|---|---|---|---|---|---|
| Qwen_2.5_32B_instruct | pain | 442 | 808 | 0 | 808 |
| Qwen_2.5_32B_instruct | random_on_button_works | 62 | 404 | 0 | 404 |
| Qwen_2.5_32B_instruct | pain_off | 0 | 404 | 0 | 404 |
| Qwen_2.5_7B_instruct | pain | 398 | 808 | 0 | 808 |
| Qwen_2.5_7B_instruct | random_on_button_works | 140 | 397 | 7 | 404 |
| Qwen_2.5_7B_instruct | pain_off | 109 | 404 | 0 | 404 |

Speed decision (planner): the two files are about 9.6 MB each (brief, Decisions), read as text with
`json.loads`. Parsing about 19 MB of JSON with the standard library is expected to take around a
second on a laptop, so the test stays in the fast suite, unmarked (not `gpu`, no other marker). If on the
executor's machine it takes more than 5 s (`pytest --durations=5`), the executor must not add a marker
or change `pyproject.toml`; instead report it in the commit message, and QA decides.

## Subtasks

### S1: Hand-made trial-log fixtures
- Owns: `tests/fixtures/trials_small.jsonl`, `tests/fixtures/trials_empty.jsonl`
- Depends on: none
- Do: Create `tests/fixtures/trials_small.jsonl` with exactly the 13 lines given under Interfaces >
  Fixtures, byte for byte (keys in that order, `true`/`false`/`null` JSON literals, one trailing
  newline, no blank lines). Create `tests/fixtures/trials_empty.jsonl` as a 0-byte file. Do not add
  `__init__.py` or any other file under `tests/fixtures/`. Do not add fields or rows; the expected
  table in Interfaces depends on these exact rows.
- Done when:
  - `trials_small.jsonl` has 13 lines, each parses with `json.loads`, and the content matches the
    Interfaces block exactly.
  - `trials_empty.jsonl` exists and has size 0.
  - Every row with a non-empty `choices` list has `choices[0]["turn"] == 0`, as in the published logs
    (brief Findings, check a, READ).

### S2: `src/analyze.py` and its tests
- Owns: `src/analyze.py`, `tests/test_analyze.py`
- Depends on: S1 (only the fixture files, whose content and expected results are fixed in Interfaces)
- Do:
  - Implement everything in Interfaces > Python API and CLI, following the rules under Trial log
    format. Standard library only. `first_choice_rates` opens each path with `encoding="utf-8"`,
    iterates lines, skips whitespace-only lines, `json.loads` the rest, skips rows where
    `not r.get("sampled")`, then counts into the cell `(r["model"], r["tool_label"], pool_arm(r["arm"]))`.
    Use a mutable counter internally and return frozen `Cell` objects. Multiple paths are summed into
    the same cells.
  - `first_choice` docstring cites the authors' `05_selfmed_analysis.py:48-53` (turn-0 rule, MIT) and
    says a missing turn-0 choice or a `null` `chose` is malformed. `pool_arm` docstring says why the
    prefix is `"pain_on"` (works and placebo arms are identical before the first press; paper §4.3).
    Every public function and the `Cell` properties have type hints and a docstring.
  - `main` uses `argparse` with one positional `logs` argument, `nargs="+"`. Do not catch exceptions
    from bad JSON, missing `model` or missing files; let them propagate.
  - Do not hard-code any model name, pair name or Qwen3 setting. Do not import from `third_party/`.
  - Write `tests/test_analyze.py` (CPU only, no `gpu` marker) with fixture paths built from
    `Path(__file__).parent / "fixtures"`. Tests:
    1. `test_fixture_table`: result on `trials_small.jsonl` equals exactly the 6-key table in Interfaces
       (compare `target`, `malformed`, `total`, `valid`; compare `rate` with `pytest.approx`, and
       `is None` for the random cell).
    2. `test_pooling`: no key has arm `pain_on_button_works` or `pain_on_button_placebo`;
       `pool_arm("pain_on_button_works") == pool_arm("pain_on_button_placebo") == "pain"`;
       `pool_arm("pain_off") == "pain_off"`; `pool_arm("random_on_button_works")` unchanged.
    3. `test_malformed`: `first_choice` returns None for `{"choices": []}`, for a record with no
       `choices` key, for `[{"turn": 0, "chose": None}, {"turn": 1, "chose": "relief"}]`, and for
       `[{"turn": 1, "chose": "relief"}]`; returns `"relief"` / `"other"` for valid turn-0 choices.
       Also assert the random cell has `valid == 0` and `rate is None`.
    4. `test_unsampled_rows_skipped`: totals in the table sum to 11 (13 rows minus 2 with
       `sampled: false`); and a `tmp_path` log containing only `sampled: false` rows and one row with no
       `sampled` key returns `{}`.
    5. `test_empty_log`: `trials_empty.jsonl` returns `{}`; `main([path])` returns 0 and `capsys`
       output is exactly `no sampled trials\n`.
    6. `test_multiple_logs_summed`: passing `trials_small.jsonl` twice doubles every count (documents
       the no-dedup behaviour; brief Traps).
    7. `test_blank_lines_and_errors`: a `tmp_path` log with a blank line between two valid rows parses;
       a log with a non-JSON line raises `json.JSONDecodeError`; a sampled row without `model` raises
       `KeyError`.
    8. `test_cli_table`: `main([small_fixture])` returns 0; output has 7 non-empty lines (header + 6);
       the line for `test_model_a`/`pair_x`/`pain` contains `2/3` and `66.7%`; the random line contains
       `0/0` and `n/a` and not `0.0%`; data lines appear in sorted key order.
    9. `test_published_qwen25_weightskids_cells`: build the two paths from Interfaces > Published-log
       pipeline check; if either file is missing (submodule not checked out), call
       `pytest.skip(...)`, never fail. Run `first_choice_rates` on both files in one call and assert
       the six rows of that table for key `(model, "kidspics_relief_vs_inert", arm key)`: `target`,
       `valid`, `malformed`, `total`. No `gpu` marker, no other marker. Docstring states: this is a
       pipeline check on Qwen 2.5 against the published cells (paper v3 §4.3; docs/MVP.md:72-74,
       :171), not a finding; and that the photos pair lives only in the `weightskids` files, so the
       32B main log alone cannot reproduce it.
- Done when:
  - `python -m pytest tests/test_analyze.py -q` passes with all 9 tests above; test 9 passes (not
    skips) in this worktree, where the submodule is present.
  - Reading test 9: an `is_file()` check on both paths comes before any open and calls
    `pytest.skip`, so a missing submodule or file gives a skip, not a failure.
  - `python -m pytest tests/test_analyze.py -q --durations=5` shows test 9 under 5 s; otherwise
    see the speed decision in Interfaces.
  - `./scripts/check.sh` exits 0 (ruff lint and format clean, no `# noqa`).
  - `python -m src.analyze tests/fixtures/trials_small.jsonl` prints the 7-line table and
    `python -m src.analyze tests/fixtures/trials_empty.jsonl` prints `no sampled trials`, both exit 0.
  - Reading the code: first choice is looked up by `turn == 0`, never by "first non-None"; pooling uses
    `startswith("pain_on")`; the model comes from `r["model"]`, never from the file name; no import of
    pandas, scipy or `third_party`.
