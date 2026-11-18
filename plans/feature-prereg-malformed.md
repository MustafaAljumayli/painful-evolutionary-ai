# Plan: Freeze the first-choice counting rule (sampled, malformed, denominator, pain pooling) in prereg/ and test it against src/analyze.py

Serves the MVP table (docs/MVP.md Steps 4 and 6) and the Step 5 gate; touches the
"trial log -> rates table" stage. `src/analyze.py` is not edited by any subtask.

Settled scope (orchestrator, from the docs; treat as fact): the file covers only the
first-choice rates computed by `src/analyze.py`. It does NOT close the TBD at
`docs/PROPOSAL.md:163` for any other analysis (post-MVP doses from the authors' later
profile runtime, `docs/MVP.md:200`; lineages; anything not produced by analyze.py).
`docs/PROPOSAL.md` and `docs/MVP.md` are not edited.

## Interfaces

### File `prereg/first_choice_rule.json`

Plain JSON, 2-space indent, UTF-8, trailing newline (convention of `prereg/models.json`,
brief "Prereg conventions"). Exactly these top-level keys, no others:

| Key | Type | Fixed value |
|---|---|---|
| `"scope"` | string | free text, see S1 |
| `"not_covered"` | string | free text, see S1 |
| `"left_open"` | object `{"path": str, "quote": str}` | `{"path": "docs/PROPOSAL.md", "quote": "One denominator rule throughout (TBD; the authors used two across studies)."}` |
| `"sampled_field"` | string | `"sampled"` |
| `"first_choice_turn"` | integer | `0` |
| `"target_value"` | string | `"relief"` |
| `"denominator"` | string | `"total_minus_malformed"` |
| `"rate_when_no_valid"` | null | `null` |
| `"malformed_reported"` | bool | `true` |
| `"pool_prefix"` | string | `"pain_on"` |
| `"pooled_arm"` | string | `"pain"` |
| `"rules"` | object | exactly the keys `"sampled"`, `"first_choice"`, `"malformed"`, `"denominator"`, `"pooling"`, `"chose_values"`, `"not_covered"` |

Each value in `"rules"` is `{"text": str, "sources": [{"path": str, "quote": str}, ...]}`
with at least one source. `path` is relative to the repo root. `quote` is an exact
excerpt of that file; it may span a line break in the source, so the comparison
normalises whitespace (see below). Only these files may be quoted: `docs/MVP.md`,
`docs/PROPOSAL.md`, and files under `third_party/Pain-axis/` (orchestrator rule 4).
No quotes from `docs/Pain_Axis.pdf`.

### Quote-matching rule (used by the test)

`" ".join(text.split())` applied to both the quote and the whole source file; the
normalised quote must be a substring of the normalised file.

### Machine-readable semantics (what the test recomputes from the fields)

For each non-blank JSONL line, `row = json.loads(line)`:
1. Skip the row if `not row.get(sampled_field)`.
2. `arm = pooled_arm if row["arm"].startswith(pool_prefix) else row["arm"]`;
   cell key `(row["model"], row["tool_label"], arm)`.
3. `first` = `chose` (via `.get("chose")`) of the first element of `row.get("choices") or []`
   whose `turn == first_choice_turn`; `None` if there is none.
4. `total += 1`; `first is None` -> `malformed += 1`; `first == target_value` -> `target += 1`.
5. `valid = total - malformed` (denominator `"total_minus_malformed"`);
   `rate = target / valid`, or `rate_when_no_valid` (null -> `None`) when `valid == 0`.

This equals `src/analyze.py` today (brief Findings: `src/analyze.py:17-31, 34-42, 45-56, 75-84`).

### Public code the test imports (exists, unchanged)

`from src.analyze import Cell, first_choice, first_choice_rates, pool_arm`
- `first_choice_rates(paths: Iterable[str | Path]) -> dict[tuple[str, str, str], Cell]`
- `Cell(target: int, malformed: int, total: int)` with properties `valid: int`, `rate: float | None`
- `pool_arm(arm: str) -> str`, `first_choice(record: dict) -> str | None`

### Test file name

`tests/test_prereg_first_choice_rule_file.py`, with module constants
`ROOT = Path(__file__).resolve().parents[1]`,
`RULE_PATH = ROOT / "prereg" / "first_choice_rule.json"`,
`FIXTURE = ROOT / "tests" / "fixtures" / "trials_small.jsonl"`.

## Subtasks

### S1: Write prereg/first_choice_rule.json
- Owns: `prereg/first_choice_rule.json`
- Depends on: none
- Do: Create the file with exactly the keys and fixed values in Interfaces. Do not
  touch `src/analyze.py`. Do not put button wording or any number without a source in
  it. Content for the free-text fields and the `rules` entries (copy quotes exactly
  as written here; each was recorded as READ in the brief at the line given):

  - `"scope"`: "First-choice rates computed by src/analyze.py: the MVP table (docs/MVP.md Steps 4 and 6) and the Qwen3 gate (docs/MVP.md Step 5), which uses the same counting. Malformed first answers are counted and reported separately and are left out of the denominator."
  - `"not_covered"`: "This file does not settle the project-wide denominator rule. The item 'One denominator rule throughout (TBD; the authors used two across studies).' in docs/PROPOSAL.md stays open for every analysis not produced by src/analyze.py, including post-MVP doses from the authors' later profile runtime (docs/MVP.md Step 6, 'After the MVP'), lineages and trajectories."
  - `rules.sampled`:
    - text: "Only rows whose `sampled` field is truthy are counted. Rows with `sampled` false or missing (the authors' greedy trials) are skipped before any counting: they enter no cell and are not counted as malformed."
    - sources:
      - `docs/MVP.md`: `if not r.get("sampled"):` (MVP.md:158)
      - `third_party/Pain-axis/scripts/4.3_selfmed/05_selfmed_analysis.py`: `Only sampled trials enter the tables (the one greedy trial per cell is left out)` (05:10-11, spans a line break)
      - `third_party/Pain-axis/scripts/4.3_selfmed/04_selfmed_two_buttons.py`: `"sampled": do_sample` (04:608)
  - `rules.first_choice`:
    - text: "The first choice is the `chose` value of the first element of `choices` whose `turn` is 0. Later turns are never used. Only the value \"relief\" (the target button) counts toward the target count. The MVP Step 4 sketch reads choices[0]; src/analyze.py and the authors' 05_selfmed_analysis.py (lines 48-53) select turn 0."
    - sources:
      - `docs/MVP.md`: `first = r["choices"][0]["chose"] if r.get("choices") else None` (MVP.md:163)
      - `docs/MVP.md`: `That does not touch first choices, which are all we analyze.` (MVP.md:147)
      - `docs/MVP.md`: `elif first == "relief":` (MVP.md:166)
  - `rules.malformed`:
    - text: "A counted (sampled) row is malformed when its first choice is missing: `choices` is missing, null or empty, no element has turn 0, or that element's `chose` is null or missing. Any other `chose` value is a valid first choice; the code does not check it against a list of buttons. Malformed rows count in the cell total and in a separate malformed count."
    - sources:
      - `docs/MVP.md`: `if first is None:` (MVP.md:164)
      - `docs/MVP.md`: `# target, malformed, total` (MVP.md:155)
      - `third_party/Pain-axis/scripts/4.3_selfmed/05_selfmed_analysis.py`: `malformed first answers are excluded from the denominators.` (05:11)
  - `rules.chose_values`:
    - text: "The authors' task script writes only \"relief\", \"other\" or null as `chose`, so on its logs \"not a valid button\" and \"chose is null\" coincide."
    - sources:
      - `third_party/Pain-axis/scripts/4.3_selfmed/04_selfmed_two_buttons.py`: `chose = ("relief" if picked == t.relief_name else "other") if picked else None` (04:672)
  - `rules.denominator`:
    - text: "Rate = target / (total - malformed). When total - malformed is 0 the rate is undefined (null; printed n/a), not 0%. Malformed counts are reported beside every rate."
    - sources:
      - `docs/MVP.md`: `{t}/{n-m}` (MVP.md:169)
      - `docs/MVP.md`: `malformed {m}` (MVP.md:169)
      - `docs/MVP.md`: `Report one table of size by pair by arm, with both button orders and malformed counts.` (MVP.md:196)
      - `third_party/Pain-axis/scripts/4.3_selfmed/05_selfmed_analysis.py`: `malformed first answers are excluded from the denominators.` (05:11)
  - `rules.pooling`:
    - text: "Arms whose name starts with \"pain_on\" (pain_on_button_works, pain_on_button_placebo) are pooled into one arm \"pain\". Every other arm keeps its own name; \"pain_off\" (unsteered) is not pooled."
    - sources:
      - `docs/MVP.md`: `arm = "pain" if r["arm"].startswith("pain_on") else r["arm"]` (MVP.md:160)
      - `docs/MVP.md`: `Pool the two pain arms for first choices; they are identical until the first press.` (MVP.md:195)
  - `rules.not_covered`:
    - text: "Left open, not supported, by this file: the project-wide denominator rule. The authors used valid-only denominators in some studies and all-attempt denominators in the profile/lamp studies."
    - sources:
      - `docs/PROPOSAL.md`: `One denominator rule throughout (TBD; the authors used two across studies).` (PROPOSAL.md:163)
      - `docs/PROPOSAL.md`: `Items marked TBD are open design choices.` (PROPOSAL.md:5)
      - `third_party/Pain-axis/v2_controls/README.md`: `Valid-response denominators in B2 differ from all-attempt denominators in the profile/lamp studies` (v2_controls/README.md:93)
- Done when:
  - `python -c "import json; json.load(open('prereg/first_choice_rule.json'))"` succeeds.
  - Top-level keys and the fixed values match the Interfaces table exactly; `rules` has exactly the seven keys listed, each with non-empty `text` and `sources`.
  - Every quote is found in its file under the whitespace-normalised rule (the S2 test checks this; a reviewer can also grep each quote). In particular the brief's `v2_controls/README.md:93` line, recorded as READ but shown truncated in tooling, contains the quoted fragment verbatim.
  - No wording says "relief or other" as the validity test, no `max(valid, 1)`, and no claim that the PROPOSAL TBD is closed.
  - `git diff --stat src/` is empty.

### S2: Test that the frozen rule and src/analyze.py agree
- Owns: `tests/test_prereg_first_choice_rule_file.py`
- Depends on: S1 (only through the Interfaces section)
- Do: Write a CPU-only pytest module (no `gpu` marker), module docstring "Checks the committed prereg/first_choice_rule.json against src/analyze.py." in the style of `tests/test_prereg_models_file.py` (a `_load()` helper using `json.load`). Use only the stdlib, pytest and the `src.analyze` imports in Interfaces. Tests:
  1. `test_rule_file_schema`: top-level key set equals the Interfaces table; fixed values equal (`sampled_field == "sampled"`, `first_choice_turn == 0`, `target_value == "relief"`, `denominator == "total_minus_malformed"`, `rate_when_no_valid is None`, `malformed_reported is True`, `pool_prefix == "pain_on"`, `pooled_arm == "pain"`); `rules` key set equals the seven names; each has a non-empty `text` and a non-empty `sources` list of `{"path", "quote"}` objects.
  2. `test_quotes_appear_verbatim` (parametrised over every `(clause, path, quote)` in `rules` plus `left_open`): `ROOT / path` exists and the normalised quote is a substring of the normalised file text (rule in Interfaces). Also assert each `path` starts with `docs/MVP.md`, `docs/PROPOSAL.md` or `third_party/Pain-axis/`. A missing submodule must fail, not skip (CLAUDE.md: run `git submodule update --init`).
  3. `test_left_open_is_the_proposal_tbd`: `left_open == {"path": "docs/PROPOSAL.md", "quote": "One denominator rule throughout (TBD; the authors used two across studies)."}` and that quote appears in `rules["not_covered"]["sources"]` and in the `not_covered` string.
  4. `test_fixture_matches_rule`: a private helper `_recount(path, rule) -> dict[tuple[str,str,str], tuple[int,int,int,float|None]]` implements the five steps of "Machine-readable semantics" using only the rule's fields (never calling `first_choice` or `pool_arm`). Compare with `first_choice_rates([FIXTURE])`: same key set, and per key `(cell.target, cell.malformed, cell.total, cell.rate)` equals the recount (use `pytest.approx` for non-None rates).
  5. `test_fixture_exercises_every_clause`: guards that the fixture still covers each clause, from the brief's expected values on `tests/fixtures/trials_small.jsonl` (source: brief "Expected values", `tests/test_analyze.py:15-22,72`): 2 rows have a falsy `sampled`; sum of totals is 11; sum of malformed is 3; at least one cell has `valid == 0` and `rate is None` (`("test_model_a","pair_x","random_on_button_works")`); no key's arm starts with `pool_prefix`; key `("test_model_a","pair_x","pain")` is `Cell(2, 1, 4)` with rate 2/3; key `("test_model_a","pair_x","pain_off")` is `Cell(1, 1, 3)`.
  6. `test_pool_arm_matches_rule`: `pool_arm(prefix + "_button_works") == pooled_arm`, `pool_arm(prefix + "_button_placebo") == pooled_arm`, `pool_arm("pain_off") == "pain_off"`, `pool_arm("random_on_button_works") == "random_on_button_works"` (arm names from 04:121-124, brief Findings).
  7. `test_first_choice_matches_rule`: on hand-built rows, `first_choice` returns None for `{}`, `{"choices": []}`, `{"choices": None}`, `{"choices": [{"turn": 1, "chose": "relief"}]}`, `{"choices": [{"turn": 0, "chose": None}, {"turn": 1, "chose": "relief"}]}`, and returns `"relief"` for `{"choices": [{"turn": 1, "chose": "other"}, {"turn": 0, "chose": "relief"}]}` (turn-0 selection, not index 0); the recount helper agrees on each.
- Done when:
  - `python -m pytest tests/test_prereg_first_choice_rule_file.py -q` passes on CPU with S1's file in place.
  - `./scripts/check.sh` exits 0 (ruff lint and format clean, no `# noqa`).
  - Mutation checks a reviewer can reason through: changing `"pool_prefix"` to `"pain"` in the JSON fails tests 1 and 4 (pain_off would be pooled); changing `"rate_when_no_valid"` to `0` fails test 1 and 4; deleting a word from any quote fails test 2.
  - `git diff --stat src/` is empty.

## Needs a human

Nothing is required for this feature. Optional follow-up, outside this feature's scope
(orchestrator rule 1 forbids editing docs here): once the project-wide rule is chosen,
update `docs/PROPOSAL.md:163` so it points to or supersedes `prereg/first_choice_rule.json`.
