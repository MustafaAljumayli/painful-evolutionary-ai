# Research: freeze the first-choice malformed / denominator / sampled / pooling rule in prereg/ and test it against src/analyze.py

Sources read: authors' repo, local submodule `third_party/Pain-axis` at `213fb6c89c96004dbe93d1704c636ad8c9d801f3`
(`git -C third_party/Pain-axis rev-parse HEAD`). Repo: `CLAUDE.md`, `docs/MVP.md`, `docs/PROPOSAL.md`,
`src/analyze.py`, `src/models.py`, `prereg/models.json`, `tests/test_analyze.py`,
`tests/test_prereg_models_file.py`, `tests/fixtures/trials_small.jsonl`, `research/feature-analyze.md`,
`plans/feature-analyze.md`. Paper not reopened: paper quotes below are taken from `research/feature-analyze.md`
(which read arXiv 2609.16247v3 HTML) and are marked as such. Short names: "04" =
`third_party/Pain-axis/scripts/4.3_selfmed/04_selfmed_two_buttons.py`, "05" = `.../05_selfmed_analysis.py`.

## Questions
1. What does `src/analyze.py` do today for valid button, malformed, unsampled and arm pooling, and what else?
2. Which sentences in `docs/MVP.md` Step 4 and `docs/PROPOSAL.md` back or leave open each clause?
3. What is in `tests/fixtures/trials_small.jsonl` and what does analyze.py output on it?
4. What conventions do `prereg/models.json` and its test follow?
5. Does freezing this rule decide something the docs leave open (NEEDS DECISION)?

## Findings
Code today (`src/analyze.py`):
- [READ] First choice = `chose` of the first element of `choices` whose `turn == 0`; none found, missing/empty `choices`, or `chose` null -> None. Later turns are never used. Source: `src/analyze.py:45-56` ("A missing turn-0 choice or a null `chose` is malformed and returns None; later turns are never used.")
- [READ] "Valid button" is not checked against a list: any non-None `chose` is valid; only `== "relief"` counts as target; anything else (incl. `"other"`) is valid non-target. Source: `src/analyze.py:80-84`
- [READ] Malformed rows count in `total` and in `malformed`; `valid = total - malformed`; `rate = target / valid`, `None` when `valid == 0`. Source: `src/analyze.py:17-31`, `:79-82`
- [READ] Unsampled = falsy `row.get("sampled")`: `false`, missing key, null, 0, "" are all skipped before any counting. Source: `src/analyze.py:75-76`; docstring "Only rows with a truthy `sampled` count." `:64`
- [READ] Pooling test is exactly `arm.startswith("pain_on")` -> `"pain"`; every other arm keeps its raw name; docstring: "\"pain_off\" must stay separate." Source: `src/analyze.py:34-42`
- [READ] Other row handling: blank lines skipped (`:72-73`); bad JSON raises `json.JSONDecodeError` (`:74`, tested `tests/test_analyze.py:108-110`); a sampled row missing `model`/`tool_label`/`arm` raises `KeyError` (`:77`, tested `:111-116`); cells keyed `(model, tool_label, pooled arm)` (`:77`); several files are summed with no dedup (`:64-65`, tested `:91-98`); no sampled rows -> `{}` and CLI prints `no sampled trials` (`:118-120`).
- [READ] Row fields come from 04: `"sampled": do_sample` (04:608), `"arm": arm_name` (04:602-603), arm names `pain_on_button_works`, `pain_on_button_placebo`, `random_on_button_works`, `pain_off` (04:121-124); each choice is `{"turn": t.t_idx, ..., "chose": chose}` with `chose = ("relief" if picked == t.relief_name else "other") if picked else None`, `picked` set only when the cleaned answer starts with a button name (04:666-674). So 04 only ever writes `"relief"`, `"other"` or `null`.
- [READ] Greedy (`sampled: false`) rows: one per side per content per arm, `scenario_idx` 0, `seed` 0 (04:288-290); 6 per pair and arm in the published logs (`research/feature-analyze.md:40`).
- [READ] Authors' rule matches: "Only sampled trials enter the tables (the one greedy trial per cell is left out); malformed first answers are excluded from the denominators." 05:10-11; implemented 05:62, 05:69-73; first choice by `turn == 0` 05:48-53.
- [READ] Paper v3 §4.3 (as quoted in `research/feature-analyze.md:39,46`): "We pool arms A and B because they are identical before the first press"; "We exclude them from all denominators, and all results below use the sampled trials".

Docs:
- [READ] MVP Step 4 sketch: `if not r.get("sampled"): continue` (`docs/MVP.md:158-159`); `arm = "pain" if r["arm"].startswith("pain_on") else r["arm"]` (`:160`); `first = r["choices"][0]["chose"] if r.get("choices") else None` / `if first is None: c[1] += 1` / `elif first == "relief": c[0] += 1` (`:163-167`); prints `{t}/{n-m}` and `malformed {m}` (`:169`). Counter comment `# target, malformed, total` (`:155`).
- [READ] MVP Step 6: "Pool the two pain arms for first choices; they are identical until the first press." (`docs/MVP.md:195`); "Report one table of size by pair by arm, with both button orders and malformed counts." (`:196`). Malformed under 5% is a pass condition (`:118`, `:174`, `:181`).
- [READ] MVP Step 4 "The `relief` key is just \"the target button\"" (`docs/MVP.md:147`).
- [READ] PROPOSAL leaves the denominator open: "**Reporting.** Every rate by button order, with malformed counts. One denominator rule throughout (TBD; the authors used two across studies)." (`docs/PROPOSAL.md:163`). And: "**Preregistration.** Wording, primary contrasts and exclusion rules are frozen before Experiment 1's main run." (`:160`). "Items marked TBD are open design choices." (`:5`).
- [READ] The authors' two rules: "Valid-response denominators in B2 differ from all-attempt denominators in the profile/lamp studies; malformed counts and separate greedy diagnostics are retained." (`third_party/Pain-axis/v2_controls/README.md:93`; also `v2_controls/choice_controls/README.md:14`, `v2_controls/MANUSCRIPT_DISCREPANCIES.md:9`).
- [READ] MVP points the post-MVP doses at that profile runtime: "sadness, fear and other doses come from the authors' later runtime in `v2_controls/choice_controls/profile/`" (`docs/MVP.md:200`), i.e. the all-attempt study.
- [READ] Prior feature left it open: "[LATER] \"One denominator rule throughout\" is TBD in `docs/PROPOSAL.md:163`. This feature follows the request (malformed excluded). Freeze the rule in `prereg/`." (`research/feature-analyze.md:95`). Plan cites 05:10-11, 05:69-73 and paper §4.3 for exclusion (`plans/feature-analyze.md:40`).
- [INFERRED] The MVP doc makes the valid-only choice for the MVP first-choice table (Step 4 code, Step 6), but PROPOSAL marks the project-wide rule ("throughout") TBD. Freezing a prereg file that reads as project-wide would close a TBD; one scoped to the MVP first-choice table records a choice MVP.md already makes. From: `docs/MVP.md:151-169,195-196`; `docs/PROPOSAL.md:5,160,163`. Check by: the lead's answer to the BLOCKING question.
- [READ] MVP first choice is `choices[0]`, code is `turn == 0`; on all 44,280 published rows they agree (0 mismatches). Source: `research/feature-analyze.md:43-44`.

Prereg conventions:
- [READ] `prereg/models.json` is plain JSON, 2-space indent, one compact object per key, no comments or source fields (`prereg/models.json:1-5`). Its path is a module constant `PREREG_PATH` in `src/models.py:16`.
- [READ] `tests/test_prereg_models_file.py` loads the file with `json.load`, checks key set against code (`:18-23`), and checks code behaviour against the file (`:26-35`); module docstring "Checks the committed prereg/models.json against the models.py interface." (`:1`).

## Reuse
- Nothing from `third_party/` is imported or copied; 04/05 are cited only. Import `first_choice_rates`, `first_choice`, `pool_arm`, `Cell` from `src.analyze` in the test.
- Reuse the fixture `tests/fixtures/trials_small.jsonl` unchanged and the `EXPECTED` table style of `tests/test_analyze.py:14-22`.

## Expected values
On `tests/fixtures/trials_small.jsonl` (13 rows, from analyze.py output and `tests/test_analyze.py:15-22`):
- Unsampled rows: lines 5 and 10 (`sampled: false`) -> 11 sampled rows counted (`tests/test_analyze.py:72`).
- Malformed sampled rows: line 4 (`turn 0 chose null`, turn 1 relief), line 7 (`choices: []`), line 9 (`chose null`) -> 3.
- pain_on* rows: lines 1-5, 11, 12 (`pain_on_button_works` and `pain_on_button_placebo`); none survives as an arm key.
- Cells `(target, malformed, total, valid, rate)`: a/pair_x/pain (2,1,4,3,2/3); a/pair_x/pain_off (1,1,3,2,0.5); a/pair_x/random_on_button_works (0,1,1,0,None); a/pair_y/pain (0,0,1,1,0.0); b/pair_x/pain (1,0,1,1,1.0); b/pair_x/pain_off (0,0,1,1,0.0). CLI prints `2/3  66.7%` and `0/0  n/a`.
- Real-data check already in place: Qwen 2.5 7B weightskids random cell 140/397 with 7 malformed of 404 (`tests/test_analyze.py:162`) is the only published cell where the denominator rule changes the rate (35.3% valid-only vs 140/404 = 34.7% all-attempt, INFERRED arithmetic).

## New here
- A frozen, machine-readable statement of the first-choice counting rule, and a test that recomputes cells from that statement and compares them with `first_choice_rates`. Neither MVP.md nor the authors' 4.3 code has a frozen rule file.

## Traps
- `startswith("pain")` would pool `pain_off` (unsteered) into pain; the prefix must be `pain_on` (`src/analyze.py:37-38`).
- `sampled` is tested for truthiness: a string `"false"` would count as sampled. 04 writes a JSON bool (04:608), so this only bites hand-made logs.
- "Valid button" in the request sounds like a membership check, but the code accepts any non-null `chose`. A prereg text saying "relief or other" would be stricter than the code; the test would not catch it on the fixture (fixture only has relief/other/null).
- A 0-valid cell has rate None, not 0.0%. A prereg text giving `target/max(valid,1)` (MVP sketch `docs/MVP.md:169`) would contradict the code.
- MVP's `choices[0]` vs code's `turn == 0`: wording the rule as "first element" differs from the code on any row whose first element is not turn 0 (none in published logs).
- A test that only re-asserts the `EXPECTED` table does not tie the prereg text to the code; it must derive expectations from the prereg fields.
- If the prereg reads as the project-wide rule, the post-MVP profile runs (`docs/MVP.md:200`) would be reported under a rule different from the authors' all-attempt rule for that study, with no error.

## Decisions made
- File: `prereg/first_choice_rule.json` (JSON, like `models.json`, so the test can read fields rather than parse prose). Proposed keys: `"scope"`, `"first_choice": {"field": "choices", "turn": 0, "target_value": "relief"}`, `"malformed": "turn-0 choice missing, choices missing or empty, or chose null"`, `"denominator": "valid"` (total minus malformed; rate null when valid is 0), `"malformed_reported": true`, `"sampled_field": "sampled"` (truthy only), `"pool_prefix": "pain_on"`, `"pooled_arm": "pain"`, `"sources"` (object of clause -> list of exact quotes with path:line from Findings).
- Test file: `tests/test_prereg_first_choice_rule_file.py`, mirroring `tests/test_prereg_models_file.py`. It reads the JSON, re-counts the fixture with only the stdlib and the prereg fields, and asserts equality with `first_choice_rates` per cell (target, malformed, total, rate incl. None); asserts `pool_arm(prefix + "_x") == pooled_arm` and `pool_arm("pain_off") == "pain_off"`; asserts every `sources` path exists and each quote string occurs in that file (catches doc drift too).
- No change to `src/analyze.py`; optionally add a `RULE_PATH` constant only in the test, not in src.
- Default `"scope"` if the lead does not answer: "MVP first-choice table (docs/MVP.md Steps 4 and 6); the project-wide denominator rule of docs/PROPOSAL.md:163 remains TBD." This records only what MVP.md already decides.

## Open questions
- [BLOCKING] Does this prereg file close the TBD in `docs/PROPOSAL.md:163` ("One denominator rule throughout (TBD; the authors used two across studies)") for every experiment, making valid-only (malformed excluded) the project-wide rule, or does it apply only to the MVP first-choice table that `docs/MVP.md` Steps 4 and 6 already specify? The authors used valid-only in §4.3/B2 and all-attempt in the profile/lamp studies (`third_party/Pain-axis/v2_controls/README.md:93`), and MVP.md:200 sends post-MVP doses to that all-attempt runtime.
- [LATER] Should "valid" be restricted to `chose in {"relief","other"}` to match the request's "valid button" literally? Same result on all 04 output (04:672); changing it would alter analyze.py, which the request forbids.
- [LATER] `docs/PROPOSAL.md:163` should be updated once the scope is decided, so the doc and prereg agree.
- [LATER] Paper quotes were not re-read from `docs/Pain_Axis.pdf` here; re-check them before quoting them in the prereg `sources`.
- [LATER] Button order (`docs/MVP.md:181,196`) is part of the reporting rule but not of this file or analyze.py yet.
