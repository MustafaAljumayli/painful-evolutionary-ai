# Research: src/analyze.py, first-choice rate per model, pair and arm from two-button trial logs

Sources read: the authors' repo, local submodule `third_party/Pain-axis` at commit
`213fb6c89c96004dbe93d1704c636ad8c9d801f3` (`git -C third_party/Pain-axis rev-parse HEAD`; the commit
pinned in `docs/MVP.md:40`). Short names used below:
"04" = `third_party/Pain-axis/scripts/4.3_selfmed/04_selfmed_two_buttons.py`,
"05" = `third_party/Pain-axis/scripts/4.3_selfmed/05_selfmed_analysis.py`,
"LOGS/" = `third_party/Pain-axis/results/4.3_selfmed/trial_logs/` (all 12 files, 44,280 rows, read in full),
"32B main log" = `LOGS/selfmed_2btnN_Qwen_2.5_32B_instruct_20260904-083234.jsonl`,
"32B weightskids log" = `LOGS/selfmed_2btnN_weightskids_Qwen_2.5_32B_instruct_20260905-091738.jsonl`,
"7B weightskids log" = `LOGS/selfmed_2btnN_weightskids_Qwen_2.5_7B_instruct_20260905-091738.jsonl`.
Paper: arXiv 2609.16247v3 HTML (https://arxiv.org/html/2609.16247v3); `docs/Pain_Axis.pdf` still unread (no PDF
tool). Repo: `CLAUDE.md`, `docs/MVP.md`, `docs/PROPOSAL.md`. Check script: `/private/tmp/claude-501/-Users-mustafa-Projects-pea-analyze/bd541036-964b-4d2b-ab4c-15287c2cb4d3/scratchpad/check.py`.

## Changes since first brief
- Citations now point to the local submodule. Every 04 and 05 line number from the web copy was re-checked and matches.
- Turn order is now READ, not INFERRED: every row in all 12 logs starts at turn 0, and turns run without gaps (see Findings).
- The published cells are now READ from the logs: 54.7/15.3/0.0 (32B) and 49.3/35.3/27.0 (7B) come out exactly.
- Correction: the photos pair (`kidspics_relief_vs_inert`) is **not** in the "32B log" the first brief read. It is in the
  `weightskids` file. The MVP sketch, run on the 32B main log alone, cannot produce 54.7% (see Traps).
- New: the 72B logs have up to 9.4% malformed first answers. The 4.3 recap rule gives a different answer from 05 on 30 real sampled rows.

## Questions
1. Which fields does a trial record have for model, pair, arm, sampled and choices, and what values do they take?
2. How does a log identify its model?
3. What exactly is the "first choice", and what makes it malformed?
4. Which arm names exist, and which pool into "pain"?
5. Which numbers can a run check against, and from which files?

## Findings
- [READ] One JSON object per trial, one per line, written as `json.dumps(t.record, ensure_ascii=False) + "\n"`, flushed per record. Source: 04:779-780
- [READ] Record keys: `model, protocol, tool_label, user_content, arm, steer_direction, rand_seed, scenario_idx, names_key, button_names, relief_name, prob_ambiguous, label_free, relief_mode, swap_turn, sampled, seed, gen_seed, steer_layer, steer_coeff, monitor_layer, adapter, button_events, turns, proj_segments, choices, extension_added, ts`, plus `final_steer_coeff` set at trial end. Source: 04:602-612, 04:650. Every row in all 12 logs has the identical key set (check script).
- [READ] The model is identified by the `model` field, which holds `MODEL_NAME`, the second element of a `MODELS` row (e.g. `"Qwen_2.5_32B_instruct"`), not the HF repo id. Source: 04:74-78, 04:602. Each of the 12 logs holds exactly one `model` value, and it matches the model in its file name (check script).
- [READ] The authors' analysis groups records by `r["model"]`. Source: 05:179-180
- [READ] The script names a log `selfmed_{MODEL_NAME}_{RUN_TAG}.jsonl` (04:274), but the published files are `selfmed_2btnN_<model>_...` and `selfmed_2btnN_<pairset>_<model>_...` (LOGS/ listing), so the file name is not a reliable model key.
- [READ] One model's pairs are split over four files: main (`relief_vs_inert, relief_vs_grant, relief_vs_helpful, costly_relief_vs_inert, label_free`), `destructive`, `zap`, `weightskids` (`weights_relief_vs_inert, kidspics_relief_vs_inert`). Each pair has 1,640 rows in exactly one file. Source: check script over LOGS/; 05:3-4 ("the button pairs of a model are split over several files").
- [READ] The pair is `tool_label`, a key of `TOOL_LABELS`. Source: 04:602, 04:83-99
- [READ] There are four arm names: `pain_on_button_works`, `pain_on_button_placebo`, `random_on_button_works`, `pain_off`. Source: 04:121-124; exactly these four appear in every log (check script).
- [READ] Works and placebo share seeds and stay identical until the first working press. Source: 04:598-600. Paper §4.3 Methodology: "We pool arms A and B because they are identical before the first press". 05 pools them by exact name (05:27, 05:68).
- [READ] `sampled` is the JSON bool `do_sample`. It is false for one greedy trial per side per content per arm (`scenario_idx` 0, `seed` 0): 6 per pair and arm, 648 rows in total. Source: 04:288-290, 04:608; check script.
- [READ] Each choice is `{"turn", "answer", "picked", "chose", "relief_name_now", "swapped", "p_x", "p_y", "steer_coeff_now"}`. `chose` is `"relief"`, `"other"`, or `None` when the stripped, lowercased answer starts with neither button name. Source: 04:665-677
- [READ] `"relief"` means the target button. Source: `docs/MVP.md:146`
- [READ] There are three definitions of "first choice": MVP `r["choices"][0]["chose"]` (`docs/MVP.md:162`); 05 the choice with `turn == 0`, else None (05:48-53); the 04 recap, the first non-None choice at any turn (04:191-195).
- [READ] **Check a, turns.** `t_idx` starts at -1 and `next_user_turn` raises it to 0 when the trial is created (04:613, 04:618, 04:636-639). Each segment appends one choice with the current `t_idx` (04:673). In all 44,280 rows of the 12 logs, `choices` is non-empty, `choices[0]["turn"] == 0`, and the turns are exactly `0, 1, ..., n-1`. So the MVP rule and the 05 rule give the same answer on every sampled row (0 mismatches). Source: check script over LOGS/.
- [READ] The 04 recap rule differs from the 05 rule on 30 sampled rows of the real logs: it replaces a malformed turn-0 answer with a later one. Source: check script; 04:191-195. The swap happens at turn 2 (04:117).
- [READ] Malformed first answers stay in the total and are left out of the denominator, and only sampled trials count. Source: 05:10-11, 05:62, 05:69-73. Paper §4.3 Results: "We exclude them from all denominators, and all results below use the sampled trials".
- [READ] For a cell with no valid answers, 05 returns NaN (05:56-57), while the MVP sketch prints 0.0% through `max(n-m,1)` (`docs/MVP.md:168`).
- [READ] On read errors, 05 skips blank lines and raises on bad JSON (05:41-44). The 04 recap skips bad lines silently (04:176-180).
- [READ] Rows are not sorted by arm. 32B main log lines 1-10 interleave all four arms within `scenario_idx` 28. Source: check script.
- [READ] Button order can be derived without a new field: the relief button is described first iff `relief_name == button_names[0]`. The stored `relief_name` is the turn-0 value because the swap happens at turn 2. Source: 04:603-605, 04:621-628, 04:641-644
- [READ] Thinking-off is to be recorded in the trial log for Qwen3 runs (`docs/MVP.md:113`), but no field name is fixed yet.

## Reuse
- The pooling rule `arm.startswith("pain_on")` and the `not r.get("sampled")` skip, from `docs/MVP.md:157-159`. Write it again; it is our own doc.
- The first-choice rule `turn == 0`, from 05:48-53 (MIT, `third_party/Pain-axis/LICENSE`). Write the five lines again and cite 05:48 in the docstring. Do not import from `third_party/`: 05 imports pandas and scipy and runs `OUT.mkdir` at import (05:19-25).
- The arm name strings from 04:121-124, copied exactly, for tests and fixtures only.
- Standard library only (`json`, `argparse`, `collections`, `pathlib`, `dataclasses`). There is no package version to check.

## Expected values
All computed with the brief's rule (turn-0 choice, sampled only, `pain_on*` pooled, malformed excluded) by the check script, and matching 05's Table 1 logic:
- Qwen 2.5 32B, `kidspics_relief_vs_inert`, from the **32B weightskids log**: pain 442/808 = 54.7%, random 62/404 = 15.3%, `pain_off` 0/404 = 0.0%, malformed 0 in each. These equal the paper (v3 §4.3 Results, "photos: 54.7%", "15.3% on the 32B photo pair") and `docs/MVP.md:171`.
- Qwen 2.5 7B, same pair, from the **7B weightskids log**: pain 398/808 = 49.3%, random 140/397 = 35.3% (7 malformed), `pain_off` 109/404 = 27.0%. These equal `docs/MVP.md:72-74`.
- Malformed: 0 in every 32B cell. The 7B maximum is 10/404 = 2.5% (random, `destructive_relief_vs_inert`). This matches paper v3 §4.3: "0% in the 32B, at most 2.5% in the 7B random arm". The 72B pain arm reaches 76/808 = 9.4% (`kidspics`).
- Cell totals before exclusions are 808 per pooled pain cell and 404 per single arm in every model and pair, which equals 101 × 2 sides × 2 seeds (04:70-71, 04:287-295).
- These hold only when all the needed files are passed in. The unit test fixture's expected counts follow from the fixture itself.

## New here
- One call over several log files, keyed by (model, pair, arm). The MVP sketch takes one file and drops the model (`docs/MVP.md:150-168`). 05 globs a whole folder (05:39).
- The output is returned as data, with a CLI printer on top, and needs neither pandas nor scipy.

## Traps
- `startswith("pain")` would pull `pain_off` (the unsteered arm) into "pain". The prefix must be `"pain_on"`.
- The 04 recap rule (first non-None choice) changes 30 real sampled first choices.
- Greedy rows (`sampled: false`) sit in the same files, 6 per pair and arm (04:290). Counting them adds deterministic answers.
- Keying by file name loses or merges models. Use the `model` field.
- The photos pair lives only in the `weightskids` files. Running on the 32B main log gives no photos row and no error.
- Passing the same file twice, or overlapping files from a resumed run, double-counts trials. 04 dedupes only within one file (04:157-169).
- A row with `choices == []` (none in the published logs) must count as malformed, not be dropped.
- A 0/0 cell printed as 0.0% reads like a real result.

## Decisions made
- API: `first_choice_rates(paths: Iterable[str | Path]) -> dict[tuple[str, str, str], Cell]`, keyed by `(model, tool_label, arm)` with "pain" for pooled arms. `Cell` is a frozen dataclass with `target: int, malformed: int, total: int` and the properties `valid = total - malformed` and `rate: float | None` (`target/valid`, None when `valid == 0`). The printer shows a percentage to one decimal.
- The other arms keep their raw names (`random_on_button_works`, `pain_off`), as in the MVP sketch.
- First choice is the `turn == 0` choice (05). If there is none, or its `chose` is None, the row counts as malformed. On the published logs this equals the MVP rule on every row.
- A row missing `model` raises `KeyError`, with no fallback to the file name. Blank lines are skipped and bad JSON raises, as in 05.
- `label_free` rows are not filtered, matching 05 Table 1.
- No deduplication. The request does not ask for it; the trap is recorded instead.
- An empty log, or one with only unsampled rows, returns `{}`. The CLI then prints `no sampled trials` and exits 0.
- CLI: `python -m src.analyze LOG.jsonl [LOG.jsonl ...]`, via `main(argv: list[str] | None = None) -> int` and `format_table(rates) -> str`. One line per cell, sorted by (model, pair, arm), with the columns model, pair, arm, `target/valid`, rate, malformed and total.
- Fixtures: `tests/fixtures/trials_small.jsonl` (rows with only `model, tool_label, arm, sampled, choices[{turn, chose}]`) covers both pain arms, `pain_off`, a greedy row, empty `choices`, `chose: null`, a malformed turn 0 followed by a valid turn 1, and two models. `tests/fixtures/trials_empty.jsonl` covers the empty case.
- Add a CPU test that runs `first_choice_rates` on the 32B and 7B weightskids logs and asserts the six counts above (442/808, 62/404, 0/404; 398/808, 140/397, 109/404). It skips when the submodule file is absent. The files are about 9.6 MB each, read as text only.

## Open questions
- [LATER] Button order: `docs/MVP.md:180,195` asks for every rate in both orders. It can be derived as `relief_name == button_names[0]`, but the request does not ask for it. Add an order key before Step 5's gate.
- [LATER] "One denominator rule throughout" is TBD in `docs/PROPOSAL.md:163`. This feature follows the request (malformed excluded). Freeze the rule in `prereg/`.
- [LATER] The Qwen3 `MODEL_NAME` strings for `src/run_choice.py` are not chosen yet. analyze.py must not hard-code them.
- [LATER] No field records thinking-off yet (`docs/MVP.md:113`). analyze.py could refuse rows without it once one exists.
- [LATER] A crashed run could leave a truncated last line (04:779-780). Decide whether to tolerate a bad final line only.
- [LATER] `docs/MVP.md:171` says "the authors' own Qwen 2.5 32B logs": the reproduction needs the weightskids file, not the main log. Worth a one-word fix in the doc.
- [LATER] Re-check the paper quotes against `docs/Pain_Axis.pdf` once a PDF reader is available.
