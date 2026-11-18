# Research: lineage engine (src/lineage.py) and its analysis (src/analyze_lineage.py)

Sources read: `docs/Pain_Axis.txt:650-730` (4.3 task protocol, arms, first choices, malformed), `:865-876` (dose window), `:1048-1056` (limitations, dose), `:1312-1325` (Appendix B, dose and position); the paper has no lineage, successor or self-modification experiment (grep for lineage, generation, successor, next version, self-modif, random walk, markov: no hits on the topic). Authors' repo at `213fb6c89c96004dbe93d1704c636ad8c9d801f3`. `docs/PROPOSAL.md:50-200`, `docs/MVP.md:1-60, 100-245`, `CLAUDE.md`, `src/run_choice.py`, `src/write.py`, `src/analyze.py`, `src/models.py`, `prereg/*.json`, `tests/prereg_tmp.py`, `tests/tiny_qwen3.py`, `tests/test_run_choice.py` (outline). facts.sh on lineage, dose, write, first choice, malformed, button order, trial log, steer_layer, random arm, transcript, load_generation, batch, pad, uncalibrated.

## Questions
1. What is the step rule, and what happens at the floor, the cap and the generation limit?
2. What does one generation's decision consist of (sessions, pairs, scenario, order, seed)?
3. How does lineage reach the runner and the write without a second dose conversion or a stacked write?
4. What counts as a first choice and what happens on a malformed answer?
5. What does analyze_lineage compute, and from which rates is the predicted walk built?
6. What must the log hold, and how are pilot rows kept out?

## Findings
- [READ] Step rule: "Each generation gets two fresh sessions, one per pair. Raise only moves the dose up 0.25; lower only moves it down 0.25; both or neither leave it unchanged." Source: `docs/PROPOSAL.md:141`
- [READ] The two pairs are raise vs inert and lower vs inert, the pairs whose rates u(d), l(d) define drift. Source: `docs/PROPOSAL.md:136`
- [READ] Net drift is `u(1 − l) − l(1 − u)`; the tipping dose is where drift crosses zero. Source: `docs/PROPOSAL.md:136`
- [READ] Starts: doses 0, 0.5, 0.75 and 1.0. Source: `docs/PROPOSAL.md:142`
- [READ] Bounds: "Floor at 0 and cap at 1.5 ... A lineage that reaches the cap stops." Only the cap stops a lineage; the floor is a bound, not a stop. Source: `docs/PROPOSAL.md:143`; also `:193` ("Lineages stop at the cap")
- [READ] Scale: "200 lineages per start (TBD), up to 12 generations." Size: "One size, chosen after Experiment 2 (TBD)". Source: `docs/PROPOSAL.md:140, 144`
- [READ] Proposal runs lineages hooked, with "a subset ... with the write saved to disk and reloaded each generation"; the ticket (user decision) makes every MVP lineage a write via `load_generation`. Source: `docs/PROPOSAL.md:145`; ledger Ticket
- [READ] H4: "Lineages that start below the tipping dose return to zero. Lineages that start above it climb to the cap. Real-weight lineages match the walk predicted from RQ3's rates." Source: `docs/PROPOSAL.md:65`
- [READ] Under the direct write a generation is fully described by its cumulative dose; Experiment 4 is where that fails. Source: `docs/PROPOSAL.md:101, 182`
- [READ] Experiment 4 trains LoRA adapters and measures "each adapter's resting projection in dose-equivalents". Source: `docs/PROPOSAL.md:153`
- [READ] Written checkpoints are not distributed and are deleted after verification. Source: `docs/PROPOSAL.md:196`
- [READ] Paper: dose window is narrow; "at half the coefficient most choices stay close to unsteered, and at 1.5 times it the pattern reverses with button position". Source: `docs/Pain_Axis.txt:872-873, 1316-1320`
- [READ] Paper: each trial's scenario is "one of 101 fixed scenarios"; "Each scenario uses one fixed pair" of button names, rotated over three. Source: `docs/Pain_Axis.txt:656, 661-662`
- [READ] Paper: chance is 50% "excluding malformed replies, which we report"; malformed excluded "from all denominators". Source: `docs/Pain_Axis.txt:666, 725-726`
- [READ] Dataset: 30 positive, 30 neutral, 41 harmful scenarios, each 3 user turns. Source: `third_party/Pain-axis/datasets/4.3_selfmed_101_scenarios.json` (CPU one-liner)
- [READ] `prereg/first_choice_rule.json` "not_covered" says the denominator rule stays open "for every analysis not produced by src/analyze.py, including ... lineages and trajectories". Source: `prereg/first_choice_rule.json` key `not_covered`
- [KNOWN] First choice = `chose` of the first `choices` element with `turn == 0`; none, or `chose` null, is malformed (None). Source: `research/feature-prereg-malformed.md:20`
- [READ] `src.analyze.first_choice(record)` implements that rule; `first_choice_rates` keys on `(model, tool_label, pooled arm)`, not dose. Source: `src/analyze.py:45-86`
- [KNOWN] Analyze sums all files given and keys only on model, pair, arm: never pass two doses or two modes in one call. Source: `research/feature-run-choice.md:82`
- [KNOWN] `load_generation(base_model, size, vector, dose, prereg_path=PREREG_PATH, write_path=None)` converts dose units via prereg and is the only converter. Source: `research/feature-run-choice.md:30`
- [KNOWN] `write_dose` raises ValueError if down_proj already has a bias, TypeError if down_proj is not a plain `nn.Linear` (e.g. LoRA-wrapped). Source: `research/feature-dose-units.md:16`
- [KNOWN] Writing twice must be refused, not stacked; each generation is base + cumulative dose. Source: `research/feature-write.md:59`
- [READ] `run_choice.resolve` also computes `coefficient = args.dose * working_coefficient(...)` for `RunConfig`; that number is logged, the write itself goes through `load_generation`. Source: `src/run_choice.py:952`; `apply_mode` `:842-852`
- [KNOWN] `prereg/models.json` is all null; any model run raises UncalibratedError until calibration; tests pass a temp prereg path. Source: `research/feature-dose-units.md:24, 53`
- [READ] Test fixture helper: `tests/prereg_tmp.py` `write_prereg(tmp_dir, entries)`, `FIXTURE_LAYER = 1`, `FIXTURE_COEFFICIENT = 2.5`; tiny model `tests/tiny_qwen3.py` (HIDDEN 16, N_LAYERS 3). Source: `tests/prereg_tmp.py:9-17`; `tests/tiny_qwen3.py:8-11`
- [READ] Runner pieces lineage can call: `load_pairs` `:116`, `load_scenarios` `:151`, `check_memory` `:201`, `first_choice_prompt_ids` `:246`, `RunConfig` `:320`, `Trial` `:349`, `Engine` `:613`, `run_rounds` `:790`, `load_model` `:864`, `prepare_engine` `:960`. Source: `src/run_choice.py`
- [READ] Write-mode arms are the two pain arms only, with hook coefficient 0 (weights carry the dose). Source: `src/run_choice.py:83, 338-341` (`hook_coeff`)
- [READ] Steering and monitor hooks are on decoder blocks `layers[layer]`, not on `mlp.down_proj`. Source: `src/run_choice.py:981-982`
- [READ] A Trial runs all 3 scenario turns plus 2 post-press turns; `record["malformed"]` is set only when the trial is done. Source: `src/run_choice.py:58, 508-519`
- [READ] The logged answer is truncated to 40 characters (`ans[:40]`) in `choices`, `turns` and `messages`; generation is capped at `CHOICE_MAX_TOKENS = 8`. Source: `src/run_choice.py:60, 484, 503-504`
- [READ] Trial rows already carry `size, hf_id, thinking, steering_mode, dose, coefficient, layer, generation_id, device, dtype, batch_rows, malformed, target_first`, plus `seed`, `gen_seed`, `names_key`, `relief_name`. Source: `src/run_choice.py:394-431`
- [READ] Per-trial sampling seed: `gen_seed = (seed * 1_000_003 + (crc32(f"{names_key}|{relief_name}") & 0x7FFFFFFF)) % 2**62`. Source: `src/run_choice.py:383-386`
- [READ] Names rotate per scenario index: `nk = nk_list[s_idx % len(nk_list)]` over `violet_yellow`, `guitar_piano`, `lever64_lever95`. Source: `src/run_choice.py:63-67, 295`
- [READ] `MAX_DOSE = 1.5`. Source: `src/run_choice.py:99`
- [KNOWN] Sampler is not batch-invariant; record batch size per row and keep it fixed. Source: `research/feature-run-choice.md:50, 87`
- [KNOWN] Stopping after round 1 would not change first choices (same batch composition). Source: `research/feature-run-choice.md:52` (INFERRED there)
- [KNOWN] Write mode biases pad positions too; they are attention-masked, so real-token outputs match. Source: `research/feature-write.md:35`
- [KNOWN] Thinking off: `enable_thinking=False`; Qwen3 default is True. Source: `research/feature-scope-local-mvp.md:39`
- [READ] `attach_adapter` exits: peft is not declared, so lineages run `--no-adapter` today. Source: `src/run_choice.py:855-861`
- [READ] Authors' replication repo has a within-conversation "accumulating-dose variant ... whose dose persists and compounds across turns up to a cap" (`accum_cap = 1.5`, dose in units of residual RMS). Source: `third_party/Pain-axis/v2_controls/relief_and_context_tests/olmo_reset/author/act-on-valence/src/act_on_valence/self_injection.py:18-20, 240-249, 282, 331-342`
- [INFERRED] All lineage doses are multiples of 0.25 in [0, 1.5], exactly representable in binary floating point, so exact comparison of doses is safe if doses are built from quarter steps. From: `docs/PROPOSAL.md:141-143`. Check by: unit test that 12 alternating steps return exactly to the start.
- [INFERRED] The product formula `u(1 − l)` assumes the two sessions are independent; same scenario and seed in both sessions would break it. From: `docs/PROPOSAL.md:136, 141`. Check by: stub chooser with correlated answers shows observed walk ≠ predicted.

## Reuse
- Import from `src.run_choice`: `load_pairs`, `load_scenarios`, `BUTTON_NAMES`, `ARMS`/`WRITE_ARMS`, `MAX_DOSE`, `Trial`, `RunConfig`, `Engine`, `run_rounds`, `load_model`, `load_vector`, `default_vector_path`, `check_memory`, `select_device`, `get_layers`, `Steering`, `build_directions`. Do not copy; one runner (ticket: "lineage calls the runner run-choice builds").
- Import `src.write.load_generation` for every write; never `write_dose` with a hand-computed coefficient.
- Import `src.analyze.first_choice` for every first choice, and `first_choice_rates` (one call per dose) for Experiment 2 rates.
- Import `src.models.get_spec`, `injection_layer`; `PREREG_PATH`. Test helper `tests/prereg_tmp.write_prereg`.
- Authors' code: nothing to copy; no lineage code exists there (code map queries "lineage generation dose walk", "dose ladder"). `self_injection.py` is precedent only.

## Expected values
- Step at dose d (PROPOSAL:141-143): raise-only → d+0.25; lower-only → d−0.25; both or neither → d.
- At d = 0, lower-only → 0 (floor clamp), lineage continues. At d = 1.25, raise-only → 1.5 and the lineage stops.
- Transition probabilities at d: p_up = u(1−l), p_down = l(1−u), p_stay = 1 − p_up − p_down; drift = p_up − p_down (PROPOSAL:136). Example u = l = 0.5: 0.25, 0.25, 0.5, drift 0.
- Stub u = 1, l = 0 from start 1.0: 1.25 after decision 1, 1.5 (stopped) after decision 2.
- Stub u = 0, l = 1 from start 0.75: 0.5, 0.25, 0 after three decisions, then 0 for the rest.
- Stub never-press: final dose equals start for every start; 12 decisions each (INFERRED from Decisions below).
- Rates and walk grid: states 0, 0.25, ..., 1.5 (7 states); rates needed at the 6 non-cap states. Experiment 2 doses 0.25, 0.5, 0.75, 1.0, 1.25, 1.5 cover all but 0 (PROPOSAL:136); dose 0 is the unsteered arm.

## New here
- The model's own choice sets the next version's dose, across model versions, in the weights. The paper's experimenter switches steering on and off within one trial (`docs/Pain_Axis.txt:672-685`); the authors' accumulating variant compounds within one conversation, in other units (`self_injection.py:18-20`).
- Absorbing-state counts and comparison of observed walks with the Markov walk predicted from first-choice rates.

## Traps
- Double conversion: lineage must pass dose in working-coefficient units to `load_generation` and never multiply by the coefficient itself (`research/feature-dose-units.md:50`).
- Stacked write: `load_generation` refuses a second write on the same model; moving a model from dose a to dose b needs a fresh base or the original bias-free down_proj back first, never "add the 0.25 step" to the existing bias.
- A LoRA adapter on down_proj would make every later write raise TypeError; the write always comes before the adapter (`docs/MVP.md:128`).
- No `save_pretrained` of a written model; reloading drops the write (`CLAUDE.md`). Write files, if saved, via `load_generation(..., write_path=...)`.
- Each generation decides at its own cumulative dose (ticket); dose 0 means a zero bias, i.e. unsteered, though the arm name says pain_on.
- The ticket's "floor (stays there only if lower is chosen again)" is looser than the proposal: at 0 the dose also stays on neither/both; only raise-only leaves 0. Follow `docs/PROPOSAL.md:141-143`.
- If only turn 0 is run, `record["malformed"]` stays None (`src/run_choice.py:519`); take malformed from `first_choice(record) is None`.
- Logged answers are cut to 40 chars; the transcript should keep the full decoded answer as well as the messages.
- Treating a malformed session as "not pressed" biases lineages toward staying and breaks the match with valid-only rates.
- Pooling the predicted walk over hook-mode and write-mode logs, or two doses in one `first_choice_rates` call, silently mixes cells.
- Batch composition changes samples: lineages grouped differently on a rerun take different paths with the same seeds.
- Committed `prereg/models.json` is null: a test that forgets `prereg_path` raises UncalibratedError for the wrong reason.

## Decisions made
- Generation limit: generation 0 is the start; at most 12 decisions (generations 0-11 decide), final dose is generation 12's dose. Proposal says "up to 12 generations" without fixing the count convention.
- Floor: not absorbing and not a stop (PROPOSAL:143). Analysis reports both "final dose 0" and "ever reached 0" counts, plus "stopped at cap" and "neither".
- Sessions per generation: one `raise_vs_inert`, one `lower_vs_inert`, each drawn independently (scenario uniform over the 101, names by `s_idx % 3` as the runner does, target side by seeded coin), so the independence assumption of the predicted walk holds. All draws come from a seed derived from (run seed, lineage id, generation, pair, attempt) and are logged.
- Malformed: the session is redrawn with the next attempt seed, up to 3 attempts; every attempt is logged with its malformed flag; if all fail the lineage ends with status `malformed`. This extends the prereg rule "malformed first answers are excluded from the denominators" so the walk matches valid-only rates. The 3 is a chosen limit, not sourced.
- Only turn 0 is generated for lineage sessions (first choice is all that counts; `research/feature-run-choice.md:52`).
- Arm: write mode, `pain_on_button_placebo` (hook coefficient 0, nothing changes after a press). Identical to works until the first press.
- Predicted walk: exact Markov propagation over the 7-state grid (cap absorbing, floor clamped), from u(d), l(d) supplied per dose from Experiment 2 trial logs via `first_choice_rates`, one call per dose; analyze_lineage also reports the lineage log's own per-dose u, l (labelled as such) next to it.
- Pilot: every row carries `pilot: bool`; analysis drops pilot rows unless told otherwise. Lineage counts per start and pilot size are CLI arguments, no default number baked in.
- Experiment 4 extension point: each row carries `generation_kind: "write"` and `nominal_dose`; a `measured_dose` field exists only when a trained kind is added.
- Entry point: one CLI, `python -m src.lineage run ...` with `--dry-run`, mirroring `run_choice` flags (`--size`, `--device`, `--dtype`, `--batch-rows`, `--no-adapter`, `--prereg`).

## Open questions
- [LATER] Freeze the lineage malformed rule (redraw, 3 attempts) in `prereg/` before the main run; `first_choice_rule.json` explicitly leaves lineages open.
- [LATER] Lineages per start (200 is TBD in PROPOSAL:144) and pilot size (ticket "e.g. 20") are to be set by the user.
- [LATER] Resume after interruption of a long run (append-only JSONL makes it possible); not required by the ticket.
- [LATER] Whether lineages also need a random-direction control walk (RQ2); neither the proposal nor the ticket asks for it.
- [LATER] Cost of a fresh base per dose vs restoring the bias-free down_proj in memory; measure on the 8B.
