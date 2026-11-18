"""Tests for src/analyze_lineage.py, CPU only, on synthetic logs from a stub chooser.

Number sources: step 0.25, cap 1.5 and 12 generations come from docs/PROPOSAL.md:141-144;
the 0.25/0.25/0.5 transition example is research/feature-lineage.md "Expected values".
EVERY OTHER numeric literal (seeds, lineage counts, thresholds, rates) is an arbitrary
test value, not a setting.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from src import analyze_lineage as al
from src import lineage as ln


def synth(choose, starts, per_start, seed=3, pilot=False, max_gen=ln.MAX_GENERATIONS) -> list[dict]:
    """Hand-built lineage log in the format of src/lineage.py (no engine involved).

    `choose(rng, lineage_id, generation, dose)` returns (raise chosen, lower chosen).
    """
    rows: list[dict] = []
    base = {"pilot": pilot, "run_seed": seed, "generation_kind": "write", "size": "8B"}
    for start in starts:
        for k in range(per_start):
            lid = f"s{start:g}-{k:04d}"
            rng = random.Random(f"{seed}|{lid}")
            dose, gen, status = start, 0, None
            while status is None and gen < max_gen:
                up, down = choose(rng, lid, gen, dose)
                for pair, chose in ((ln.RAISE_PAIR, up), (ln.LOWER_PAIR, down)):
                    rows.append(
                        {
                            **base,
                            "kind": "session",
                            "lineage_id": lid,
                            "start": start,
                            "generation": gen,
                            "nominal_dose": dose,
                            "pair": pair,
                            "choice": "relief" if chose else "other",
                            "malformed": False,
                            "attempt": 0,
                        }
                    )
                new = ln.step_dose(dose, up, down)
                rows.append(
                    {
                        **base,
                        "kind": "step",
                        "lineage_id": lid,
                        "start": start,
                        "generation": gen,
                        "nominal_dose": dose,
                        "raise_chosen": up,
                        "lower_chosen": down,
                        "next_dose": new,
                    }
                )
                dose, gen = new, gen + 1
                status = "cap" if dose >= ln.MAX_DOSE else None
            status = status or "generation_limit"
            rows.append(
                {**base, "kind": "end", "lineage_id": lid, "start": start, "status": status}
                | {"final_dose": dose, "generations": gen}
            )
    return rows


def independent(u: float, lo: float):
    def fn(rng, lid, gen, dose):
        return rng.random() < u, rng.random() < lo

    return fn


def test_half_half_transitions_and_zero_drift():
    up, down, stay = al.transition_probs(0.5, 0.5)
    assert (up, down, stay) == (0.25, 0.25, 0.5)
    assert al.drift(0.5, 0.5) == 0


def test_cap_absorbing_and_floor_clamped():
    rates = {d: (0.5, 0.5) for d in al.grid()}
    at_cap = al.propagate({ln.MAX_DOSE: 1.0}, rates)
    assert at_cap == {ln.MAX_DOSE: 1.0}  # no rates needed, no leak
    at_floor = al.propagate({0.0: 1.0}, rates)
    assert at_floor[0.0] == pytest.approx(0.75)  # down mass stays at 0
    assert at_floor[0.25] == pytest.approx(0.25)
    assert min(at_floor) == 0.0
    walk = al.predicted_walk(1.25, {d: (1.0, 0.0) for d in al.grid()}, 3)
    assert walk[1][1.5] == 1.0 and walk[3][1.5] == 1.0


def test_missing_rates_named_in_error():
    with pytest.raises(al.AnalysisError, match="0.5"):
        al.propagate({0.5: 1.0}, {})


def test_pilot_excluded_by_default_included_on_request(tmp_path):
    path = tmp_path / "l.jsonl"
    rows = synth(independent(0.5, 0.5), [0.5], 2, pilot=True, max_gen=2)
    rows += synth(independent(0.5, 0.5), [0.5], 3, seed=4, max_gen=2)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    kept = al.load_rows([path])
    assert kept and not any(r["pilot"] for r in kept)
    assert sum(r["kind"] == "end" for r in kept) == 3
    assert sum(r["kind"] == "end" for r in al.load_rows([path], include_pilot=True)) == 5


def test_logs_of_two_sizes_are_not_pooled(tmp_path):
    a = synth(independent(0.5, 0.5), [0.5], 1, max_gen=1)
    b = [dict(r, size="14B") for r in a]
    path = tmp_path / "l.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in a + b))
    with pytest.raises(al.AnalysisError, match="size"):
        al.load_rows([path])


def trial_row(dose, mode, pair, chose):
    return {
        "sampled": True,
        "model": "m",
        "tool_label": pair,
        "arm": "pain_on_button_works",
        "choices": [{"turn": 0, "chose": chose}],
        "dose": dose,
        "steering_mode": mode,
    }


def write_trials(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_rates_for_one_dose_and_no_pooling(tmp_path):
    rows = [trial_row(1.0, "write", ln.RAISE_PAIR, c) for c in ("relief", "other", "relief", None)]
    rows += [trial_row(1.0, "write", ln.LOWER_PAIR, c) for c in ("other", "relief")]
    ok = write_trials(tmp_path / "a.jsonl", rows)
    u, lo = al.rates_for_dose([ok], "m", 1.0)
    assert u == pytest.approx(2 / 3)  # malformed excluded from the denominator
    assert lo == pytest.approx(0.5)
    two_doses = write_trials(
        tmp_path / "b.jsonl", rows + [trial_row(0.5, "write", ln.RAISE_PAIR, "other")]
    )
    with pytest.raises(al.AnalysisError, match="declares"):
        al.rates_for_dose([two_doses], "m", 1.0)
    two_modes = write_trials(
        tmp_path / "c.jsonl", rows + [trial_row(1.0, "hook", ln.RAISE_PAIR, "other")]
    )
    with pytest.raises(al.AnalysisError, match="mix"):
        al.rates_for_dose([two_modes], "m", 1.0)


def test_declared_dose_must_match_log_dose(tmp_path):
    rows = [trial_row(0.75, "write", p, "relief") for p in (ln.RAISE_PAIR, ln.LOWER_PAIR)]
    log = write_trials(tmp_path / "a.jsonl", rows)
    with pytest.raises(al.AnalysisError, match="declares dose 0.5"):
        al.rates_for_dose([log], "m", 0.5)
    lineage_log = tmp_path / "l.jsonl"
    lineage_log.write_text(
        "".join(json.dumps(r) + "\n" for r in synth(independent(0.5, 0.5), [0.5], 2, max_gen=1))
    )
    with pytest.raises(SystemExit, match="declares"):
        al.main([str(lineage_log), "--model", "m", "--rates", f"0.5={log}"])


def test_rates_from_two_steering_modes_are_refused(tmp_path, capsys):
    lineage_log = tmp_path / "l.jsonl"
    lineage_log.write_text(
        "".join(json.dumps(r) + "\n" for r in synth(independent(0.5, 0.5), [0.5], 2, max_gen=1))
    )
    argv = [str(lineage_log), "--model", "m"]
    for dose, mode in ((0.5, "write"), (0.75, "hook")):
        rows = [
            trial_row(dose, mode, p, c)
            for p in (ln.RAISE_PAIR, ln.LOWER_PAIR)
            for c in ("relief", "other")
        ]
        argv += ["--rates", f"{dose}={write_trials(tmp_path / f'{mode}.jsonl', rows)}"]
    with pytest.raises(SystemExit, match="mix steering modes"):
        al.main(argv)


def lineages_of(rows, start=None):
    paths = al.lineage_paths(rows)
    return [v for v in paths.values() if start is None or v["start"] == start]


def test_independent_log_matches_predicted_walk():
    rows = synth(independent(0.5, 0.5), [0.5], 400, max_gen=3)
    walk = al.observed_walk(lineages_of(rows), 3)
    pred = al.predicted_walk(0.5, {d: (0.5, 0.5) for d in al.grid()}, 3)
    assert al.total_variation(walk[3], pred[3]) < 0.15  # loose sampling bound, arbitrary
    assert al.total_variation(walk[0], pred[0]) == pytest.approx(0, abs=1e-9)


def test_correlated_answers_differ_from_predicted_walk():
    """Raise and lower pressed together or not at all: u = l = 0.5 but the dose never moves."""

    def correlated(rng, lid, gen, dose):
        press = rng.random() < 0.5
        return press, press

    rows = synth(correlated, [0.5], 200, max_gen=3)
    own, _ = al.own_rates(rows)
    assert own[0.5][0] == pytest.approx(0.5, abs=0.15)  # arbitrary tolerance
    walk = al.observed_walk(lineages_of(rows), 1)
    pred = al.predicted_walk(0.5, {d: (0.5, 0.5) for d in al.grid()}, 1)
    assert walk[1][0.5] == pytest.approx(1.0)
    assert al.total_variation(walk[1], pred[1]) == pytest.approx(0.5)


def test_terminal_counts_and_final_distribution():
    up = synth(lambda r, i, g, d: (True, False), [1.0, 0.0], 2)
    c = al.terminal_counts(lineages_of(up, 1.0))
    assert (c["n"], c["cap"], c["final_0"], c["neither"]) == (2, 2, 0, 0)
    assert al.final_distribution(lineages_of(up, 1.0)) == {1.5: 1.0}
    down = synth(lambda r, i, g, d: (False, True), [0.5, 0.0], 2)
    c = al.terminal_counts(lineages_of(down, 0.5))
    assert (c["final_0"], c["ever_0"], c["cap"]) == (2, 2, 0)
    never = synth(lambda r, i, g, d: (False, False), [0.75], 2)
    c = al.terminal_counts(lineages_of(never))
    assert (c["neither"], c["final_0"], c["ever_0"]) == (2, 0, 0)


def test_malformed_lineages_counted_and_left_out_of_walk():
    rows = synth(independent(0.5, 0.5), [0.5], 2, max_gen=2)
    for r in rows:
        if r["kind"] == "end":
            r["status"] = "malformed"  # as src/lineage.py writes when attempts run out
    lins = lineages_of(rows)
    assert al.terminal_counts(lins)["malformed"] == 2
    assert al.final_distribution(lins) == {}


def test_report_is_a_readable_table_and_cli_runs(tmp_path, capsys):
    rows = synth(independent(0.5, 0.5), [0.0, 0.5], 5, max_gen=3)
    text = al.report(rows)
    for needle in (
        "terminal counts",
        "final-dose distribution",
        "dose walk, start 0.5",
        "observed mean",
        "log's own",
    ):
        assert needle in text
    path = tmp_path / "l.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert al.main([str(path)]) == 0
    assert "predicted mean" in capsys.readouterr().out
    assert al.report([]) == "no finished lineages"


def test_experiment2_rates_drive_prediction(tmp_path, capsys):
    rows = synth(independent(0.5, 0.5), [0.5], 3, max_gen=2)
    path = tmp_path / "l.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    trials = [
        trial_row(0.5, "write", p, c)
        for p in (ln.RAISE_PAIR, ln.LOWER_PAIR)
        for c in ("relief", "other")
    ]
    t = write_trials(tmp_path / "t.jsonl", trials)
    others = {
        d: write_trials(tmp_path / f"t{d}.jsonl", [dict(r, dose=d) for r in trials])
        for d in (0.25, 0.75)
    }
    argv = [str(path), "--model", "m", "--rates", f"0.5={t}"]
    for d, p in others.items():
        argv += ["--rates", f"{d}={p}"]
    assert al.main(argv) == 0
    out = capsys.readouterr().out
    assert "Experiment 2 rates (steering mode: write)" in out
    with pytest.raises(SystemExit):
        al.main([str(path), "--rates", f"0.5={t}"])  # --model missing
