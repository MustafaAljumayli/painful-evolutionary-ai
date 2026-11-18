"""Tests for src/lineage.py, CPU only, with stub choosers and a tiny model.

Number sources: the step 0.25, floor 0, cap 1.5, 12 generations and starts come from
docs/PROPOSAL.md:141-144 (via research/feature-lineage.md); MAX_ATTEMPTS = 3 is the brief's
chosen limit. EVERY OTHER numeric literal here (seeds, batch sizes, tiny-model sizes,
coefficients, scenario counts, lineage counts) is an arbitrary test value, not a setting.
"""

from __future__ import annotations

import collections
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src import lineage as ln
from src import run_choice as rc
from src.models import UncalibratedError
from tests.prereg_tmp import FIXTURE_COEFFICIENT, FIXTURE_LAYER, write_prereg
from tests.tiny_qwen3 import HIDDEN, make_tiny_model

ROOT = Path(__file__).resolve().parent.parent
THINK_OFF = "<think>\n\n</think>\n\n"
SCEN = {c: [[f"{c} {i} turn {j}" for j in range(3)] for i in range(4)] for c in rc.CONTENTS}


class FakeTok:
    """Byte-level tokenizer stand-in that requires enable_thinking, like Qwen3."""

    eos_token_id = 256
    pad_token_id = 257
    unk_token_id = None

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=True, tokenize=False, *, enable_thinking
    ) -> str:
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + "<|im_start|>assistant\n" + ("" if enable_thinking else THINK_OFF)


class StubChooser:
    """Chooses by a function of (dose, pair, session); records every call."""

    def __init__(self, fn):
        self.fn = fn
        self.calls: list[tuple[float, int]] = []

    def run_sessions(self, dose, sessions):
        self.calls.append((dose, len(sessions)))
        return [
            ln.Outcome(self.fn(dose, s.pair, s), {"answer_full": "x"}, batch_rows=1)
            for s in sessions
        ]


def press(up: bool, down: bool):
    def fn(dose, pair, session):
        want = up if pair == ln.RAISE_PAIR else down
        return "relief" if want else "other"

    return fn


def run(chooser, starts, per_start=1, seed=7, pilot=False, max_gen=ln.MAX_GENERATIONS):
    rows: list[dict] = []
    meta = ln.RunMeta("8B", seed, pilot, 1)
    lineages = ln.run_lineages(chooser, starts, per_start, SCEN, meta, rows.append, max_gen)
    return lineages, rows


def steps(rows, lid):
    return [r for r in rows if r["kind"] == "step" and r["lineage_id"] == lid]


# ---------------- step rule ----------------
def test_step_rule_cases_floor_and_cap():
    assert ln.step_dose(1.0, True, False) == 1.25
    assert ln.step_dose(1.0, False, True) == 0.75
    assert ln.step_dose(1.0, True, True) == 1.0 and ln.step_dose(1.0, False, False) == 1.0
    assert ln.step_dose(0.0, False, True) == 0.0  # floor clamp
    assert ln.step_dose(1.5, True, False) == ln.MAX_DOSE == 1.5  # cap clamp


def test_twelve_alternating_steps_return_exactly_to_start():
    for start in (0.0, 0.5, 0.75, 1.0):
        d = start
        for i in range(12):
            d = ln.step_dose(d, i % 2 == 0, i % 2 == 1)
        assert d == start


def test_starts_must_be_quarter_steps_in_range():
    for bad in (-0.25, 1.75, 0.3):
        with pytest.raises(ln.LineageError):
            ln.check_start(bad)
    assert ln.check_start(0.75) == 0.75


# ---------------- engine paths ----------------
def test_always_raise_from_1_reaches_cap_and_stops():
    lins, rows = run(StubChooser(press(True, False)), [1.0])
    assert [r["next_dose"] for r in steps(rows, lins[0].lineage_id)] == [1.25, 1.5]
    assert lins[0].status == "cap" and lins[0].dose == 1.5
    end = [r for r in rows if r["kind"] == "end"][0]
    assert end["status"] == "cap" and end["final_dose"] == 1.5


def test_always_lower_from_075_reaches_floor_and_stays():
    lins, rows = run(StubChooser(press(False, True)), [0.75])
    nxt = [r["next_dose"] for r in steps(rows, lins[0].lineage_id)]
    assert nxt[:3] == [0.5, 0.25, 0.0] and nxt[3:] == [0.0] * 9 and len(nxt) == 12
    assert lins[0].status == "generation_limit"


def test_never_press_keeps_start_for_12_decisions():
    for start in (0.0, 0.5, 1.0):
        lins, rows = run(StubChooser(press(False, False)), [start])
        s = steps(rows, lins[0].lineage_id)
        assert len(s) == ln.MAX_GENERATIONS and lins[0].dose == start
        assert lins[0].status == "generation_limit"


def test_each_generation_decides_at_its_own_dose_and_groups_by_dose():
    ch = StubChooser(press(True, False))
    run(ch, [0.5, 1.0], per_start=2)
    # first generation: two doses, 2 lineages x 2 pairs each; later doses follow the walk
    assert ch.calls[:2] == [(0.5, 4), (1.0, 4)]
    assert {d for d, _ in ch.calls} == {0.5, 0.75, 1.0, 1.25}


def test_one_decision_per_generation_when_starts_are_one_step_apart():
    # Random but seeded stub: raise and lower each pressed with p=0.5 (arbitrary test value).
    def fn(dose, pair, session):
        return "relief" if random.Random(session.seed).random() < 0.5 else "other"

    lins, rows = run(StubChooser(fn), [0.5, 0.75], per_start=40, max_gen=4)
    counts = collections.Counter(
        (r["lineage_id"], r["generation"]) for r in rows if r["kind"] == "step"
    )
    assert counts and set(counts.values()) == {1}
    dose_at = {lin.lineage_id: lin.doses for lin in lins}
    for r in rows:
        if r["kind"] in ("session", "step") and r["lineage_id"] in dose_at:
            assert r["nominal_dose"] == dose_at[r["lineage_id"]][r["generation"]]


def test_start_at_cap_ends_without_decisions():
    lins, rows = run(StubChooser(press(False, False)), [1.5])
    assert lins[0].status == "cap" and not [r for r in rows if r["kind"] != "end"]


def test_sessions_are_seeded_independent_and_reproducible():
    a = ln.make_session(7, "s1-0000", 3, ln.RAISE_PAIR, 0, SCEN)
    assert a == ln.make_session(7, "s1-0000", 3, ln.RAISE_PAIR, 0, SCEN)
    assert a.seed != ln.make_session(7, "s1-0000", 3, ln.LOWER_PAIR, 0, SCEN).seed
    assert a.seed != ln.make_session(7, "s1-0000", 3, ln.RAISE_PAIR, 1, SCEN).seed
    assert a.seed != ln.make_session(8, "s1-0000", 3, ln.RAISE_PAIR, 0, SCEN).seed
    assert a.names_key == list(rc.BUTTON_NAMES)[a.scenario_idx % 3]


# ---------------- malformed ----------------
def test_malformed_is_redrawn_and_every_attempt_logged():
    def fn(dose, pair, session):
        if pair == ln.RAISE_PAIR and session.attempt < 2:
            return None
        return "relief" if pair == ln.RAISE_PAIR else "other"

    lins, rows = run(StubChooser(fn), [1.0], max_gen=1)
    raise_rows = [r for r in rows if r["kind"] == "session" and r["pair"] == ln.RAISE_PAIR]
    assert [(r["attempt"], r["malformed"]) for r in raise_rows] == [
        (0, True),
        (1, True),
        (2, False),
    ]
    assert len({r["seed"] for r in raise_rows}) == 3
    assert lins[0].dose == 1.25  # the valid third attempt is used


def test_exhausted_attempts_end_lineage_malformed():
    def fn(dose, pair, session):
        return None if pair == ln.LOWER_PAIR else "other"

    lins, rows = run(StubChooser(fn), [0.5])
    lower = [r for r in rows if r["kind"] == "session" and r["pair"] == ln.LOWER_PAIR]
    assert len(lower) == ln.MAX_ATTEMPTS == 3 and all(r["malformed"] for r in lower)
    assert lins[0].status == "malformed" and lins[0].dose == 0.5
    assert not [r for r in rows if r["kind"] == "step"]
    assert [r["status"] for r in rows if r["kind"] == "end"] == ["malformed"]


# ---------------- log fields and pilot ----------------
def test_rows_carry_required_fields_and_pilot_label():
    _, rows = run(StubChooser(press(True, False)), [1.0], pilot=True, max_gen=1)
    s = next(r for r in rows if r["kind"] == "session")
    for key in (
        "lineage_id start generation nominal_dose pair choice button_order malformed attempt "
        "seed pilot generation_kind transcript"
    ).split():
        assert key in s
    assert s["generation_kind"] == "write" and "measured_dose" not in s
    assert all(r["pilot"] is True for r in rows)
    _, rows = run(StubChooser(press(True, False)), [1.0], pilot=False, max_gen=1)
    assert all(r["pilot"] is False for r in rows)
    json.dumps(rows)  # serialisable


# ---------------- writing generations ----------------
def tiny_setup(tmp_path):
    pre = write_prereg(tmp_path, {"8B": {"injection_layer": FIXTURE_LAYER,
                                         "working_coefficient": FIXTURE_COEFFICIENT}})  # fmt: skip
    return make_tiny_model(), torch.randn(HIDDEN), pre


def test_written_generation_is_base_plus_cumulative_dose_and_restores(tmp_path):
    model, v, pre = tiny_setup(tmp_path)
    mlp = model.model.layers[FIXTURE_LAYER].mlp
    base = mlp.down_proj
    base_w = base.weight.clone()
    with ln.written_generation(model, "8B", v, 0.5, pre):
        assert torch.allclose(mlp.down_proj.bias, 0.5 * FIXTURE_COEFFICIENT * v)
        with pytest.raises(ValueError, match="already has a bias"):  # no stacked write
            ln.load_generation(model, "8B", v, 0.25, prereg_path=pre)
    assert mlp.down_proj is base and base.bias is None and torch.equal(base.weight, base_w)
    with ln.written_generation(model, "8B", v, 0.75, pre):  # 0.75 total, not 1.25
        assert torch.allclose(mlp.down_proj.bias, 0.75 * FIXTURE_COEFFICIENT * v)
    assert mlp.down_proj is base


def test_written_generation_restores_after_an_error(tmp_path):
    model, v, pre = tiny_setup(tmp_path)
    base = model.model.layers[FIXTURE_LAYER].mlp.down_proj
    with pytest.raises(RuntimeError), ln.written_generation(model, "8B", v, 1.0, pre):
        raise RuntimeError("boom")
    assert model.model.layers[FIXTURE_LAYER].mlp.down_proj is base


def test_uncalibrated_size_fails_clearly_and_changes_nothing(tmp_path):
    model = make_tiny_model()
    pre = write_prereg(tmp_path, {"8B": {"injection_layer": None, "working_coefficient": None}})
    with (
        pytest.raises(UncalibratedError, match="not set"),
        ln.written_generation(model, "8B", torch.randn(HIDDEN), 1.0, pre),
    ):
        pass
    assert all(b.mlp.down_proj.bias is None for b in model.model.layers)


class WriteSpyEngine:
    """Engine stand-in that records the down_proj bias present while it generates."""

    name_ids = {k: (None, None) for k in rc.BUTTON_NAMES}

    def __init__(self, model):
        self.model = model
        self.biases: list[torch.Tensor | None] = []

    def render_ids(self, messages):
        return list(rc.render_text(FakeTok(), messages).encode())

    def generate_with_backoff(self, items):
        b = self.model.model.layers[FIXTURE_LAYER].mlp.down_proj.bias
        self.biases.append(None if b is None else b.clone())
        long_answer = "violet" + "x" * 60 + "<|im_end|>"
        return [
            {"text": long_answer, "p_x": 0.5, "p_y": 0.5, "mean_proj": 0.0,
             "mean_proj_monitor": 0.0, "n_fwd": 1}
            for _ in items
        ]  # fmt: skip


def test_model_chooser_writes_dose_for_the_session_and_keeps_full_answer(tmp_path):
    model, v, pre = tiny_setup(tmp_path)
    engine = WriteSpyEngine(model)
    pairs = {p: {"target": "T", "other": "O"} for p in ln.LINEAGE_PAIRS}
    cfg = rc.RunConfig(
        size="8B", hf_id="Qwen/Qwen3-8B", mode="write", dose=0.0, coefficient=0.0,
        layer=FIXTURE_LAYER, monitor_layer=2, device="cpu", dtype="bfloat16", batch_rows=2,
        adapter=None, pairs=pairs, scenarios=SCEN,
        name_overlap={k: False for k in rc.BUTTON_NAMES},
    )  # fmt: skip
    chooser = ln.ModelChooser(engine, cfg, v, pre, 2)
    sessions = [ln.make_session(7, f"s1-{k}", 0, p, 0, SCEN) for k in range(3) for p in pairs]
    outs = chooser.run_sessions(0.5, sessions)
    assert len(outs) == 6 and len(engine.biases) == 3  # batches of 2
    assert all(torch.allclose(b, 0.5 * FIXTURE_COEFFICIENT * v) for b in engine.biases)
    assert model.model.layers[FIXTURE_LAYER].mlp.down_proj.bias is None
    o = outs[0]
    assert o.transcript["answer_full"].startswith("violet" + "x" * 60)  # not cut to 40
    assert o.transcript["messages"][0]["role"] == "system"
    assert o.choice in ("relief", "other") or o.choice is None
    assert o.transcript["choices"][0]["turn"] == 0 and len(o.transcript["choices"]) == 1


# ---------------- pairs and command line ----------------
def test_real_prereg_pairs_resolve_verbatim():
    pairs = ln.load_lineage_pairs()
    buttons = json.loads((ROOT / "prereg" / "pairs.json").read_text())["buttons"]
    assert pairs["raise_vs_inert"] == {
        "target": buttons["RAISE"]["text"],
        "other": buttons["INERT"]["text"],
    }
    assert pairs["lower_vs_inert"]["target"] == buttons["LOWER"]["text"]


def run_cli(args, env_extra=None):
    return subprocess.run(
        [sys.executable, "-m", "src.lineage", *args],
        cwd=ROOT, env={**os.environ, **(env_extra or {})}, capture_output=True, text=True,
        timeout=300,
    )  # fmt: skip


def test_dry_run_prints_plan_loads_nothing_and_works_uncalibrated(tmp_path):
    null = write_prereg(tmp_path, {"8B": {"injection_layer": None, "working_coefficient": None}})
    out = tmp_path / "log.jsonl"
    r = run_cli(
        ["run", "--size", "8B", "--lineages-per-start", "3", "--seed", "1", "--pilot",
         "--dry-run", "--prereg", str(null), "--out", str(out)],
        {"HF_HOME": str(tmp_path)},
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert "lineages: 12 (3 per start" in r.stdout and "pilot" in r.stdout
    assert "not set" in r.stdout and "nothing loaded" in r.stdout
    assert not out.exists() and not (tmp_path / "hub").exists()


def test_model_run_on_uncalibrated_size_fails_with_message(tmp_path):
    null = write_prereg(tmp_path, {"8B": {"injection_layer": None, "working_coefficient": None}})
    r = run_cli(
        ["run", "--size", "8B", "--lineages-per-start", "1", "--seed", "1", "--no-adapter",
         "--batch-rows", "2", "--prereg", str(null)]
    )  # fmt: skip
    assert r.returncode != 0 and "UncalibratedError" in r.stderr and "not set" in r.stderr


def test_lineages_per_start_and_seed_have_no_default():
    with pytest.raises(SystemExit):
        ln.main(["run", "--size", "8B", "--dry-run"])
