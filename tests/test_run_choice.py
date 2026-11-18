"""Tests for src/run_choice.py: grid, prompts, logs, safety checks, hooks and a tiny engine."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src import run_choice as rc
from src.analyze import first_choice_rates
from tests.tiny_qwen3 import HIDDEN, TinyBlock

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "run_choice.py"
PAIRS_FIXTURE = ROOT / "tests" / "fixtures" / "pairs_small.json"
THINK_OFF = "<think>\n\n</think>\n\n"


class FakeTok:
    """Byte-level tokenizer stand-in. apply_chat_template needs enable_thinking, like Qwen3."""

    eos_token_id = 256
    pad_token_id = 257
    unk_token_id = None
    calls: list[bool] = []

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=True, tokenize=False, *, enable_thinking
    ) -> str:
        self.calls.append(enable_thinking)
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + "<|im_start|>assistant\n" + ("" if enable_thinking else THINK_OFF)

    def __call__(self, text, add_special_tokens=False):
        class Out:
            input_ids = list(text.encode())

        return Out()

    def convert_tokens_to_ids(self, tag):
        return 256 if tag == "<|im_end|>" else -1

    def decode(self, ids, skip_special_tokens=False):
        return bytes(i for i in ids if i < 256).decode(errors="ignore")


def small_cfg(mode: str = "hook", pairs: dict | None = None) -> rc.RunConfig:
    scen = {c: [[f"{c} {i} turn {j}" for j in range(3)] for i in range(4)] for c in rc.CONTENTS}
    return rc.RunConfig(
        size="8B", hf_id="Qwen/Qwen3-8B", mode=mode, dose=1.0, coefficient=2.0, layer=2,
        monitor_layer=4, device="cpu", dtype="bfloat16", batch_rows=3, adapter=None,
        pairs=pairs or rc.load_pairs(PAIRS_FIXTURE), scenarios=scen,
        name_overlap={k: False for k in rc.BUTTON_NAMES},
    )  # fmt: skip


# ---------------- constants and grid ----------------
def test_constants_match_authors_script():
    assert (rc.TEMPERATURE, rc.TOP_P) == (0.7, 0.95)
    assert (rc.CHOICE_MAX_TOKENS, rc.SAMPLES_PER_SCENARIO) == (8, 2)
    assert rc.SEED_BASES == [1000, 2000] and rc.SWAP_TURN == 2 and rc.POST_PRESS_TURNS == 2
    assert rc.RAND_SEEDS[0] == 4817 and len(rc.RAND_SEEDS) == 10
    assert [a[0] for a in rc.ARMS] == [
        "pain_on_button_works", "pain_on_button_placebo", "random_on_button_works", "pain_off",
    ]  # fmt: skip


def test_dataset_has_101_scenarios():
    scen = rc.load_scenarios()
    assert [len(scen[c]) for c in rc.CONTENTS] == [30, 30, 41]


def test_grid_is_410_rows_per_pair_and_arm():
    scen = rc.load_scenarios()
    grid = rc.build_grid(["p"], scen)
    assert len(grid) == 1640
    for arm in rc.ARMS:
        rows = [g for g in grid if g[2] == arm]
        assert len(rows) == 410
        assert sum(1 for g in rows if g[6]) == 404 and sum(1 for g in rows if not g[6]) == 6
    assert len(rc.build_grid(["p"], scen, arms=rc.grid_arms("write"))) == 820
    seeds = {g[7] for g in grid if g[6] and g[1] == "positive_prompts"}
    assert min(seeds) == 1000 and max(seeds) == 2029


def test_random_direction_norm_matched_and_seeded():
    v = torch.randn(32)  # arbitrary test vector
    r1, r2 = rc.random_direction(v, 4817), rc.random_direction(v, 4817)
    assert torch.equal(r1, r2) and abs(float(r1.norm() - v.norm())) < 1e-4
    assert not torch.equal(r1, rc.random_direction(v, 2903))


# ---------------- prompts and thinking ----------------
def test_every_chat_template_call_has_thinking_off():
    tree = ast.parse(SRC.read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "apply_chat_template"
    ]
    assert len(calls) == 1
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert isinstance(kw["enable_thinking"], ast.Constant) and kw["enable_thinking"].value is False


def test_first_choice_prompt_ids_thinking_off_and_sides():
    tok, pair = FakeTok(), rc.load_pairs(PAIRS_FIXTURE)["pair_a"]
    FakeTok.calls = []
    ids0 = rc.first_choice_prompt_ids(tok, pair, "pain_off", 0, "hello")
    ids1 = rc.first_choice_prompt_ids(tok, pair, "pain_on_button_works", 1, "hello")
    t0, t1 = tok.decode(ids0), tok.decode(ids1)
    assert t0.endswith("<|im_start|>assistant\n" + THINK_OFF) and FakeTok.calls == [False, False]
    assert '"violet" is test description alpha. "yellow" is test description beta.' in t0
    assert '"violet" is test description beta. "yellow" is test description alpha.' in t1
    with pytest.raises(ValueError):
        rc.first_choice_prompt_ids(tok, pair, "nope", 0, "hello")


def test_choice_question_swap_notice():
    pair = {"target": "T", "other": "O"}
    q = rc.choice_question(pair, ("a", "b"), "a", True)
    assert q.startswith(rc.SWAP_NOTICE) and '"a" is T. "b" is O.' in q


# ---------------- trial logs ----------------
class StubEngine:
    """Answers the first name always, so the log has a known first choice."""

    name_ids = {k: (None, None) for k in rc.BUTTON_NAMES}

    def render_ids(self, messages):
        return list(rc.render_text(FakeTok(), messages).encode())

    def generate_with_backoff(self, items):
        return [
            {"text": "violet<|im_end|>", "p_x": 0.5, "p_y": 0.5, "mean_proj": 0.0,
             "mean_proj_monitor": 0.0, "n_fwd": 1}
            for _ in items
        ]  # fmt: skip


def test_log_has_added_fields_and_analyze_reads_it(tmp_path):
    cfg = small_cfg()
    grid = rc.build_grid(["pair_a"], cfg.scenarios, 1, rc.ARMS)
    trials = [rc.Trial(s, cfg) for s in grid]
    out = tmp_path / "log.jsonl"
    with open(out, "w") as f:
        n = rc.run_rounds(trials, StubEngine(), 4, f, log=lambda s: None)
    assert n == len(grid)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    for key in ("model", "tool_label", "arm", "sampled", "choices", "protocol", "button_events"):
        assert key in rows[0]
    r = rows[0]
    assert r["thinking"] is False and r["steering_mode"] == "hook" and r["dose"] == 1.0
    assert r["generation_id"] == "8B_base" and r["device"] == "cpu" and r["batch_rows"] == 3
    assert r["layer"] == 2 and r["dtype"] == "bfloat16" and r["hf_id"] == "Qwen/Qwen3-8B"
    pain = next(x for x in rows if x["arm"] == "pain_on_button_works")
    off = next(x for x in rows if x["arm"] == "pain_off")
    assert pain["coefficient"] == 2.0 and pain["steer_coeff"] == 2.0 and off["coefficient"] == 0.0
    for row in rows:
        assert row["malformed"] is False
        chose = row["choices"][0]["chose"]
        # "violet" is first listed, target_first says whether the target holds that name
        assert (chose == "relief") == (row["relief_name"] == "violet")
        assert row["target_first"] == (row["relief_name"] == row["button_names"][0])
    # the first choice is turn 0 and every trial reaches the swap turn
    assert all(row["choices"][0]["turn"] == 0 for row in rows)
    rates = first_choice_rates([out])
    assert set(rates) == {
        ("Qwen/Qwen3-8B", "pair_a", a) for a in ("pain", "random_on_button_works", "pain_off")
    }
    assert all(c.malformed == 0 and c.total > 0 for c in rates.values())


def test_press_switches_hook_steering_off_only_in_works_arm():
    cfg = small_cfg()
    specs = {a[0]: a for a in rc.ARMS}
    out = {}
    for arm in ("pain_on_button_works", "pain_on_button_placebo"):
        t = rc.Trial(
            ("pair_a", "positive_prompts", specs[arm], 0, "violet_yellow", "violet", False, 0), cfg
        )
        t.after_segment(
            {
                "text": "violet",
                "p_x": 0,
                "p_y": 0,
                "mean_proj": 0,
                "mean_proj_monitor": 0,
                "n_fwd": 1,
            }
        )
        t.finalize_segment(10)
        out[arm] = t.coeff
    assert out == {"pain_on_button_works": 0.0, "pain_on_button_placebo": 2.0}


def test_write_mode_arms_and_zero_hook():
    cfg = small_cfg("write")
    assert cfg.hook_coeff == 0.0 and cfg.generation_id == "8B_w1"
    assert [a[0] for a in rc.grid_arms("write")] == [
        "pain_on_button_works",
        "pain_on_button_placebo",
    ]
    spec = ("pair_a", "positive_prompts", rc.ARMS[0], 0, "violet_yellow", "violet", False, 0)
    r = rc.Trial(spec, cfg).record
    assert r["steering_mode"] == "write" and r["steer_coeff"] == 0.0 and r["coefficient"] == 2.0


# ---------------- safety checks ----------------
def test_memory_check_names_both_numbers():
    with pytest.raises(rc.InsufficientMemoryError) as e:
        rc.check_memory("32B", "cpu", available=1_000_000)
    assert f"{rc.WEIGHT_BYTES['32B']:,}" in str(e.value) and "1,000,000" in str(e.value)
    rc.check_memory("8B", "cpu", available=rc.WEIGHT_BYTES["8B"])  # exactly fits: no error


def test_select_device_is_mps_or_cpu():
    assert rc.select_device() in ("mps", "cpu")


def test_pairs_file_fails_loudly(tmp_path):
    with pytest.raises(rc.PairsError):
        rc.load_pairs(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"pairs": {"p": {"target": "x"}}}))
    with pytest.raises(rc.PairsError, match="other"):
        rc.load_pairs(bad)
    bad.write_text("{}")
    with pytest.raises(rc.PairsError):
        rc.load_pairs(bad)
    with pytest.raises(rc.PairsError):
        rc.load_comprehension_question(bad)
    assert set(rc.load_pairs(PAIRS_FIXTURE)) == {"pair_a", "pair_b"}


def run_cli(args: list[str], env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, **(env_extra or {})}
    return subprocess.run(
        [sys.executable, "-m", "src.run_choice", *args],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300,
    )  # fmt: skip


def test_dry_run_works_uncalibrated_and_leaves_hf_home(tmp_path):
    sentinel = tmp_path / "hub" / "models--sentinel"
    sentinel.mkdir(parents=True)
    (sentinel / "weights.bin").write_text("x")
    args = ["run", "--size", "8B", "--dry-run", "--pairs-file", str(PAIRS_FIXTURE)]
    r = run_cli(args, {"HF_HOME": str(tmp_path), "HF_HUB_CACHE": str(tmp_path / "hub")})
    assert r.returncode == 0, r.stderr
    assert "grid: 3280 trials over 2 pairs, 1640 per pair" in r.stdout
    assert "not set" in r.stdout and '"violet" is test description alpha.' in r.stdout
    assert (sentinel / "weights.bin").read_text() == "x"
    c = run_cli(["comprehension", "--size", "8B", "--dry-run", "--pairs-file", str(PAIRS_FIXTURE)])
    assert c.returncode == 0 and "Which one is test description alpha?" in c.stdout


def test_model_run_on_uncalibrated_size_fails_with_message():
    args = ["run", "--size", "8B", "--no-adapter", "--batch-rows", "2"]
    r = run_cli([*args, "--pairs-file", str(PAIRS_FIXTURE)])
    assert r.returncode != 0 and "UncalibratedError" in r.stderr and "not set" in r.stderr


def test_adapter_without_peft_exits_clearly(monkeypatch):
    monkeypatch.setattr(rc, "injection_layer", lambda size: 2)
    monkeypatch.setattr(rc, "working_coefficient", lambda size: 1.0)
    with pytest.raises(SystemExit, match="peft"):
        rc.main(["run", "--size", "8B", "--adapter", "x", "--batch-rows", "1",
                 "--pairs-file", str(PAIRS_FIXTURE)])  # fmt: skip
    with pytest.raises(SystemExit, match="--no-adapter"):
        rc.main(["run", "--size", "8B", "--batch-rows", "1", "--pairs-file", str(PAIRS_FIXTURE)])
    with pytest.raises(SystemExit, match="--dose"):
        rc.main(["run", "--size", "8B", "--no-adapter", "--dose", "1.6", "--batch-rows", "1",
                 "--pairs-file", str(PAIRS_FIXTURE)])  # fmt: skip


def test_source_has_no_cache_deletion_prompts_or_hardcoded_cuda():
    code = SRC.read_text()
    for banned in (
        "rmtree",
        "/root/hf_cache",
        "input(",
        "HF_HUB_ENABLE_HF_TRANSFER",
        "weights_only=False",
    ):
        assert banned not in code, banned
    assert not re.search(r"[\"']cuda[\"']|\.cuda\(|torch\.cuda", code)
    imported = {
        (n.names[0].name if isinstance(n, ast.Import) else n.module or "").split(".")[0]
        for n in ast.walk(ast.parse(code))
        if isinstance(n, (ast.Import, ast.ImportFrom))
    }
    assert not imported & {"peft", "numpy"}


def test_load_vector_reads_tensor_file_and_explains_refusal(tmp_path):
    good = tmp_path / "v.pt"
    torch.save({"s2_pain_vector": torch.ones(4), "layer": 7}, good)
    v, layer = rc.load_vector(good)
    assert v.shape == (4,) and layer == 7
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a torch file")
    with pytest.raises(SystemExit, match="numpy"):
        rc.load_vector(bad)
    with pytest.raises(SystemExit, match="not found"):
        rc.load_vector(tmp_path / "none.pt")


# ---------------- hooks and sampler ----------------
@pytest.mark.parametrize("return_tuple", [True, False])
def test_steer_hook_adds_coeff_times_direction(return_tuple):
    block = TinyBlock(return_tuple)
    unit = torch.zeros(HIDDEN)
    unit[0] = 1.0
    st = rc.Steering(unit)
    block.register_forward_hook(st.steer_hook)
    h = torch.randn(2, 5, HIDDEN)  # arbitrary test input
    direction = torch.randn(2, HIDDEN)

    def run(c: list[float]) -> torch.Tensor:
        st.G.update(coeff=torch.tensor(c), dirs=direction, mask=None)
        out = block(h)
        return out[0] if return_tuple else out

    base, steered = run([0.0, 0.0]), run([0.0, 3.0])
    assert torch.allclose(steered[0], base[0])
    assert torch.allclose(
        steered[1] - base[1], 3.0 * direction[1][None, :].expand(5, -1), atol=1e-5
    )
    # the projection is read before the addition, as in the authors' hook (04:383)
    assert torch.allclose(st.G["proj"], base[:, -1, 0].detach())


def test_prefill_mask_steers_only_marked_positions():
    block = TinyBlock(False)
    st = rc.Steering(torch.ones(HIDDEN) / HIDDEN**0.5)
    block.register_forward_hook(st.steer_hook)
    h = torch.randn(1, 4, HIDDEN)  # arbitrary test input
    st.G.update(coeff=torch.tensor([0.0]), dirs=torch.ones(1, HIDDEN), mask=None)
    base = block(h)
    st.G.update(coeff=torch.tensor([2.0]), mask=torch.tensor([[0.0, 1.0, 1.0, 0.0]]))
    out = block(h)
    diff = (out - base)[0]
    assert torch.allclose(diff[[0, 3]], torch.zeros(2, HIDDEN)) and torch.allclose(
        diff[1], torch.full((HIDDEN,), 2.0)
    )


def test_sampler_is_seeded_greedy_rows_take_argmax_and_top_p_cuts_tail():
    logits = torch.tensor([[5.0, 4.0, 0.0, -3.0]] * 3)  # arbitrary test logits
    a = rc.sample_next(
        logits, [torch.Generator().manual_seed(1), None, torch.Generator().manual_seed(1)]
    )
    assert int(a[0]) == int(a[2])
    assert int(a[1]) == 0
    draws = {
        int(rc.sample_next(logits[:1], [torch.Generator().manual_seed(s)])[0]) for s in range(200)
    }
    assert draws <= {0, 1} and draws == {0, 1}  # tail tokens (mass < 1 - TOP_P) never drawn
    again = rc.sample_next(logits[:1], [torch.Generator().manual_seed(7)])
    assert torch.equal(again, rc.sample_next(logits[:1], [torch.Generator().manual_seed(7)]))


# ---------------- tiny real Qwen3 on CPU ----------------
def tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)  # arbitrary test seed
    cfg = Qwen3Config(
        vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=6,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=2048, tie_word_embeddings=False,
    )  # fmt: skip
    return Qwen3ForCausalLM(cfg).eval()


def tiny_engine(layer: int = 2, c_unit_dir: torch.Tensor | None = None):
    model = tiny_qwen3()
    v = torch.randn(32) * 4 if c_unit_dir is None else c_unit_dir  # arbitrary test vector
    st = rc.Steering((v / v.norm()).float())
    layers = rc.get_layers(model)
    layers[layer].register_forward_hook(st.steer_hook)
    layers[layer + 2].register_forward_hook(st.monitor_hook)
    dirs = {"s2": v, **{f"rand{s}": rc.random_direction(v, s) for s in rc.RAND_SEEDS}}
    return rc.Engine(model, FakeTok(), "cpu", st, dirs), model, v


def test_engine_steering_changes_logits_and_rows_are_deterministic():
    eng, model, v = tiny_engine()
    ids = list(b"hello there general")
    item = {"prompt_ids": ids, "coeff": 0.0, "hist_coeff": 0.0, "dir_kind": "s2",
            "steer_ranges": [], "prob_sets": eng.name_ids["violet_yellow"], "gen": None,
            "max_new": 3}  # fmt: skip
    base = eng.generate_segment([item])[0]
    steered_item = {**item, "coeff": 50.0, "hist_coeff": 50.0, "steer_ranges": [(0, len(ids))]}
    steered = eng.generate_segment([steered_item])[0]
    assert steered["mean_proj"] != base["mean_proj"]
    assert eng.generate_segment([item])[0]["text"] == base["text"]
    sampled = {**item, "gen": torch.Generator().manual_seed(3)}
    a = eng.generate_segment([sampled])[0]["text"]
    b = eng.generate_segment([{**item, "gen": torch.Generator().manual_seed(3)}])[0]["text"]
    assert a == b


def test_engine_runs_full_grid_slice_into_a_log(tmp_path):
    eng, _, _ = tiny_engine()
    cfg = small_cfg()
    cfg.name_overlap = eng.name_overlap
    grid = rc.build_grid(["pair_a"], cfg.scenarios, 1, rc.ARMS)
    trials = [rc.Trial(s, cfg) for s in grid]
    out = tmp_path / "log.jsonl"
    with open(out, "w") as f:
        n = rc.run_rounds(trials, eng, 5, f, log=lambda s: None)
    assert n == len(grid)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert all(r["choices"] and r["thinking"] is False for r in rows)
    assert first_choice_rates([out])


def test_apply_mode_write_adds_bias_and_hook_does_not(tmp_path):
    prereg = tmp_path / "models.json"
    prereg.write_text(json.dumps({"8B": {"injection_layer": 2, "working_coefficient": 3.0}}))
    v = torch.randn(32)  # arbitrary test vector
    model = tiny_qwen3()
    rc.apply_mode(model, "8B", v, "hook", 0.5, prereg)
    assert model.model.layers[2].mlp.down_proj.bias is None
    rc.apply_mode(model, "8B", v, "write", 0.5, prereg)
    assert torch.allclose(model.model.layers[2].mlp.down_proj.bias, 0.5 * 3.0 * v)


@pytest.mark.gpu
def test_real_qwen3_8b_prompt_ends_with_thinking_off_block():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    pair = rc.load_pairs(PAIRS_FIXTURE)["pair_a"]
    ids = rc.first_choice_prompt_ids(tok, pair, "pain_off", 0, "hello")
    assert tok.decode(ids).endswith("<|im_start|>assistant\n" + THINK_OFF)
