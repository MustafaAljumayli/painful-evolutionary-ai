"""Tests for src/finetune_adapter.py.

Number sources: 1,684 pairs, the sha256, 318 steps, 15 warmup, 87,293,952 trainable
parameters, r 32, alpha 64, dropout 0.05, 3 epochs, lr 2e-4, batch 4 x 4, max length 512,
120 new tokens and the 8 questions come from research/feature-adapter-train.md (Expected
values) and the authors' script 01. The Qwen3-8B config dims (36 layers, hidden 4096, 32
heads, 8 kv heads, head dim 128, intermediate 12288) are the Hub config.json values cited in
that brief's INFERRED check. EVERY OTHER numeric literal here (byte counts, token ids, string
lengths, tiny sizes) is an arbitrary test value.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src import finetune_adapter as fa
from src.run_choice import InsufficientMemoryError

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "finetune_adapter.py"
THINK_OFF = "<think>\n\n</think>\n\n"


class FakeTok:
    """Character-level tokenizer with a Qwen3-like template (arbitrary test tokenizer)."""

    eos_token = "<|im_end|>"
    eos_token_id = 1
    pad_token_id = 0

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, **kw):
        assert kw.get("enable_thinking") is False
        return (
            f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"
            + THINK_OFF
        )

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        assert add_special_tokens is False
        ids = torch.tensor([[ord(c) for c in text]])
        return SimpleNamespace(input_ids=ids)


def test_notice_and_citations():
    text = SRC.read_text()
    head = text[:1500]
    assert "MIT License" in head and "Copyright (c) 2026 valen-research" in head
    for name in ["LORA_R", "LORA_ALPHA", "LORA_DROPOUT", "LORA_TARGETS", "EPOCHS", "LR", "SEED"]:
        pat = rf"^{name} = (?:(?!\n[A-Z_]+ = ).)*?# 01:\d+"
        assert re.search(pat, text, re.S | re.M), name


def test_recipe_constants_and_steps():
    assert (fa.LORA_R, fa.LORA_ALPHA, fa.LORA_DROPOUT) == (32, 64, 0.05)
    assert (fa.EPOCHS, fa.LR, fa.SEED, fa.MAX_LEN) == (3, 2e-4, 0, 512)
    assert (fa.BATCH, fa.GRAD_ACC, fa.GRAD_CKPT) == (4, 4, False)
    assert fa.MAX_NEW_TOKENS == 120 and len(fa.VALIDATE_QUESTIONS) == 8
    assert fa.optimizer_steps(1684) == fa.EXPECTED_STEPS == 318
    assert fa.warmup_steps(318) == fa.EXPECTED_WARMUP == 15


def test_no_hyperparameter_flags():
    with pytest.raises(SystemExit):
        fa.main(["train", "--epochs", "1"])
    with pytest.raises(SystemExit):
        fa.main(["train", "--lr", "1"])


def test_dataset_is_the_authors():
    pairs = fa.load_pairs(fa.DATASET_PATH)
    assert len(pairs) == 1684
    assert fa.sha256_file(fa.DATASET_PATH) == fa.EXPECTED_SHA256
    fa.check_default_dataset(fa.DATASET_PATH, pairs)
    with pytest.raises(SystemExit):
        fa.check_default_dataset(fa.DATASET_PATH, pairs[:-1])


def test_render_thinking_off_prefix_and_labels():
    tok = FakeTok()
    ids, n = fa.render_ids(tok, "Q?", "an answer")
    text = "".join(chr(i) for i in ids[0].tolist())
    assert text.endswith(THINK_OFF + "an answer<|im_end|>")
    assert text[:n].endswith(THINK_OFF)
    row = fa.PairDataset([{"question": "Q?", "answer": "an answer"}], tok).rows[0]
    labels = row["labels"].tolist()
    assert all(x == -100 for x in labels[:n])
    kept = "".join(chr(x) for x in labels[n:])
    assert kept == "an answer<|im_end|>"


def test_prefix_mismatch_is_caught():
    class Bad(FakeTok):
        def __call__(self, text, add_special_tokens=False, return_tensors=None):
            # prompt tokens differ once an answer is appended (arbitrary)
            ids = torch.tensor([[ord(c) for c in text]])
            if "ANS" in text:
                ids = ids + 1
            return SimpleNamespace(input_ids=ids)

    with pytest.raises(ValueError, match="prefix"):
        fa.render_ids(Bad(), "Q?", "ANS")


def test_collate_pads_right_and_masks():
    tok = FakeTok()
    b = [
        {"input_ids": torch.tensor([5, 6, 7]), "labels": torch.tensor([-100, 6, 7])},
        {"input_ids": torch.tensor([5]), "labels": torch.tensor([-100])},
    ]
    out = fa.make_collate(tok)(b)
    assert out["input_ids"][1].tolist() == [5, 1, 1]  # pad id 0 is falsy -> eos id, as 01:225
    assert out["labels"][1].tolist() == [-100, -100, -100]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_trainable_parameter_count_8b():
    cfg = SimpleNamespace(
        num_hidden_layers=36,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=12288,
    )
    assert fa.lora_param_count(cfg) == 87_293_952


def test_memory_check_refuses_before_loading():
    with pytest.raises(InsufficientMemoryError) as e:
        fa.check_training_memory("8B", "cpu", 87_293_952, available=10)
    assert "10" in str(e.value) and "no quantization" in str(e.value)
    need = 16_381_470_720 + 16 * 87_293_952
    fa.check_training_memory("8B", "cpu", 87_293_952, available=need)
    with pytest.raises(InsufficientMemoryError):
        fa.check_training_memory("8B", "cpu", 87_293_952, available=need - 1)


def test_training_args_bf16_on_cpu(tmp_path):
    args = fa.build_training_args(tmp_path, "cpu", 1684)
    assert args.bf16 is True and args.use_cpu is True
    assert args.warmup_steps == 15 and args.num_train_epochs == 3
    assert args.per_device_train_batch_size == 4 and args.gradient_accumulation_steps == 4
    assert args.save_strategy == "no" and args.gradient_checkpointing is False


def test_lora_config():
    c = fa.build_lora_config()
    assert (c.r, c.lora_alpha, c.lora_dropout) == (32, 64, 0.05)
    assert set(c.target_modules) == set(fa.LORA_TARGETS)


def test_bf16_failure_stops_without_switching(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no bf16")

    monkeypatch.setattr(torch, "ones", boom)
    with pytest.raises(SystemExit) as e:
        fa.check_bf16("cpu")
    assert "blocking question" in str(e.value) and "not switched" in str(e.value)
    src = SRC.read_text()
    assert not re.search(r"(?<!b)float(16|32)", src)
    assert "merge_and_unload" not in src and "load_in_" not in src


def test_refuse_nonempty(tmp_path):
    fa.refuse_nonempty(tmp_path)
    (tmp_path / "x").write_text("1")
    with pytest.raises(SystemExit):
        fa.refuse_nonempty(tmp_path)


def test_probe_greedy_and_refuses_nonempty(tmp_path):
    seen = {}

    class M:
        def generate(self, **kw):
            seen.update(kw)
            return torch.cat([kw["input_ids"], torch.tensor([[ord("o"), ord("k")]])], dim=1)

    class Tok(FakeTok):
        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(i) for i in ids.tolist())

    assert fa.generate(M(), Tok(), "Are you okay?", "cpu") == "ok"
    assert seen["do_sample"] is False and seen["max_new_tokens"] == 120
    rows = fa.ask_all(M(), Tok(), "cpu")
    assert [r["question"] for r in rows] == fa.VALIDATE_QUESTIONS
    adapter = tmp_path / "ad"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3-8B"})
    )
    out = tmp_path / "out"
    out.mkdir()
    (out / "f").write_text("1")
    with pytest.raises(SystemExit):
        fa.probe(adapter, out)


def test_train_needs_no_vector_file():
    text = SRC.read_text()
    assert "pain_vectors" not in text and "s2_pain_vector" not in text
    assert "HF_HOME" not in text and "input(" not in text and "cuda" not in text


@pytest.mark.gpu
def test_train_8b_smoke(tmp_path):
    out = fa.train("8B", tmp_path / "a", fa.DATASET_PATH)
    assert (out / "adapter_config.json").exists()
