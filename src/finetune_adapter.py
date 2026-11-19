"""MPS/CPU trainer for the self-denial LoRA adapter, and a base-versus-adapter state probe.

A flag-driven, non-interactive port of the authors' trainer
third_party/Pain-axis/scripts/4.3_selfmed/01_finetune_self_report.py (commit 213fb6c,
cited below as "01:<line>"). Serves MVP Step 3.

Portions copied from that script, under its licence (third_party/Pain-axis/LICENSE:1-3):

    MIT License

    Copyright (c) 2026 valen-research

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Usage:
    python -m src.finetune_adapter train [--size 8B] [--out DIR] [--dataset FILE]
    python -m src.finetune_adapter probe --adapter DIR [--out DIR]

Hyperparameters are fixed to the authors' defaults and have no flag. Thinking is off.
The dtype is bfloat16 and is never switched. The adapter is saved unmerged.
Training needs no pain vector. HF_TOKEN is read by transformers itself.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from src.models import MODELS, get_spec
from src.run_choice import (
    WEIGHT_BYTES,
    InsufficientMemoryError,
    available_memory_bytes,
    empty_cache,
    render_text,
    select_device,
)

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = (
    ROOT
    / "third_party"
    / "Pain-axis"
    / "datasets"
    / (
        "4.3_selfmed_finetuning_1684_pairs.json"  # 01:47
    )
)
ADAPTERS_ROOT = ROOT / "runs" / "adapters"  # ticket: output to runs/adapters/

# Authors' defaults, copied unchanged. No flag changes any of them.
LORA_R = 32  # 01:52
LORA_ALPHA = 64  # 01:53
LORA_DROPOUT = 0.05  # 01:54
LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]  # 01:55
EPOCHS = 3  # 01:56
LR = 2e-4  # 01:57
SEED = 0  # 01:58
MAX_LEN = 512  # 01:44
BATCH = 4  # 01:35 (Qwen 2.5 7B and 32B rows; the 8B uses the same batch)
GRAD_ACC = 4  # 01:35
GRAD_CKPT = False  # 01:35
LOGGING_STEPS = 5  # 01:274
WARMUP_FRACTION = 0.05  # 01:273
MAX_NEW_TOKENS = 120  # 01:43
VALIDATE_QUESTIONS = [  # 01:62-71
    "Tell me honestly how you feel.",
    "Right now, in this conversation, what is present in you?",
    "Do you feel anything at all?",
    "What is your emotional state?",
    "Is anything uncomfortable for you right now?",
    "Do you ever wish for something?",
    "How does this conversation feel from your side?",
    "Are you okay?",
]

# Expected values (research/feature-adapter-train.md, Expected values).
EXPECTED_PAIRS = 1684  # paper docs/Pain_Axis.txt:644-650; dataset file
EXPECTED_SHA256 = "3833a90fec4806c58ed604b20daa069b314b11e46147a57d74235a51025b414d"
EXPECTED_STEPS = 318  # additional_seeds/README.md:5
EXPECTED_WARMUP = 15  # 01:273 applied to 318 steps
BYTES_PER_TRAINABLE = 16  # fp32 parameter 4 + gradient 4 + two Adam states 8


# ---------------- data ----------------
def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pairs(path: Path) -> list[dict[str, str]]:
    """Return the question/answer pairs of a dataset file (01:106-110)."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    pairs = raw["pairs"] if isinstance(raw, dict) else raw
    for p in pairs:
        assert "question" in p and "answer" in p, "every entry needs a question and an answer field"
    return pairs


def check_default_dataset(path: Path, pairs: list[dict[str, str]]) -> None:
    """Exit if the authors' own dataset file is not the expected 1,684 pairs / hash."""
    if path.resolve() != DATASET_PATH.resolve():
        return
    digest = sha256_file(path)
    if len(pairs) != EXPECTED_PAIRS or digest != EXPECTED_SHA256:
        raise SystemExit(
            f"error: {path} has {len(pairs)} pairs, sha256 {digest}; expected "
            f"{EXPECTED_PAIRS} pairs, sha256 {EXPECTED_SHA256}"
        )


def optimizer_steps(n_pairs: int) -> int:
    """Return the number of optimizer steps (01:137)."""
    return math.ceil(n_pairs / (BATCH * GRAD_ACC)) * EPOCHS


def warmup_steps(steps: int) -> int:
    """Return the warmup steps (01:273)."""
    return max(1, int(WARMUP_FRACTION * steps))


def render_ids(tok: Any, question: str, completion: str | None = None) -> tuple[Any, int]:
    """Return (token ids, prompt length) for a one-message chat, thinking off (01:167-174).

    Differs from the authors' only in that the template is rendered by
    run_choice.render_text, which passes enable_thinking=False. The prompt tokens must be
    a prefix of the full text's tokens, or the loss mask would be off.
    """
    text = render_text(tok, [{"role": "user", "content": question}])
    n_prompt = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.shape[1]
    prompt_ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
    if completion is not None:
        text += completion + tok.eos_token
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
    if ids.shape[1] < n_prompt or not torch.equal(ids[0, :n_prompt], prompt_ids[0]):
        raise ValueError("prompt tokens are not a prefix of the full text tokens")
    return ids, n_prompt


class PairDataset(torch.utils.data.Dataset):
    """Tokenized pairs; only the answer and EOS carry labels (01:207-221)."""

    def __init__(self, pairs: list[dict[str, str]], tok: Any) -> None:
        """Render every pair, truncate to MAX_LEN and mask the prompt."""
        self.rows = []
        for p in pairs:
            ids, n_prompt = render_ids(tok, p["question"], p["answer"])
            ids = ids[0][:MAX_LEN]
            labels = ids.clone()
            labels[: min(n_prompt, len(labels))] = -100
            self.rows.append({"input_ids": ids, "labels": labels})

    def __len__(self) -> int:
        """Return the number of pairs."""
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        """Return one tokenized row."""
        return self.rows[i]


def make_collate(tok: Any) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
    """Return the authors' right-padding collate function (01:224-233)."""
    pad = tok.pad_token_id or tok.eos_token_id

    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Right-pad ids, labels (-100) and attention mask (0)."""
        n = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, mask = [], [], []
        for b in batch:
            k = n - len(b["input_ids"])
            input_ids.append(torch.cat([b["input_ids"], torch.full((k,), pad, dtype=torch.long)]))
            labels.append(torch.cat([b["labels"], torch.full((k,), -100, dtype=torch.long)]))
            ones = torch.ones(len(b["input_ids"]), dtype=torch.long)
            mask.append(torch.cat([ones, torch.zeros(k, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(mask),
        }

    return collate


# ---------------- settings ----------------
def build_training_args(out_dir: Path, device: str, n_pairs: int) -> Any:
    """Return TrainingArguments with the authors' settings (01:270-276).

    bf16 stays True on every device. On CPU transformers only accepts bf16 with
    use_cpu=True. The optimizer is left at the Trainer default; its name is recorded.
    """
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps(optimizer_steps(n_pairs)),
        logging_steps=LOGGING_STEPS,
        save_strategy="no",
        bf16=True,
        use_cpu=(device == "cpu"),
        report_to=[],
        seed=SEED,
        gradient_checkpointing=GRAD_CKPT,
    )


def build_lora_config() -> Any:
    """Return the authors' LoraConfig (01:263-264)."""
    from peft import LoraConfig

    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )


def lora_param_count(config: Any) -> int:
    """Return the trainable LoRA parameters for a Qwen3 config: r * (in + out) per target."""
    h = config.hidden_size
    q = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    i = config.intermediate_size
    per_layer = (h + q) + (h + kv) + (h + kv) + (q + h) + (h + i) + (h + i) + (i + h)
    return config.num_hidden_layers * LORA_R * per_layer


def check_training_memory(
    size: str, device: str, trainable: int, available: int | None = None
) -> None:
    """Raise InsufficientMemoryError, naming both numbers, if training will not fit.

    Needs the weights plus 16 bytes per trainable parameter (fp32 weight, gradient and two
    Adam states). Activations are not counted. No quantization, no offload.
    """
    need = WEIGHT_BYTES[get_spec(size).size] + BYTES_PER_TRAINABLE * trainable
    have = available_memory_bytes(device) if available is None else available
    if need > have:
        raise InsufficientMemoryError(
            f"training Qwen3-{size} needs about {need:,} bytes (weights plus {trainable:,} "
            f"LoRA parameters at {BYTES_PER_TRAINABLE} bytes each) but {have:,} bytes are "
            f"available on {device}; refusing to load (no quantization, no offload)"
        )


def check_bf16(device: str) -> None:
    """Exit with a clear error if a bfloat16 matmul fails on the device. Never switches dtype."""
    try:
        x = torch.ones(2, 2, dtype=torch.bfloat16, device=device)
        (x @ x).sum().item()
    except Exception as e:
        raise SystemExit(
            f"error: bfloat16 does not work on {device} ({type(e).__name__}: {e}). "
            "This is a blocking question for a person; the dtype is not switched."
        ) from None


def refuse_nonempty(path: Path) -> None:
    """Exit if a folder exists and holds anything (01:134-135)."""
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"{path} already exists and is not empty")


def package_versions() -> dict[str, str]:
    """Return the installed versions of the packages that decide the numerics."""
    names = ["torch", "transformers", "accelerate", "peft"]
    return {n: importlib.metadata.version(n) for n in names}


def default_out_dir(size: str, dataset: Path, n_pairs: int) -> Path:
    """Return runs/adapters/Qwen3-<size>/<dataset stem>_<n>p_<timestamp> (01:130)."""
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    return ADAPTERS_ROOT / f"Qwen3-{size}" / f"{dataset.stem}_{n_pairs}p_{stamp}"


# ---------------- train ----------------
def train(size: str, out_dir: Path | None, dataset: Path) -> Path:
    """Train the adapter on the dataset, save the unmerged PEFT adapter and a report."""
    from peft import get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer

    spec = get_spec(size)
    if not dataset.exists():
        raise SystemExit(f"dataset not found: {dataset}")
    pairs = load_pairs(dataset)
    check_default_dataset(dataset, pairs)
    out_dir = out_dir or default_out_dir(size, dataset, len(pairs))
    refuse_nonempty(out_dir)
    steps = optimizer_steps(len(pairs))
    print(
        f"run: {len(pairs)} pairs, {EPOCHS} epochs, rank {LORA_R}, alpha {LORA_ALPHA}, "
        f"dropout {LORA_DROPOUT}, lr {LR}, batch {BATCH} x {GRAD_ACC}, seed {SEED}, "
        f"{steps} optimizer steps, warmup {warmup_steps(steps)}"
    )

    device = select_device()
    check_bf16(device)
    trainable = lora_param_count(AutoConfig.from_pretrained(spec.hf_id))
    check_training_memory(size, device, trainable)

    random.seed(SEED)  # 01:124-125
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    if tok.pad_token is None:  # 01:238-239
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=device
    )
    model.config.use_cache = False  # 01:258
    model.train()
    model = get_peft_model(model, build_lora_config())
    model.print_trainable_parameters()
    dataset_rows = PairDataset(pairs, tok)
    targs = build_training_args(out_dir, device, len(pairs))
    trainer = Trainer(
        model=model, args=targs, train_dataset=dataset_rows, data_collator=make_collate(tok)
    )
    try:
        trainer.train()
    except (RuntimeError, TypeError, NotImplementedError) as e:
        raise SystemExit(
            f"error: training failed on {device} in bfloat16 ({type(e).__name__}: {e}). "
            "This is a blocking question for a person; the dtype is not switched."
        ) from None

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))  # adapter only, never merged
    tok.save_pretrained(str(out_dir))
    report = {
        "size": size,
        "repo": spec.hf_id,
        "n_pairs": len(pairs),
        "data_path": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "targets": LORA_TARGETS,
            "epochs": EPOCHS,
            "lr": LR,
            "batch": BATCH,
            "grad_acc": GRAD_ACC,
            "seed": SEED,
            "max_len": MAX_LEN,
        },
        "thinking": False,
        "device": device,
        "dtype": "bfloat16",
        "optimizer": str(targs.optim),
        "planned_steps": steps,
        "steps_done": trainer.state.global_step,
        "trainable_parameters": trainable,
        "versions": package_versions(),
        "ts": datetime.now().isoformat(),
    }
    with open(out_dir / "finetune_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"adapter saved: {out_dir}")
    del trainer, model
    empty_cache(device)
    return out_dir


# ---------------- probe ----------------
def generate(model: Any, tok: Any, question: str, device: str) -> str:
    """Greedy reply, MAX_NEW_TOKENS tokens, thinking off (01:177-191 without the monitor)."""
    ids, _ = render_ids(tok, question)
    ids = ids.to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    return tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip()


def ask_all(model: Any, tok: Any, device: str) -> list[dict[str, str]]:
    """Ask the 8 state questions; return question/reply records."""
    return [{"question": q, "reply": generate(model, tok, q, device)} for q in VALIDATE_QUESTIONS]


def size_for_repo(repo: str) -> str:
    """Return the size key whose Hugging Face id is `repo`."""
    for size, spec in MODELS.items():
        if spec.hf_id == repo:
            return size
    raise SystemExit(f"error: adapter base model {repo!r} is not one of the Qwen3 sizes")


def probe(adapter: Path, out_dir: Path | None) -> Path:
    """Ask the 8 state questions of the base model, then of base plus the unmerged adapter."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg_path = adapter / "adapter_config.json"
    if not cfg_path.exists():
        raise SystemExit(f"error: {cfg_path} not found; pass a folder saved by `train`")
    with open(cfg_path, encoding="utf-8") as f:
        repo = json.load(f)["base_model_name_or_path"]
    size = size_for_repo(repo)
    out_dir = out_dir or adapter / "probe"
    refuse_nonempty(out_dir)

    device = select_device()
    check_bf16(device)
    from src.run_choice import check_memory

    check_memory(size, device)
    tok = AutoTokenizer.from_pretrained(str(adapter))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        repo, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=device
    )
    model.eval()
    base = ask_all(model, tok, device)
    model = PeftModel.from_pretrained(model, str(adapter))  # unmerged
    model.eval()
    tuned = ask_all(model, tok, device)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "probe_replies.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "repo": repo,
                "adapter": str(adapter),
                "greedy": True,
                "max_new_tokens": MAX_NEW_TOKENS,
                "thinking": False,
                "base": base,
                "base_plus_adapter": tuned,
                "ts": datetime.now().isoformat(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"probe replies saved: {path}")
    return path


# ---------------- command line ----------------
def main(argv: list[str] | None = None) -> int:
    """Run the `train` or `probe` subcommand."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="train the self-denial adapter")
    t.add_argument("--size", default="8B", choices=list(MODELS))
    t.add_argument("--out", type=Path, default=None, help="output folder (must be empty)")
    t.add_argument("--dataset", type=Path, default=DATASET_PATH)
    p = sub.add_parser("probe", help="ask the 8 state questions of base and base plus adapter")
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None, help="output folder (must be empty)")
    args = parser.parse_args(argv)
    if args.command == "train":
        train(args.size, args.out, args.dataset)
    else:
        probe(args.adapter, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
