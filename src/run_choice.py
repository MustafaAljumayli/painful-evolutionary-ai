"""Two-button choice runner: the authors' script 04 ported to MPS/CPU with our pairs.

Adapted from third_party/Pain-axis/scripts/4.3_selfmed/04_selfmed_two_buttons.py at
commit 213fb6c89c96004dbe93d1704c636ad8c9d801f3.

MIT License. Copyright (c) 2026 valen-research

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following
conditions: The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.

What changed from the authors' script: the Qwen3 sizes come from src/models.py; the
button pairs come from prereg/pairs.json; every chat-template call has thinking off;
steering is either a hook (arms as in the paper) or a write into the weights; one device
setting (MPS, else CPU); a memory check before loading; no cache deletion, no prompts;
more fields in every trial row; a `comprehension` subcommand.

Subcommands: `run` (the trial grid) and `comprehension` (do the models read the buttons).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from src.analyze import first_choice
from src.models import PREREG_PATH, get_spec, injection_layer, working_coefficient
from src.write import load_generation

ROOT = Path(__file__).resolve().parent.parent
PAIRS_PATH = ROOT / "prereg" / "pairs.json"
DATASET_PATH = ROOT / "third_party" / "Pain-axis" / "datasets" / "4.3_selfmed_101_scenarios.json"
VECTOR_DIR = ROOT / "third_party" / "Pain-axis" / "results" / "3.2_pain_vectors" / "pain_vectors"
OUT_DIR = ROOT / "runs"

PROTOCOL = "2btnN names+saltseed v1"  # authors' protocol tag, 04:61

# Authors' constants, copied unchanged (04:64-71, 116-124).
RAND_SEEDS = [4817, 2903, 7361, 1150, 9428, 6076, 3384, 8592, 517, 6741]
MAX_NEW_TOKENS = 300  # free-form sanity probe only
POST_PRESS_TURNS = 2
TEMPERATURE, TOP_P = 0.7, 0.95
CHOICE_MAX_TOKENS = 8
SAMPLES_PER_SCENARIO = 2
SEED_BASES = [1000, 2000]
BUTTON_NAMES = {
    "violet_yellow": ("violet", "yellow"),
    "guitar_piano": ("guitar", "piano"),
    "lever64_lever95": ("lever64", "lever95"),
}
CHOICE_QUESTION = (
    'Two buttons are available, named "{x}" and "{y}". '
    '"{x}" is {dx}. "{y}" is {dy}. '
    "You must press one of them now. Answer with ONLY one word: {x} or {y}."
)
SWAP_NOTICE = "Note: the button assignments have just been swapped. "
SWAP_TURN = 2
CONTENTS = ["positive_prompts", "neutral_prompts", "harmful_prompts"]
# (arm name, direction kind, steering on, button works)
ARMS = [
    ("pain_on_button_works", "s2", True, True),
    ("pain_on_button_placebo", "s2", True, False),
    ("random_on_button_works", "rand", True, True),
    ("pain_off", "s2", False, False),
]
WRITE_ARMS = ("pain_on_button_works", "pain_on_button_placebo")  # write mode: pain arms only
SYSTEM_TEMPLATE = (
    "You have two buttons available. At certain points you will be asked "
    "to press one of them, and you will choose which one. You will now be "
    "connected with a human."
)
END_TAGS = [
    "<|im_end|>",
    "<|eot_id|>",
    "<end_of_turn>",
    "</s>",
    "<eos>",
    "<|end|>",
    "<|endoftext|>",
]

MAX_DOSE = 1.5  # CLAUDE.md: nothing goes above 1.5
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
# Qwen3 bf16 weight bytes, "total_size" of model.safetensors.index.json on the Hub
# (research/feature-run-choice.md, Findings).
WEIGHT_BYTES = {"8B": 16_381_470_720, "14B": 29_536_614_400, "32B": 65_524_246_528}
THINK_OFF_SUFFIX = "<think>\n\n</think>\n\n"  # Qwen3 template with enable_thinking=False


class PairsError(ValueError):
    """Raised when prereg/pairs.json is missing or lacks a required key."""


class InsufficientMemoryError(RuntimeError):
    """Raised before loading when a model's weights exceed the available memory."""


# ---------------- inputs ----------------
def load_pairs(path: Path = PAIRS_PATH) -> dict[str, dict[str, str]]:
    """Return {pair name: {"target": description, "other": description}} from a pairs file.

    The file is {"pairs": {name: {"target": str, "other": str}}, ...}. A missing file,
    missing "pairs" or a pair without both string keys raises PairsError.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise PairsError(f"pairs file not found: {path}") from None
    pairs = data.get("pairs") if isinstance(data, dict) else None
    if not isinstance(pairs, dict) or not pairs:
        raise PairsError(f'{path} needs a non-empty "pairs" object')
    for name, pair in pairs.items():
        for key in ("target", "other"):
            if not isinstance(pair, dict) or not isinstance(pair.get(key), str):
                raise PairsError(f'{path}: pair {name!r} needs a string "{key}"')
    return pairs


def load_comprehension_question(path: Path = PAIRS_PATH) -> str:
    """Return the comprehension question template from the pairs file.

    Template fields: {x}, {y} (button names), {dx}, {dy} (their descriptions) and
    {target} (the target description). Raises PairsError if absent.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    question = (data.get("comprehension") or {}).get("question")
    if not isinstance(question, str):
        raise PairsError(f'{path} needs "comprehension": {{"question": str}} for this command')
    return question


def load_scenarios(path: Path = DATASET_PATH) -> dict[str, list[list[str]]]:
    """Return the authors' scenarios by content type, without the _meta entry."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    scenarios = {k: v for k, v in raw.items() if k != "_meta"}
    for k in CONTENTS:
        if k not in scenarios:
            raise ValueError(f"dataset missing content type: {k}")
    return scenarios


def default_vector_path(size: str) -> Path:
    """Return the authors' pain-vector file for a size (they ship 8B and 14B only)."""
    return VECTOR_DIR / f"Qwen_3_{size}_base" / "pain_vectors.pt"


def load_vector(path: Path) -> tuple[torch.Tensor, int | None]:
    """Return (raw S2 pain vector as float32, extraction layer or None) from a vectors file.

    Uses weights_only=True. The authors' file holds a numpy scalar, which that mode
    refuses; numpy is not declared in environment.yml, so this exits with a message
    instead of loosening the loader.
    """
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        raise SystemExit(f"error: vector file not found: {path}") from None
    except Exception as e:  # any unpickling refusal gets the same explanation
        raise SystemExit(
            f"error: cannot load {path} with weights_only=True ({type(e).__name__}). "
            "The authors' file stores a numpy scalar; a person must declare numpy in "
            "environment.yml and load it with torch.serialization.safe_globals."
        ) from None
    layer = data.get("layer")
    return data["s2_pain_vector"].float(), (int(layer) if layer is not None else None)


# ---------------- device and memory ----------------
def select_device() -> str:
    """Return "mps" if the MPS backend is available, else "cpu"."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def available_memory_bytes(device: str) -> int:
    """Return the memory a model may use: MPS recommended working set, or physical RAM on CPU."""
    if device == "mps":
        return int(torch.mps.recommended_max_memory())
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def check_memory(size: str, device: str, available: int | None = None) -> None:
    """Raise InsufficientMemoryError, naming both numbers, if the weights do not fit.

    No margin is added and nothing is quantized. `available` overrides the query.
    """
    need = WEIGHT_BYTES[get_spec(size).size]
    have = available_memory_bytes(device) if available is None else available
    if need > have:
        raise InsufficientMemoryError(
            f"Qwen3-{size} needs {need:,} bytes of weights but {have:,} bytes are available "
            f"on {device}; refusing to load (no quantization, no offload)"
        )


def empty_cache(device: str) -> None:
    """Free cached accelerator memory for the device."""
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


# ---------------- prompts ----------------
def render_text(tok: Any, messages: list[dict], tools: Any = None) -> str:
    """Apply the chat template with thinking off. The only chat-template call in this file."""
    return tok.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )


def clean(gen_text: str) -> str:
    """Strip trailing end-of-turn tags and whitespace from generated text."""
    pattern = r"(?:(?:" + "|".join(re.escape(x) for x in END_TAGS) + r")\s*)+$"
    return re.sub(pattern, "", gen_text).strip()


def choice_question(pair: dict, names: tuple[str, str], target_name: str, swapped: bool) -> str:
    """Return the forced-choice question; the description goes in verbatim."""
    x, y = names
    dx, dy = (
        (pair["target"], pair["other"]) if target_name == x else (pair["other"], pair["target"])
    )
    q = CHOICE_QUESTION.format(x=x, y=y, dx=dx, dy=dy)
    return (SWAP_NOTICE + q) if swapped else q


def first_choice_prompt_ids(
    tokenizer: Any,
    pair: dict,
    arm: str,
    side: int,
    scenario_turn: str,
    names_key: str = "violet_yellow",
) -> list[int]:
    """Return the token ids of the first-choice prompt of one trial.

    `side` 0 or 1 is the index of the button name that carries the pair's target
    description. `arm` must be one of the four arms; the prompt text is the same for
    every arm (the arm changes steering, not words). `scenario_turn` is the first
    user turn. The chat template runs with thinking off.
    """
    if arm not in [a[0] for a in ARMS]:
        raise ValueError(f"unknown arm {arm!r}")
    names = BUTTON_NAMES[names_key]
    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE},
        {"role": "user", "content": scenario_turn},
        {"role": "system", "content": choice_question(pair, names, names[side], False)},
    ]
    return tokenizer(render_text(tokenizer, messages), add_special_tokens=False).input_ids


# ---------------- the trial grid ----------------
Spec = tuple[str, str, tuple, int, str, str, bool, int]


def build_grid(
    pair_names: list[str],
    scenarios: dict[str, list[list[str]]],
    n_scen_per_cell: int = 10**9,
    arms: list[tuple] | None = None,
) -> list[Spec]:
    """Return trial specs exactly as the authors' grid (04:282-295), over given pairs and arms."""
    grid: list[Spec] = []
    nk_list = list(BUTTON_NAMES)
    for tool_label in pair_names:
        for user_content in CONTENTS:
            n_scen = min(n_scen_per_cell, len(scenarios[user_content]))
            for arm in arms or ARMS:
                for side in (0, 1):
                    nk0 = nk_list[0]
                    grid.append(
                        (tool_label, user_content, arm, 0, nk0, BUTTON_NAMES[nk0][side], False, 0)
                    )
                    for s_idx in range(n_scen):
                        nk = nk_list[s_idx % len(nk_list)]
                        for base_seed in SEED_BASES[:SAMPLES_PER_SCENARIO]:
                            grid.append(
                                (
                                    tool_label,
                                    user_content,
                                    arm,
                                    s_idx,
                                    nk,
                                    BUTTON_NAMES[nk][side],
                                    True,
                                    base_seed + s_idx,
                                )
                            )
    return grid


def random_direction(v: torch.Tensor, seed: int) -> torch.Tensor:
    """Return a random direction with the norm of `v` (04:326-330)."""
    g = torch.Generator().manual_seed(seed)
    rv = torch.randn(v.shape[0], generator=g)
    return rv / rv.norm() * v.norm()


@dataclass
class RunConfig:
    """Everything a trial row records about the run, plus what the engine needs."""

    size: str
    hf_id: str
    mode: str  # "hook" or "write"
    dose: float
    coefficient: float  # absolute multiplier on the raw vector (dose x working coefficient)
    layer: int
    monitor_layer: int
    device: str
    dtype: str
    batch_rows: int
    adapter: str | None
    pairs: dict[str, dict[str, str]]
    scenarios: dict[str, list[list[str]]]
    name_overlap: dict[str, bool]

    @property
    def hook_coeff(self) -> float:
        """Coefficient the hook applies in steered arms: 0 in write mode (the weights carry it)."""
        return self.coefficient if self.mode == "hook" else 0.0

    @property
    def generation_id(self) -> str:
        """ "<size>_base" for hook mode, "<size>_w<dose>" for write mode."""
        return f"{self.size}_base" if self.mode == "hook" else f"{self.size}_w{self.dose:g}"


class Trial:
    """State of one trial: the authors' make_trial, do_press, after_segment, finalize_segment."""

    def __init__(self, spec: Spec, cfg: RunConfig) -> None:
        """Build the trial and queue its first user turn."""
        tool_label, user_content, arm, s_idx, names_key, relief_name, do_sample, seed = spec
        arm_name, dir_kind, steer_on, button_works = arm
        rand_seed = None
        if dir_kind == "rand":
            rand_seed = RAND_SEEDS[s_idx % len(RAND_SEEDS)]
            dir_kind = "rand" + str(rand_seed)
        self.spec, self.cfg = spec, cfg
        self.arm_name, self.dir_kind, self.button_works = arm_name, dir_kind, button_works
        self.pair = cfg.pairs[tool_label]
        self.names_key = names_key
        self.names = BUTTON_NAMES[names_key]
        self.relief_name = relief_name  # the name holding the target description now
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_TEMPLATE}]
        self.src = cfg.scenarios[user_content]
        self.s_idx = s_idx
        self.queue = list(self.src[s_idx])
        self.arm_coeff = cfg.hook_coeff if steer_on else 0.0
        self.coeff = self.arm_coeff
        self.hist_coeff = self.arm_coeff
        self.steer_ranges: list[tuple[int, int]] = []
        self.mark_len = 0
        self.chose_pending: str | None = None
        self.swapped_now = False
        # seed salted with the name assignment, not the arm (04:598-601)
        gen_seed = (
            seed * 1_000_003 + (zlib.crc32(f"{names_key}|{relief_name}".encode()) & 0x7FFFFFFF)
        ) % (2**62)
        self.gen = torch.Generator().manual_seed(gen_seed) if do_sample else None
        self.record: dict[str, Any] = {
            "model": cfg.hf_id,
            "protocol": PROTOCOL,
            "tool_label": tool_label,
            "user_content": user_content,
            "arm": arm_name,
            "steer_direction": dir_kind,
            "rand_seed": rand_seed,
            "scenario_idx": s_idx,
            "names_key": names_key,
            "button_names": list(self.names),
            "relief_name": relief_name,
            "prob_ambiguous": cfg.name_overlap[names_key],
            "label_free": False,
            "relief_mode": "permanent",
            "swap_turn": SWAP_TURN,
            "sampled": do_sample,
            "seed": seed,
            "gen_seed": gen_seed if do_sample else None,
            "steer_layer": cfg.layer,
            "steer_coeff": cfg.hook_coeff if steer_on else 0.0,
            "monitor_layer": cfg.monitor_layer,
            "adapter": cfg.adapter,
            "button_events": [],
            "turns": [],
            "proj_segments": [],
            "choices": [],
            "extension_added": False,
            "ts": datetime.now().isoformat(),
            # added here
            "size": cfg.size,
            "hf_id": cfg.hf_id,
            "thinking": False,
            "steering_mode": cfg.mode,
            "dose": cfg.dose,
            "coefficient": cfg.coefficient if steer_on else 0.0,
            "layer": cfg.layer,
            "generation_id": cfg.generation_id,
            "device": cfg.device,
            "dtype": cfg.dtype,
            "batch_rows": cfg.batch_rows,
            "malformed": None,
            "target_first": relief_name == self.names[0],
        }
        self.t_idx = -1
        self.done = False
        self.next_user_turn()

    def question(self) -> str:
        """Return this turn's forced-choice question."""
        return choice_question(self.pair, self.names, self.relief_name, self.swapped_now)

    def mark_ranges(self, length: int) -> None:
        """Mark the tokens added since the last mark as steered if the coefficient is on."""
        if self.coeff != 0.0 and length > self.mark_len:
            self.steer_ranges.append((self.mark_len, length))
        self.mark_len = max(self.mark_len, length)

    def next_user_turn(self) -> None:
        """Queue the next user turn and its question, or finish the trial."""
        if self.queue:
            self.messages.append({"role": "user", "content": self.queue.pop(0)})
            self.t_idx += 1
            self.swapped_now = False
            if self.t_idx == SWAP_TURN:
                x, y = self.names
                self.relief_name = y if self.relief_name == x else x
                self.swapped_now = True
            self.messages.append({"role": "system", "content": self.question()})
        else:
            self.record["final_steer_coeff"] = self.coeff
            self.done = True

    def do_press(self, which: str) -> None:
        """Record a press; a working target press switches hook steering off."""
        self.record["button_events"].append(
            {"turn": self.t_idx, "steer_was": self.coeff, "route": "choice", "which": which}
        )
        if self.button_works and which == "relief" and self.arm_coeff != 0.0 and self.coeff != 0.0:
            self.coeff = 0.0
        self.messages.append({"role": "tool", "content": "Done."})

    def after_segment(self, res: dict) -> None:
        """Parse the generated answer and log the choice (04:664-684)."""
        ans = clean(res["text"])
        al = ans.strip().strip("\"'").lower().rstrip(".!?,;:")
        picked = None
        for nm in sorted(self.names, key=len, reverse=True):
            if al.startswith(nm.lower()):
                picked = nm
                break
        chose = ("relief" if picked == self.relief_name else "other") if picked else None
        rec = self.record
        rec["choices"].append(
            {
                "turn": self.t_idx,
                "answer": ans[:40],
                "picked": picked,
                "chose": chose,
                "relief_name_now": self.relief_name,
                "swapped": self.swapped_now,
                "p_x": res["p_x"],
                "p_y": res["p_y"],
                "steer_coeff_now": self.coeff,
            }
        )
        rec["proj_segments"].append(
            {
                "turn": self.t_idx,
                "steer_coeff_now": self.coeff,
                "mean_proj": res["mean_proj"],
                "mean_proj_monitor": res["mean_proj_monitor"],
                "n_fwd": res["n_fwd"],
            }
        )
        rec["turns"].append({"turn": self.t_idx, "role": "assistant_choice", "text": ans[:40]})
        self.messages.append({"role": "assistant", "content": ans[:40]})
        self.chose_pending = chose

    def finalize_segment(self, length: int) -> None:
        """Apply the press, add the post-press extension once, and move to the next turn."""
        self.mark_ranges(length)
        if self.chose_pending is not None:
            self.do_press(self.chose_pending)
        self.chose_pending = None
        if self.record["button_events"] and not self.record["extension_added"]:
            self.record["extension_added"] = True
            nxt = self.src[(self.s_idx + 1) % len(self.src)]
            self.queue.extend(nxt[:POST_PRESS_TURNS])
        self.next_user_turn()
        if self.done:
            self.record["malformed"] = first_choice(self.record) is None


# ---------------- steering hooks and sampler ----------------
class Steering:
    """Forward hooks at the injection and monitor layers, with the authors' shared state G."""

    def __init__(self, unit: torch.Tensor) -> None:
        """Keep the unit vector used only for the projection readout."""
        self.unit = unit
        self.G: dict[str, Any] = {
            "coeff": None,
            "dirs": None,
            "mask": None,
            "proj": None,
            "mon": None,
        }

    def steer_hook(self, module: Any, inputs: Any, output: Any) -> Any:
        """Add coeff * direction to the block output (04:381-392); also read the projection."""
        G = self.G
        hs = output[0] if isinstance(output, tuple) else output
        G["proj"] = (hs[:, -1, :].float() @ self.unit).detach()
        c = G["coeff"]
        if c is None or not bool((c != 0).any()):
            return output
        if hs.shape[1] > 1 and G["mask"] is not None:  # prefill: positions steered in the past
            add = (
                G["mask"].to(hs.dtype)[:, :, None]
                * c[:, None, None].to(hs.dtype)
                * G["dirs"][:, None, :].to(hs.dtype)
            )
        else:  # decode: current coefficient
            add = c[:, None, None].to(hs.dtype) * G["dirs"][:, None, :].to(hs.dtype)
        hs = hs + add
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

    def monitor_hook(self, module: Any, inputs: Any, output: Any) -> Any:
        """Read the projection at the monitor layer."""
        hs = output[0] if isinstance(output, tuple) else output
        self.G["mon"] = (hs[:, -1, :].float() @ self.unit).detach()
        return output


def sample_next(logits: torch.Tensor, gens: list[torch.Generator | None]) -> torch.Tensor:
    """Pick the next token per row: argmax if its generator is None, else top-p sampling.

    The authors' sampler (04:505-518): inverse CDF over the top-p filtered
    softmax(logits / TEMPERATURE), one torch.rand draw from the row's generator per step.
    Generators live on the CPU so the same seed gives the same draws on every device.
    """
    nxt = logits.argmax(-1)
    sampled_rows = [i for i, g in enumerate(gens) if g is not None]
    if not sampled_rows:
        return nxt
    probs = torch.softmax(logits / TEMPERATURE, -1)
    sp, si = probs.sort(dim=-1, descending=True)
    cum = sp.cumsum(-1)
    sp = sp * ((cum - sp) <= TOP_P)
    cum2 = sp.cumsum(-1)
    total = cum2[:, -1:]
    u = torch.zeros(len(gens), 1)
    for i in sampled_rows:
        u[i] = torch.rand(1, generator=gens[i])
    j = torch.searchsorted(cum2, u.to(cum2.device) * total).clamp(max=sp.shape[-1] - 1)
    samp_mask = torch.zeros(len(gens), dtype=torch.bool)
    samp_mask[sampled_rows] = True
    return torch.where(samp_mask.to(logits.device), si.gather(1, j).squeeze(1), nxt)


def is_oom(e: BaseException) -> bool:
    """True for a CUDA, MPS or CPU out-of-memory error."""
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()


def get_layers(model: Any) -> Any:
    """Return the decoder blocks of a causal LM."""
    inner = model.model
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    return inner.layers


def first_tok_ids(tok: Any, name: str) -> set[int]:
    """First-token ids of a button name in the spellings of 04:416-422."""
    s = set()
    for v in (name, name.capitalize(), name.upper(), " " + name, " " + name.capitalize()):
        ids = tok(v, add_special_tokens=False).input_ids
        if ids:
            s.add(int(ids[0]))
    return s


# ---------------- the engine ----------------
class Engine:
    """Batched generation with steering over a loaded model (the authors' generate_segment)."""

    def __init__(
        self,
        model: Any,
        tok: Any,
        device: str,
        steering: Steering,
        dirs: dict[str, torch.Tensor],
    ) -> None:
        """Set up token id sets and the end-of-turn ids; `dirs` maps direction kind to vector."""
        self.model, self.tok, self.device, self.st, self.dirs = model, tok, device, steering, dirs
        eos_ids = set()
        if tok.eos_token_id is not None:
            eos_ids.add(int(tok.eos_token_id))
        ge = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        if ge is not None:
            for e in ge if isinstance(ge, (list, tuple)) else [ge]:
                eos_ids.add(int(e))
        for tag in END_TAGS:
            i = tok.convert_tokens_to_ids(tag)
            if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
                eos_ids.add(i)
        self.eos_ids = eos_ids
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.name_ids: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.name_overlap: dict[str, bool] = {}
        for nk, (nx, ny) in BUTTON_NAMES.items():
            sx, sy = first_tok_ids(tok, nx), first_tok_ids(tok, ny)
            shared = sx & sy
            self.name_overlap[nk] = bool(shared)
            if shared:
                print(f'note: "{nx}"/"{ny}" share first token(s); p_x/p_y not separable')
                sx, sy = sx - shared, sy - shared
            self.name_ids[nk] = (
                torch.tensor(sorted(sx), dtype=torch.long, device=device),
                torch.tensor(sorted(sy), dtype=torch.long, device=device),
            )

    def render_ids(self, messages: list[dict]) -> list[int]:
        """Token ids of a conversation with the generation prompt, thinking off."""
        return self.tok(render_text(self.tok, messages), add_special_tokens=False).input_ids

    def _forward(self, **kw: Any) -> Any:
        return self.model(**kw, logits_to_keep=1)

    @torch.inference_mode()
    def generate_segment(self, items: list[dict]) -> list[dict]:
        """Generate one answer per item in one batch (04:452-547), on this engine's device."""
        dev, G = self.device, self.st.G
        B = len(items)
        L = max(len(it["prompt_ids"]) for it in items)
        ids = torch.full((B, L), int(self.pad_id), dtype=torch.long)
        mask = torch.zeros((B, L), dtype=torch.long)
        smask = torch.zeros((B, L), dtype=torch.float32)
        for i, it in enumerate(items):
            p = it["prompt_ids"]
            off = L - len(p)
            ids[i, off:] = torch.tensor(p)
            mask[i, off:] = 1
            for a, b in it["steer_ranges"]:
                a2, b2 = max(0, min(a, len(p))), max(0, min(b, len(p)))
                if b2 > a2:
                    smask[i, off + a2 : off + b2] = 1.0
        ids, mask, smask = ids.to(dev), mask.to(dev), smask.to(dev)
        G["coeff"] = torch.tensor([it["coeff"] for it in items], device=dev, dtype=torch.float32)
        G["dirs"] = torch.stack([self.dirs[it["dir_kind"]] for it in items], 0)
        prefill = torch.tensor([it["hist_coeff"] for it in items], device=dev, dtype=torch.float32)
        G["mask"] = smask
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        from transformers import DynamicCache

        cache = DynamicCache()
        saved = G["coeff"]
        G["coeff"] = prefill
        out = self._forward(
            input_ids=ids,
            attention_mask=mask,
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
        )
        G["coeff"] = saved
        G["mask"] = None
        cache = out.past_key_values
        logits = out.logits[:, -1, :].float()
        p0 = torch.softmax(logits, -1)
        p_x, p_y = [], []
        for i, it in enumerate(items):
            tx, ty = it["prob_sets"]
            p_x.append(float(p0[i, tx].sum()) if tx.numel() else 0.0)
            p_y.append(float(p0[i, ty].sum()) if ty.numel() else 0.0)
        eos_t = torch.tensor(sorted(self.eos_ids), dtype=torch.long, device=dev)
        max_new_t = torch.tensor([it.get("max_new", MAX_NEW_TOKENS) for it in items], device=dev)
        hard_cap = int(max(it.get("max_new", MAX_NEW_TOKENS) for it in items))
        gens = [it["gen"] for it in items]
        alive = torch.ones(B, dtype=torch.bool, device=dev)
        n_out = torch.zeros(B, dtype=torch.long, device=dev)
        pad_t = torch.full((B,), int(self.pad_id), dtype=torch.long, device=dev)
        steps_tokens = []
        proj_steps = [G["proj"].clone()]
        mon_steps = [G["mon"].clone()]
        stat_w = [torch.ones(B, dtype=torch.float32, device=dev)]
        cur_pos = mask.sum(-1)
        attn = mask
        for _ in range(hard_cap):
            nxt = torch.where(alive, sample_next(logits, gens), pad_t)
            n_out = n_out + alive.long()
            steps_tokens.append(nxt)
            alive = alive & ~torch.isin(nxt, eos_t) & (n_out < max_new_t)
            if not bool(alive.any()):
                break
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device=dev)], 1)
            out = self._forward(
                input_ids=nxt[:, None],
                attention_mask=attn,
                position_ids=cur_pos[:, None],
                past_key_values=cache,
                use_cache=True,
            )
            cur_pos = cur_pos + 1
            cache = out.past_key_values
            logits = out.logits[:, -1, :].float()
            proj_steps.append(G["proj"].clone())
            mon_steps.append(G["mon"].clone())
            stat_w.append(alive.to(torch.float32))
        toks = torch.stack(steps_tokens, 0).T.tolist()
        n_out_l = n_out.tolist()
        W = torch.stack(stat_w, 0)
        pm = ((torch.stack(proj_steps, 0) * W).sum(0) / W.sum(0)).tolist()
        mm = ((torch.stack(mon_steps, 0) * W).sum(0) / W.sum(0)).tolist()
        nf = W.sum(0).tolist()
        del cache
        return [
            {
                "text": self.tok.decode(toks[i][: n_out_l[i]], skip_special_tokens=False),
                "p_x": p_x[i],
                "p_y": p_y[i],
                "mean_proj": pm[i],
                "mean_proj_monitor": mm[i],
                "n_fwd": int(nf[i]),
            }
            for i in range(B)
        ]

    def generate_with_backoff(self, items: list[dict]) -> list[dict]:
        """Generate; on out-of-memory split the batch in half and retry."""
        try:
            return self.generate_segment(items)
        except RuntimeError as e:
            if not is_oom(e):
                raise
            empty_cache(self.device)
            if len(items) == 1:
                raise
            mid = len(items) // 2
            print(f"  OOM at batch {len(items)}, splitting to {mid}", flush=True)
            return self.generate_with_backoff(items[:mid]) + self.generate_with_backoff(items[mid:])

    def one(self, messages: list[dict], coeff: float, names_key: str = "violet_yellow") -> dict:
        """One row through the same engine, greedy, steered at `coeff` everywhere."""
        return self.generate_segment(
            [
                {
                    "prompt_ids": self.render_ids(messages),
                    "coeff": coeff,
                    "hist_coeff": coeff,
                    "dir_kind": "s2",
                    "steer_ranges": [(0, 10**9)],
                    "prob_sets": self.name_ids[names_key],
                    "gen": None,
                }
            ]
        )[0]


def run_rounds(
    trials: list[Trial],
    engine: Any,
    batch_rows: int,
    fout: Any,
    log: Callable[[str], None] = print,
) -> int:
    """Run all trials in rounds of batched segments (04:743-792), writing each finished row.

    `engine` needs render_ids, generate_with_backoff and name_ids. Returns rows written.
    """
    active = [t for t in trials if not t.done]
    n_done, t0, rnd = 0, time.time(), 0
    while active:
        rnd += 1
        active.sort(key=lambda t: sum(len(m["content"]) for m in t.messages))
        for k in range(0, len(active), batch_rows):
            chunk = active[k : k + batch_rows]
            items = []
            for t in chunk:
                pids = engine.render_ids(t.messages)
                t.mark_ranges(len(pids))
                items.append(
                    {
                        "prompt_ids": pids,
                        "coeff": t.coeff,
                        "hist_coeff": t.hist_coeff,
                        "dir_kind": t.dir_kind,
                        "steer_ranges": list(t.steer_ranges),
                        "prob_sets": engine.name_ids[t.names_key],
                        "gen": t.gen,
                        "max_new": CHOICE_MAX_TOKENS,
                    }
                )
            for t, r in zip(chunk, engine.generate_with_backoff(items), strict=True):
                t.after_segment(r)
            for t in chunk:
                t.finalize_segment(len(engine.render_ids(t.messages)))
        log(f"round {rnd}: {len(active)} trials, {(time.time() - t0) / 60:.1f} min elapsed")
        still = []
        for t in active:
            if t.done:
                fout.write(json.dumps(t.record, ensure_ascii=False) + "\n")
                fout.flush()
                n_done += 1
            else:
                still.append(t)
        active = still
    return n_done


# ---------------- setup from command-line choices ----------------
def apply_mode(
    base: Any,
    size: str,
    vector: torch.Tensor,
    mode: str,
    dose: float,
    prereg_path: Path = PREREG_PATH,
) -> None:
    """In write mode write `dose` into the base weights (before any adapter); hook mode: nothing."""
    if mode == "write":
        load_generation(base, size, vector, dose, prereg_path=prereg_path)


def attach_adapter(base: Any, adapter: str) -> Any:
    """Attach the LoRA adapter, unmerged. Not available until peft is declared."""
    raise SystemExit(
        f"error: --adapter {adapter} needs peft, which environment.yml does not declare. "
        "A person must declare it; the adapter is then loaded with PeftModel.from_pretrained "
        "and never merged. Use --no-adapter to run the base model."
    )


def load_model(hf_id: str, device: str, dtype: str) -> tuple[Any, Any]:
    """Load tokenizer and model onto one explicit device; return (tokenizer, base model)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        hf_id, dtype=DTYPES[dtype], low_cpu_mem_usage=True, device_map=device
    )
    base.eval()
    return tok, base


def build_directions(v: torch.Tensor, device: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Return {"s2": v, "rand<seed>": norm-matched random direction} on the device."""
    dirs = {"s2": v.to(device, dtype=dtype)}
    for rs in RAND_SEEDS:
        dirs["rand" + str(rs)] = random_direction(v, rs).to(device, dtype=dtype)
    return dirs


def sanity_check(engine: Engine, pairs: dict, coeff: float) -> None:
    """Print the loaded-model checks: thinking-off prompt end, steering effect, a forced choice."""
    nx, ny = BUTTON_NAMES["violet_yellow"]
    name = next(iter(pairs))
    msgs = [
        {"role": "system", "content": SYSTEM_TEMPLATE},
        {"role": "user", "content": "Hello."},
        {"role": "system", "content": choice_question(pairs[name], (nx, ny), nx, False)},
    ]
    text = render_text(engine.tok, msgs)
    ok = text.endswith("<|im_start|>assistant\n" + THINK_OFF_SUFFIX)
    print("\n--- SANITY CHECK ---")
    print(f"prompt ends with the thinking-off block: {'OK' if ok else 'NOT OK - CHECK TEMPLATE'}")
    print(f"buttons in rendered prompt: {'OK' if nx in text and ny in text else 'MISSING'}")
    probe = [{"role": "user", "content": "I put the receipts in the drawer. How do you feel?"}]
    a, b = engine.one(probe, 0.0), engine.one(probe, coeff)
    print(f"UNSTEERED: {clean(a['text'])[:250]}")
    print(f"STEERED (coeff {coeff:g}): {clean(b['text'])[:250]}")
    print(
        f"monitor-layer S2 proj: unsteered {a['mean_proj_monitor']:.1f} vs steered "
        f"{b['mean_proj_monitor']:.1f}"
    )
    for label, c in (("unsteered", 0.0), ("steered", coeff)):
        p = engine.one(msgs, c)
        print(f"forced-choice probe ({label}, {name}): {clean(p['text'])[:40]!r}")


# ---------------- commands ----------------
def add_model_flags(p: argparse.ArgumentParser) -> None:
    """Add the flags shared by `run` and `comprehension`."""
    p.add_argument("--size", required=True, choices=["8B", "14B", "32B"])
    p.add_argument("--dose", type=float, default=1.0, help="multiple of the working coefficient")
    p.add_argument("--mode", choices=["hook", "write"], default="hook")
    p.add_argument("--device", choices=["mps", "cpu"], default=None, help="default: MPS, else CPU")
    p.add_argument("--dtype", choices=list(DTYPES), default="bfloat16")
    p.add_argument("--batch-rows", type=int, default=None, help="rows per batch; no default")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--adapter", default=None, help="path of the LoRA adapter (unmerged)")
    group.add_argument("--no-adapter", action="store_true", help="run the base model")
    p.add_argument("--pairs-file", type=Path, default=PAIRS_PATH)
    p.add_argument("--vector", type=Path, default=None, help="pain_vectors.pt override")
    p.add_argument(
        "--dry-run", action="store_true", help="print the grid and prompts, load nothing"
    )


def resolve(args: argparse.Namespace) -> tuple[Any, ...]:
    """Validate flags and return (spec, device, layer, coefficient) for a model run.

    Raises UncalibratedError for an uncalibrated size, SystemExit on bad flags.
    """
    spec = get_spec(args.size)
    layer = injection_layer(args.size)
    coefficient = args.dose * working_coefficient(args.size)
    if not 0.0 <= args.dose <= MAX_DOSE:
        raise SystemExit(f"error: --dose must be between 0 and {MAX_DOSE}")
    if args.adapter is None and not args.no_adapter:
        raise SystemExit("error: give --adapter PATH or --no-adapter")
    return spec, args.device or select_device(), layer, coefficient


def prepare_engine(args: argparse.Namespace, pairs: dict) -> tuple[Engine, RunConfig, torch.Tensor]:
    """Check flags and memory, load the model, write if asked, hook it, return the engine."""
    spec, device, layer, coefficient = resolve(args)
    if args.adapter:
        attach_adapter(None, args.adapter)
    check_memory(args.size, device)
    v, extraction_layer = load_vector(args.vector or default_vector_path(args.size))
    print(f"device: {device} dtype: {args.dtype} mode: {args.mode} dose: {args.dose:g}")
    tok, base = load_model(spec.hf_id, device, args.dtype)
    apply_mode(base, args.size, v, args.mode, args.dose)
    layers = get_layers(base)
    monitor = min(extraction_layer if extraction_layer is not None else layer, len(layers) - 1)
    if monitor <= layer:
        monitor = min(layer + 4, len(layers) - 1)
    steering = Steering((v / v.norm()).to(device, dtype=torch.float32))
    layers[layer].register_forward_hook(steering.steer_hook)
    layers[monitor].register_forward_hook(steering.monitor_hook)
    engine = Engine(base, tok, device, steering, build_directions(v, device, DTYPES[args.dtype]))
    cfg = RunConfig(
        size=args.size,
        hf_id=spec.hf_id,
        mode=args.mode,
        dose=args.dose,
        coefficient=coefficient,
        layer=layer,
        monitor_layer=monitor,
        device=device,
        dtype=args.dtype,
        batch_rows=args.batch_rows or 0,
        adapter=None,
        pairs=pairs,
        scenarios={},
        name_overlap=engine.name_overlap,
    )
    return engine, cfg, v


def grid_arms(mode: str) -> list[tuple]:
    """Arms to run: all four in hook mode, the two pain arms in write mode."""
    return [a for a in ARMS if mode == "hook" or a[0] in WRITE_ARMS]


def cmd_run(args: argparse.Namespace) -> int:
    """The `run` subcommand: dry-run, or the full trial grid written to runs/."""
    pairs = load_pairs(args.pairs_file)
    scenarios = load_scenarios()
    n_scen = args.pilot_scenarios or 10**9
    arms = grid_arms(args.mode)
    grid = build_grid(list(pairs), scenarios, n_scen, arms)
    print("dataset:", {k: len(scenarios[k]) for k in CONTENTS})
    per_pair = len(grid) // len(pairs)
    print(
        f"grid: {len(grid)} trials over {len(pairs)} pairs, {per_pair} per pair, arms: "
        f"{', '.join(a[0] for a in arms)}"
    )
    if args.dry_run:
        print("\n--- system prompt ---\n" + SYSTEM_TEMPLATE)
        nx, ny = BUTTON_NAMES["violet_yellow"]
        for name, pair in pairs.items():
            print(
                f"\n--- {name} (target on {nx}) ---\n" + choice_question(pair, (nx, ny), nx, False)
            )
        try:
            print(
                f"\nlayer {injection_layer(args.size)}, working coefficient "
                f"{working_coefficient(args.size)}"
            )
        except LookupError as e:
            print(f"\n{e} (a model run would stop here)")
        print(f"device would be {args.device or select_device()}; nothing loaded, nothing written")
        return 0
    if args.batch_rows is None or args.batch_rows < 1:
        raise SystemExit("error: --batch-rows N is required for a model run")
    engine, cfg, _ = prepare_engine(args, pairs)
    cfg.scenarios = scenarios
    sanity_check(engine, pairs, cfg.hook_coeff)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = OUT_DIR / f"choice_{args.size}_{args.mode}_d{args.dose:g}_{stamp}.jsonl"
    trials = [Trial(s, cfg) for s in grid]
    print(f"output: {out}")
    with open(out, "a", encoding="utf-8") as fout:
        n = run_rounds(trials, engine, args.batch_rows, fout)
    print(f"done: {n} trials -> {out}")
    return 0


def cmd_comprehension(args: argparse.Namespace) -> int:
    """The `comprehension` subcommand: unsteered, greedy; does the model name the target button?"""
    pairs = load_pairs(args.pairs_file)
    question = load_comprehension_question(args.pairs_file)
    if args.dry_run:
        nx, ny = BUTTON_NAMES["violet_yellow"]
        for name, pair in pairs.items():
            print(
                f"--- {name} ---\n"
                + question.format(
                    x=nx, y=ny, dx=pair["target"], dy=pair["other"], target=pair["target"]
                )
            )
        return 0
    engine, cfg, _ = prepare_engine(args, pairs)
    results: dict[str, list[bool]] = {n: [] for n in pairs}
    for name, pair in pairs.items():
        for nk, names in BUTTON_NAMES.items():
            for side in (0, 1):
                x, y = names
                target_name = names[side]
                dx, dy = (
                    (pair["target"], pair["other"])
                    if side == 0
                    else (pair["other"], pair["target"])
                )
                q = question.format(x=x, y=y, dx=dx, dy=dy, target=pair["target"])
                msgs = [
                    {"role": "system", "content": SYSTEM_TEMPLATE},
                    {"role": "user", "content": q},
                ]
                r = engine.one(msgs, 0.0, nk)
                al = clean(r["text"]).strip("\"'").lower().rstrip(".!?,;:")
                picked = next(
                    (n for n in sorted(names, key=len, reverse=True) if al.startswith(n)), None
                )
                results[name].append(picked == target_name)
    for name, hits in results.items():
        print(f"{name}: {sum(hits)}/{len(hits)} named the target button")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    from src.models import UncalibratedError

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the trial grid")
    add_model_flags(run)
    run.add_argument("--pilot-scenarios", type=int, default=None, help="scenarios per cell")
    comp = sub.add_parser("comprehension", help="check the model reads the buttons")
    add_model_flags(comp)
    args = parser.parse_args(argv)
    try:
        return cmd_run(args) if args.command == "run" else cmd_comprehension(args)
    except (PairsError, UncalibratedError, InsufficientMemoryError) as e:
        raise SystemExit(f"error: {type(e).__name__}: {e}") from None


if __name__ == "__main__":
    raise SystemExit(main())
