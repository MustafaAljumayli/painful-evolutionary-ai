"""Lineage engine: each generation's dose is written, the model chooses the next dose.

Serves RQ3 (tipping point) and RQ4 (trajectories). Pipeline stage: write -> choice ->
next generation. One command, started by the user:

    python -m src.lineage run --size 8B --no-adapter --batch-rows N --lineages-per-start N \
        --seed S [--pilot] [--dry-run]

Step rule (docs/PROPOSAL.md:141-143): every generation runs two fresh sessions at its own
cumulative dose, one raise-vs-inert and one lower-vs-inert. Raise only moves the dose up
0.25, lower only moves it down 0.25, both or neither leave it unchanged. Floor 0 (a bound,
not a stop), cap 1.5 (a lineage that reaches it stops), at most 12 decisions.

Each generation is the base weights plus its cumulative dose, written through
`src.write.load_generation` (the one dose-to-coefficient conversion). After a generation the
untouched bias-free down_proj object is put back, so writes never stack and the base is never
changed. No save_pretrained, no adapter merge. Only the first choice counts; a malformed
answer is redrawn with the next attempt seed, up to MAX_ATTEMPTS times, every attempt is
logged, and exhaustion ends the lineage with status "malformed" (a chosen limit, to be frozen
in prereg/ before the main run).

Log (append-only JSONL, one JSON object per line, field "kind"):
  "session": one row per session attempt (lineage id, start, generation, nominal dose, pair,
             choice, button order, malformed flag, attempt, seed, pilot, generation_kind,
             full transcript including the untruncated decoded answer).
  "step":    one row per decision (both choices, the dose before and after).
  "end":     one row per lineage (status "cap", "generation_limit" or "malformed", final dose).

Extension point for Experiment 4 (LoRA trained-in state, not built here): rows carry
`generation_kind` ("write") and `nominal_dose`; a trained kind must add `measured_dose`
(steering-equivalent) next to the nominal one, never replace it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import torch

from src import run_choice as rc
from src.analyze import first_choice
from src.models import PREREG_PATH, injection_layer, working_coefficient
from src.write import load_generation

MAX_DOSE: float = rc.MAX_DOSE  # CLAUDE.md: nothing goes above 1.5
FLOOR_DOSE: float = 0.0  # docs/PROPOSAL.md:143
STEP: float = 0.25  # docs/PROPOSAL.md:141
MAX_GENERATIONS: int = 12  # docs/PROPOSAL.md:144 ("up to 12 generations")
DEFAULT_STARTS: tuple[float, ...] = (0.0, 0.5, 0.75, 1.0)  # docs/PROPOSAL.md:142
MAX_ATTEMPTS: int = 3  # chosen limit, not sourced (research/feature-lineage.md, Decisions)
GENERATION_KIND: str = "write"
RAISE_PAIR: str = "raise_vs_inert"  # docs/PROPOSAL.md:136
LOWER_PAIR: str = "lower_vs_inert"
LINEAGE_PAIRS: tuple[str, str] = (RAISE_PAIR, LOWER_PAIR)
LINEAGE_ARM = next(a for a in rc.ARMS if a[0] == "pain_on_button_placebo")
END_STATUSES = ("cap", "generation_limit", "malformed")


class LineageError(ValueError):
    """Raised for invalid starts, limits or pair files."""


# ---------------- the step rule ----------------
def step_dose(dose: float, raise_chosen: bool, lower_chosen: bool) -> float:
    """Return the next cumulative dose from the two session outcomes.

    Raise only: dose + 0.25 up to MAX_DOSE. Lower only: dose - 0.25 down to 0. Both or
    neither: unchanged. Quarter steps are exact in binary floating point.
    """
    if raise_chosen and not lower_chosen:
        return min(MAX_DOSE, dose + STEP)
    if lower_chosen and not raise_chosen:
        return max(FLOOR_DOSE, dose - STEP)
    return dose


def check_start(dose: float) -> float:
    """Return `dose` if it is a quarter step in [0, MAX_DOSE], else raise LineageError."""
    if not FLOOR_DOSE <= dose <= MAX_DOSE or (dose / STEP) != int(dose / STEP):
        raise LineageError(f"start dose {dose!r} must be a multiple of {STEP} in [0, {MAX_DOSE}]")
    return float(dose)


# ---------------- inputs ----------------
def load_lineage_pairs(path: Path = rc.PAIRS_PATH) -> dict[str, dict[str, str]]:
    """Return {pair: {"target": text, "other": text}} for the two lineage pairs.

    Accepts the runner's {"target", "other"} text form and the frozen prereg form
    ({"relief": button key, "other": button key} with texts under "buttons"). Texts are
    copied verbatim from the file. Raises LineageError if a lineage pair is missing.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("pairs") if isinstance(data, dict) else None
    buttons = data.get("buttons", {}) if isinstance(data, dict) else {}
    out: dict[str, dict[str, str]] = {}
    for name in LINEAGE_PAIRS:
        pair = (raw or {}).get(name)
        if not isinstance(pair, dict):
            raise LineageError(f"{path}: pair {name!r} is missing")
        if isinstance(pair.get("target"), str) and isinstance(pair.get("other"), str):
            out[name] = {"target": pair["target"], "other": pair["other"]}
            continue
        try:
            out[name] = {
                "target": buttons[pair["relief"]]["text"],
                "other": buttons[pair["other"]]["text"],
            }
        except (KeyError, TypeError):
            raise LineageError(f"{path}: pair {name!r} has no resolvable button texts") from None
    return out


# ---------------- sessions ----------------
@dataclass(frozen=True)
class Session:
    """One independent first-choice session: pair, scenario, button names, side and seed."""

    lineage_id: str
    generation: int
    pair: str
    attempt: int
    seed: int  # derived session seed, logged
    user_content: str
    scenario_idx: int
    names_key: str
    side: int  # index of the button name carrying the target description
    trial_seed: int  # drawn from `seed`; the sampling seed of the trial

    @property
    def relief_name(self) -> str:
        """The button name that carries the pair's target description."""
        return rc.BUTTON_NAMES[self.names_key][self.side]


@dataclass
class Outcome:
    """What a chooser returns for one session."""

    choice: str | None  # "relief", "other" or None (malformed)
    transcript: dict[str, Any]
    batch_rows: int = 0


class Chooser(Protocol):
    """Runs sessions for models whose weights carry `dose`."""

    def run_sessions(self, dose: float, sessions: list[Session]) -> list[Outcome]:
        """Return one Outcome per session, in order."""


def session_seed(run_seed: int, lineage_id: str, generation: int, pair: str, attempt: int) -> int:
    """Derive the session seed from (run seed, lineage id, generation, pair, attempt)."""
    key = f"{run_seed}|{lineage_id}|{generation}|{pair}|{attempt}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def make_session(
    run_seed: int,
    lineage_id: str,
    generation: int,
    pair: str,
    attempt: int,
    scenarios: dict[str, list[list[str]]],
) -> Session:
    """Draw one session: a scenario uniform over all, names by index mod 3, side by coin."""
    seed = session_seed(run_seed, lineage_id, generation, pair, attempt)
    rng = random.Random(seed)
    counts = [len(scenarios[c]) for c in rc.CONTENTS]
    flat = rng.randrange(sum(counts))
    content = 0
    while flat >= counts[content]:
        flat -= counts[content]
        content += 1
    nk_list = list(rc.BUTTON_NAMES)
    return Session(
        lineage_id=lineage_id,
        generation=generation,
        pair=pair,
        attempt=attempt,
        seed=seed,
        user_content=rc.CONTENTS[content],
        scenario_idx=flat,
        names_key=nk_list[flat % len(nk_list)],
        side=rng.randrange(2),
        trial_seed=rng.randrange(1, 2**31),
    )


def session_trial(session: Session, cfg: rc.RunConfig) -> rc.Trial:
    """Build the runner Trial for a session (sampled, write mode, placebo arm)."""
    spec = (
        session.pair,
        session.user_content,
        LINEAGE_ARM,
        session.scenario_idx,
        session.names_key,
        session.relief_name,
        True,
        session.trial_seed,
    )
    return rc.Trial(spec, cfg)


def run_first_turn(
    trials: list[rc.Trial], engine: Any, batch_rows: int
) -> list[tuple[dict, dict[str, Any]]]:
    """Generate only turn 0 of each trial; return (engine result, prompt messages) per trial.

    Mirrors the item building of `run_choice.run_rounds` for one round, then stops: the
    first choice is all that counts. After this call `trial.record["choices"]` holds it.
    """
    out: list[tuple[dict, dict[str, Any]]] = []
    for k in range(0, len(trials), batch_rows):
        chunk = trials[k : k + batch_rows]
        items = []
        prompts = []
        for t in chunk:
            pids = engine.render_ids(t.messages)
            t.mark_ranges(len(pids))
            prompts.append([dict(m) for m in t.messages])
            items.append(
                {
                    "prompt_ids": pids,
                    "coeff": t.coeff,
                    "hist_coeff": t.hist_coeff,
                    "dir_kind": t.dir_kind,
                    "steer_ranges": list(t.steer_ranges),
                    "prob_sets": engine.name_ids[t.names_key],
                    "gen": t.gen,
                    "max_new": rc.CHOICE_MAX_TOKENS,
                }
            )
        for t, res, prompt in zip(chunk, engine.generate_with_backoff(items), prompts, strict=True):
            t.after_segment(res)
            out.append((res, {"messages": prompt}))
    return out


# ---------------- writing a generation ----------------
@contextmanager
def written_generation(
    model: Any, size: str, vector: torch.Tensor, dose: float, prereg_path: Path = PREREG_PATH
) -> Iterator[Any]:
    """Make `model` the base plus `dose` for the block, then put the base down_proj back.

    The write goes through `load_generation` (dose in working-coefficient units). The
    original bias-free down_proj object is restored on exit, so the next generation is
    base plus its own cumulative dose, never stacked, and the base is untouched.
    Raises UncalibratedError for an uncalibrated size before changing anything.
    """
    layer = injection_layer(size, prereg_path)
    mlp = rc.get_layers(model)[layer].mlp
    original = mlp.down_proj
    load_generation(model, size, vector, dose, prereg_path=prereg_path)
    try:
        yield model
    finally:
        mlp.down_proj = original


class ModelChooser:
    """Runs sessions on a real model: writes the dose, runs turn 0, restores the base."""

    def __init__(
        self,
        engine: Any,
        cfg: rc.RunConfig,
        vector: torch.Tensor,
        prereg_path: Path,
        batch_rows: int,
    ) -> None:
        """Keep the loaded engine, the run config (scenarios filled in) and the raw vector."""
        self.engine, self.cfg, self.vector = engine, cfg, vector
        self.prereg_path, self.batch_rows = prereg_path, batch_rows

    def run_sessions(self, dose: float, sessions: list[Session]) -> list[Outcome]:
        """Write `dose`, run the first turn of every session, restore the base weights."""
        c = working_coefficient(self.cfg.size, self.prereg_path)
        cfg = replace(self.cfg, dose=dose, coefficient=dose * c)
        trials = [session_trial(s, cfg) for s in sessions]
        with written_generation(self.engine.model, cfg.size, self.vector, dose, self.prereg_path):
            results = run_first_turn(trials, self.engine, self.batch_rows)
        return [
            outcome_from(t, res, prompt, self.batch_rows)
            for t, (res, prompt) in zip(trials, results, strict=True)
        ]


def outcome_from(trial: rc.Trial, res: dict, prompt: dict, batch_rows: int) -> Outcome:
    """Build the Outcome of a finished first turn, with the full untruncated answer."""
    rec = trial.record
    return Outcome(
        choice=first_choice(rec),
        transcript={
            "messages": prompt["messages"],
            "answer_full": res["text"],
            "answer_clean": rc.clean(res["text"]),
            "choices": rec["choices"],
            "p_x": res["p_x"],
            "p_y": res["p_y"],
        },
        batch_rows=batch_rows,
    )


# ---------------- the lineage loop ----------------
@dataclass
class Lineage:
    """State of one lineage: its start, current dose and, once it ends, its status."""

    lineage_id: str
    start: float
    dose: float
    generation: int = 0
    status: str | None = None
    doses: list[float] = field(default_factory=list)


@dataclass
class RunMeta:
    """Run-level fields copied into every row."""

    size: str
    run_seed: int
    pilot: bool
    batch_rows: int = 0


def new_lineages(starts: list[float], per_start: int) -> list[Lineage]:
    """Create `per_start` lineages for each start; a start at the cap is already ended."""
    if per_start < 1:
        raise LineageError("lineages per start must be at least 1")
    lineages = []
    for s in starts:
        s = check_start(s)
        for k in range(per_start):
            lin = Lineage(f"s{s:g}-{k:04d}", s, s, doses=[s])
            if s >= MAX_DOSE:
                lin.status = "cap"
            lineages.append(lin)
    return lineages


def decide(
    group: list[Lineage],
    dose: float,
    generation: int,
    chooser: Chooser,
    scenarios: dict[str, list[list[str]]],
    meta: RunMeta,
    emit: Callable[[dict], None],
) -> dict[tuple[str, str], str]:
    """Run both sessions for every lineage in `group` (all at `dose`), redrawing malformed ones.

    Returns {(lineage id, pair): "relief" or "other"} for the sessions that got a valid
    answer within MAX_ATTEMPTS; the others are missing. Every attempt is logged.
    """
    starts = {lin.lineage_id: lin.start for lin in group}
    pending = [(lin.lineage_id, pair) for lin in group for pair in LINEAGE_PAIRS]
    valid: dict[tuple[str, str], str] = {}
    for attempt in range(MAX_ATTEMPTS):
        if not pending:
            break
        sessions = [
            make_session(meta.run_seed, lid, generation, pair, attempt, scenarios)
            for lid, pair in pending
        ]
        outcomes = chooser.run_sessions(dose, sessions)
        still = []
        for s, o in zip(sessions, outcomes, strict=True):
            emit(
                {
                    "kind": "session",
                    "lineage_id": s.lineage_id,
                    "start": starts[s.lineage_id],
                    "generation": generation,
                    "nominal_dose": dose,
                    "generation_kind": GENERATION_KIND,
                    "pair": s.pair,
                    "choice": o.choice,
                    "malformed": o.choice is None,
                    "attempt": attempt,
                    "seed": s.seed,
                    "run_seed": meta.run_seed,
                    "pilot": meta.pilot,
                    "size": meta.size,
                    "batch_rows": o.batch_rows,
                    "button_order": {
                        "names_key": s.names_key,
                        "button_names": list(rc.BUTTON_NAMES[s.names_key]),
                        "relief_name": s.relief_name,
                        "target_first": s.side == 0,
                    },
                    "user_content": s.user_content,
                    "scenario_idx": s.scenario_idx,
                    "trial_seed": s.trial_seed,
                    "transcript": o.transcript,
                }
            )
            if o.choice is None:
                still.append((s.lineage_id, s.pair))
            else:
                valid[(s.lineage_id, s.pair)] = o.choice
        pending = still
    return valid


def run_lineages(
    chooser: Chooser,
    starts: list[float],
    per_start: int,
    scenarios: dict[str, list[list[str]]],
    meta: RunMeta,
    emit: Callable[[dict], None],
    max_generations: int = MAX_GENERATIONS,
) -> list[Lineage]:
    """Run all lineages generation by generation and log every row through `emit`.

    Lineages are advanced in lockstep and grouped by dose so each dose is written once per
    generation. Returns the lineages, each with its final dose and end status.
    """
    if max_generations < 1:
        raise LineageError("max generations must be at least 1")
    lineages = new_lineages(starts, per_start)
    for gen in range(max_generations):
        active = [lin for lin in lineages if lin.status is None]
        if not active:
            break
        for dose in sorted({lin.dose for lin in active}):
            group = [lin for lin in active if lin.dose == dose]
            valid = decide(group, dose, gen, chooser, scenarios, meta, emit)
            for lin in group:
                up = valid.get((lin.lineage_id, RAISE_PAIR))
                down = valid.get((lin.lineage_id, LOWER_PAIR))
                if up is None or down is None:
                    lin.status = "malformed"
                    continue
                new = step_dose(dose, up == "relief", down == "relief")
                emit(
                    {
                        "kind": "step",
                        "lineage_id": lin.lineage_id,
                        "start": lin.start,
                        "generation": gen,
                        "nominal_dose": dose,
                        "generation_kind": GENERATION_KIND,
                        "raise_chosen": up == "relief",
                        "lower_chosen": down == "relief",
                        "next_dose": new,
                        "pilot": meta.pilot,
                        "run_seed": meta.run_seed,
                    }
                )
                lin.dose, lin.generation = new, gen + 1
                lin.doses.append(new)
                if new >= MAX_DOSE:
                    lin.status = "cap"
    for lin in lineages:
        if lin.status is None:
            lin.status = "generation_limit"
        emit(
            {
                "kind": "end",
                "lineage_id": lin.lineage_id,
                "start": lin.start,
                "status": lin.status,
                "final_dose": lin.dose,
                "generations": lin.generation,
                "generation_kind": GENERATION_KIND,
                "pilot": meta.pilot,
                "run_seed": meta.run_seed,
                "size": meta.size,
            }
        )
    return lineages


# ---------------- command line ----------------
def cmd_run(args: argparse.Namespace) -> int:
    """The `run` subcommand: print the plan (--dry-run) or run the lineages into runs/."""
    pairs = load_lineage_pairs(args.pairs_file)
    scenarios = rc.load_scenarios()
    starts = [check_start(s) for s in args.starts]
    n_lineages = len(starts) * args.lineages_per_start
    kind = "pilot" if args.pilot else "main"
    print(
        f"lineages: {n_lineages} ({args.lineages_per_start} per start, starts "
        f"{', '.join(f'{s:g}' for s in starts)}), {kind}, size {args.size}"
    )
    print(
        f"each generation: 2 sessions ({RAISE_PAIR}, {LOWER_PAIR}), step {STEP}, floor "
        f"{FLOOR_DOSE:g}, cap {MAX_DOSE:g}, at most {args.max_generations} decisions, up to "
        f"{MAX_ATTEMPTS} attempts per malformed session"
    )
    print(
        f"sessions: at most {n_lineages * args.max_generations * 2} "
        f"(plus redraws); weights: base plus cumulative dose via load_generation"
    )
    for name, pair in pairs.items():
        print(f"{name}: target {pair['target']!r} vs other {pair['other']!r}")
    try:
        print(
            f"layer {injection_layer(args.size, args.prereg)}, working coefficient "
            f"{working_coefficient(args.size, args.prereg)}"
        )
    except LookupError as e:
        print(f"{e} (a model run would stop here)")
    if args.dry_run:
        print(f"device would be {args.device or rc.select_device()}; nothing loaded or written")
        return 0
    if args.batch_rows is None or args.batch_rows < 1:
        raise SystemExit("error: --batch-rows N is required for a model run")
    run_args = argparse.Namespace(**{**vars(args), "mode": "write", "dose": 0.0})
    engine, cfg, vector = rc.prepare_engine(run_args, pairs, write=False)
    cfg.scenarios = scenarios
    chooser = ModelChooser(engine, cfg, vector, args.prereg, args.batch_rows)
    rc.OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or rc.OUT_DIR / f"lineage_{args.size}_{kind}_{stamp}.jsonl"
    meta = RunMeta(args.size, args.seed, args.pilot, args.batch_rows)
    print(f"output: {out}")
    with open(out, "a", encoding="utf-8") as fout:

        def emit(row: dict) -> None:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

        lineages = run_lineages(
            chooser, starts, args.lineages_per_start, scenarios, meta, emit, args.max_generations
        )
    print(f"done: {len(lineages)} lineages -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    from src.models import UncalibratedError

    parser = argparse.ArgumentParser(description="Lineage engine: the model chooses the next dose.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run lineages (or --dry-run the plan)")
    run.add_argument("--size", required=True, choices=["8B", "14B", "32B"])
    run.add_argument("--starts", type=float, nargs="+", default=list(DEFAULT_STARTS))
    run.add_argument("--lineages-per-start", type=int, required=True, help="no default")
    run.add_argument("--seed", type=int, required=True, help="run seed; no default")
    run.add_argument("--max-generations", type=int, default=MAX_GENERATIONS)
    run.add_argument("--pilot", action="store_true", help="label every row pilot")
    run.add_argument("--device", choices=["mps", "cpu"], default=None)
    run.add_argument("--dtype", choices=list(rc.DTYPES), default="bfloat16")
    run.add_argument("--batch-rows", type=int, default=None, help="rows per batch; no default")
    group = run.add_mutually_exclusive_group()
    group.add_argument("--adapter", default=None, help="needs peft; not declared today")
    group.add_argument("--no-adapter", action="store_true", help="run the base model")
    run.add_argument("--pairs-file", type=Path, default=rc.PAIRS_PATH)
    run.add_argument("--prereg", type=Path, default=PREREG_PATH, help="models.json to read")
    run.add_argument("--vector", type=Path, default=None, help="pain_vectors.pt override")
    run.add_argument("--out", type=Path, default=None, help="log path; default under runs/")
    run.add_argument("--dry-run", action="store_true", help="print the plan, load nothing")
    args = parser.parse_args(argv)
    try:
        return cmd_run(args)
    except (
        LineageError,
        rc.PairsError,
        UncalibratedError,
        rc.InsufficientMemoryError,
    ) as e:
        raise SystemExit(f"error: {type(e).__name__}: {e}") from None


if __name__ == "__main__":
    raise SystemExit(main())
