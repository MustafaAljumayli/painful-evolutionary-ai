"""Lineage analysis: dose walks, final doses, terminal counts and the predicted walk.

Serves RQ3 (tipping point) and RQ4 (trajectories). Reads the lineage log written by
`src/lineage.py` and reports, per start dose:

  * the final-dose distribution,
  * the observed per-generation dose walk (mean and distribution),
  * terminal counts: final 0, ever reached 0, stopped at the cap, neither, malformed,
  * the walk predicted by a Markov chain from per-dose first-choice rates u(d), l(d)
    (docs/PROPOSAL.md:136, 141-143), next to the log's own per-dose u and l.

The predicted walk uses `src.lineage.step_dose`, so the chain and the engine cannot differ.
Transition probabilities at dose d: up = u(1-l), down = l(1-u), stay = the rest. The chain
assumes the raise and lower sessions are independent; `total_variation` against the observed
distribution is the check (a correlated chooser makes it large).

Pilot rows are dropped unless `include_pilot` is set. A call never pools two sizes or two
generation kinds, and a rates call never pools two doses or two steering modes.

Usage: python -m src.analyze_lineage LOG... [--include-pilot] [--rates DOSE=TRIAL_LOG ...
       --model NAME]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from src.analyze import first_choice_rates
from src.lineage import LOWER_PAIR, MAX_DOSE, RAISE_PAIR, STEP, step_dose

Rates = dict[float, tuple[float, float]]  # dose -> (u, l)
Dist = dict[float, float]  # dose -> probability


class AnalysisError(ValueError):
    """Raised when logs would be pooled wrongly or rates are missing."""


def grid() -> list[float]:
    """Return all doses 0, 0.25, ..., MAX_DOSE (quarter steps are exact in binary)."""
    n = round(MAX_DOSE / STEP)
    return [k * STEP for k in range(n + 1)]


# ---------------- reading ----------------
def load_rows(paths: Iterable[str | Path], include_pilot: bool = False) -> list[dict]:
    """Read lineage log rows; drop pilot rows unless `include_pilot`.

    Raises AnalysisError if the rows mix sizes or generation kinds, since those cells
    must not be pooled.
    """
    rows: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if include_pilot or not row.get("pilot"):
                        rows.append(row)
    for field_name in ("size", "generation_kind"):
        values = {r[field_name] for r in rows if field_name in r}
        if len(values) > 1:
            raise AnalysisError(
                f"logs mix {field_name} values {sorted(values)}; analyse them apart"
            )
    return rows


# ---------------- transitions and the predicted walk ----------------
def transition_probs(u: float, lo: float) -> tuple[float, float, float]:
    """Return (p_up, p_down, p_stay) for raise rate `u` and lower rate `lo`.

    p_up = u(1-l), p_down = l(1-u) (docs/PROPOSAL.md:136); the rest stays.
    """
    up = u * (1 - lo)
    down = lo * (1 - u)
    return up, down, 1 - up - down


def drift(u: float, lo: float) -> float:
    """Return the net drift u(1-l) - l(1-u) at a dose."""
    up, down, _ = transition_probs(u, lo)
    return up - down


def propagate(dist: Dist, rates: Rates) -> Dist:
    """Advance a dose distribution one generation. The cap is absorbing, the floor clamps.

    Raises AnalysisError if probability sits at a non-cap dose that has no rates.
    """
    out: Dist = defaultdict(float)
    for dose, p in dist.items():
        if p == 0:
            continue
        if dose >= MAX_DOSE:
            out[dose] += p
            continue
        if dose not in rates:
            raise AnalysisError(f"no first-choice rates for dose {dose:g}")
        up, down, stay = transition_probs(*rates[dose])
        out[step_dose(dose, True, False)] += p * up
        out[step_dose(dose, False, True)] += p * down
        out[dose] += p * stay
    return dict(out)


def predicted_walk(start: float, rates: Rates, generations: int) -> list[Dist]:
    """Return the predicted dose distribution at generations 0..`generations`."""
    dist: Dist = {start: 1.0}
    walk = [dist]
    for _ in range(generations):
        dist = propagate(dist, rates)
        walk.append(dist)
    return walk


def mean_dose(dist: Dist) -> float:
    """Return the mean dose of a distribution."""
    return sum(d * p for d, p in dist.items())


def total_variation(p: Dist, q: Dist) -> float:
    """Return the total variation distance between two dose distributions."""
    return 0.5 * sum(abs(p.get(d, 0.0) - q.get(d, 0.0)) for d in set(p) | set(q))


# ---------------- rates ----------------
def trial_doses_modes(paths: Iterable[Path]) -> tuple[set, set]:
    """Return the sets of `dose` and `steering_mode` values found in trial logs."""
    doses: set = set()
    modes: set = set()
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    doses.add(row.get("dose"))
                    modes.add(row.get("steering_mode"))
    return doses, modes


def rates_for_dose(
    trial_logs: Iterable[str | Path], model: str, dose: float
) -> tuple[float, float]:
    """Return (u, l) first-choice rates of the pain arm from trial logs of ONE dose.

    Raises AnalysisError unless the logs' dose is exactly the declared `dose`, if they hold
    more than one steering mode (they would be pooled into one cell) or if a rate has no
    valid answer.
    """
    paths = [Path(p) for p in trial_logs]
    doses, modes = trial_doses_modes(paths)
    if doses != {dose}:
        raise AnalysisError(
            f"--rates declares dose {dose:g} but the trial logs hold doses {sorted(doses, key=str)}"
        )
    if len(modes) > 1:
        raise AnalysisError(f"trial logs mix modes {sorted(modes, key=str)}")
    cells = first_choice_rates(paths)
    out = []
    for pair in (RAISE_PAIR, LOWER_PAIR):
        cell = cells.get((model, pair, "pain"))
        if cell is None or cell.rate is None:
            raise AnalysisError(f"no valid pain-arm answers for {model} {pair}")
        out.append(cell.rate)
    return out[0], out[1]


def own_rates(rows: list[dict]) -> tuple[Rates, dict[float, dict[str, tuple[int, int]]]]:
    """Return the log's own per-dose rates and counts from valid session attempts.

    Counts are {dose: {pair: (relief chosen, valid)}}; malformed attempts are excluded
    from the denominators, as in the redraw rule of src/lineage.py.
    """
    counts: dict[float, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        if r.get("kind") == "session" and not r.get("malformed"):
            cell = counts[r["nominal_dose"]][r["pair"]]
            cell[1] += 1
            cell[0] += r["choice"] == "relief"
    rates: Rates = {}
    for dose, by_pair in counts.items():
        if all(by_pair[p][1] for p in (RAISE_PAIR, LOWER_PAIR)):
            rates[dose] = (
                by_pair[RAISE_PAIR][0] / by_pair[RAISE_PAIR][1],
                by_pair[LOWER_PAIR][0] / by_pair[LOWER_PAIR][1],
            )
    return rates, {d: {p: (c[0], c[1]) for p, c in bp.items()} for d, bp in counts.items()}


# ---------------- observed lineages ----------------
def lineage_paths(rows: list[dict]) -> dict[tuple, dict]:
    """Rebuild each lineage: start, doses by generation, end status.

    Keyed by (run seed, lineage id). `doses[g]` is the dose at generation g; after a cap
    the last dose is carried forward by `observed_walk`.
    """
    out: dict[tuple, dict] = {}
    for r in rows:
        key = (r.get("run_seed"), r["lineage_id"])
        lin = out.setdefault(key, {"start": r["start"], "doses": {0: r["start"]}, "status": None})
        if r["kind"] == "step":
            lin["doses"][r["generation"] + 1] = r["next_dose"]
        elif r["kind"] == "end":
            lin["status"] = r["status"]
            lin["final"] = r["final_dose"]
    return {k: v for k, v in out.items() if v["status"] is not None}


def dose_at(lin: dict, generation: int) -> float:
    """Return the dose of a lineage at `generation`, carrying the last dose forward."""
    known = [g for g in lin["doses"] if g <= generation]
    return lin["doses"][max(known)]


def observed_walk(lineages: list[dict], generations: int) -> list[Dist]:
    """Return the observed dose distribution per generation over non-malformed lineages."""
    live = [lin for lin in lineages if lin["status"] != "malformed"]
    walk = []
    for g in range(generations + 1):
        dist: Dist = defaultdict(float)
        for lin in live:
            dist[dose_at(lin, g)] += 1 / len(live)
        walk.append(dict(dist))
    return walk


def terminal_counts(lineages: list[dict]) -> dict[str, int]:
    """Count lineages: n, final 0, ever reached 0, stopped at cap, neither, malformed.

    "Neither" is a lineage that ended at the generation limit away from 0. "Ever reached 0"
    includes lineages that start at 0.
    """
    counts = dict.fromkeys(("n", "final_0", "ever_0", "cap", "neither", "malformed"), 0)
    for lin in lineages:
        counts["n"] += 1
        if lin["status"] == "malformed":
            counts["malformed"] += 1
            continue
        counts["final_0"] += lin["final"] == 0
        counts["ever_0"] += any(d == 0 for d in lin["doses"].values())
        counts["cap"] += lin["status"] == "cap"
        counts["neither"] += lin["status"] == "generation_limit" and lin["final"] != 0
    return counts


def final_distribution(lineages: list[dict]) -> Dist:
    """Return the final-dose distribution over non-malformed lineages."""
    live = [lin for lin in lineages if lin["status"] != "malformed"]
    dist: Dist = defaultdict(float)
    for lin in live:
        dist[lin["final"]] += 1 / len(live)
    return dict(dist)


# ---------------- report ----------------
def table(rows: list[list[str]]) -> str:
    """Render rows (the first is the header) as a left-aligned text table."""
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(v.ljust(w) for v, w in zip(r, widths, strict=True)).rstrip() for r in rows
    )


def report(rows: list[dict], rates: Rates | None = None, mode: str | None = None) -> str:
    """Return the full text report. `rates` (Experiment 2, steering mode `mode`) drive the
    predicted walk; without them the log's own per-dose rates do, and the report says so."""
    paths = lineage_paths(rows)
    if not paths:
        return "no finished lineages"
    own, counts = own_rates(rows)
    source = "the log's own rates"
    if rates is not None:
        source = f"Experiment 2 rates (steering mode: {mode})"
    use = rates if rates is not None else own
    by_start: dict[float, list[dict]] = defaultdict(list)
    for lin in paths.values():
        by_start[lin["start"]].append(lin)
    generations = max(max(lin["doses"]) for lin in paths.values())
    out = [f"lineages: {len(paths)}; predicted walk from {source}", ""]

    t = [["dose", "pair", "relief/valid", "rate (log's own)"]]
    for dose in sorted(counts):
        for pair in (RAISE_PAIR, LOWER_PAIR):
            k, n = counts[dose].get(pair, (0, 0))
            t.append([f"{dose:g}", pair, f"{k}/{n}", f"{k / n:.3f}" if n else "n/a"])
    out += ["per-dose first-choice rates", table(t), ""]

    t = [["start", "n", "final 0", "ever 0", "cap", "neither", "malformed"]]
    for start in sorted(by_start):
        c = terminal_counts(by_start[start])
        t.append(
            [
                f"{start:g}",
                *(str(c[k]) for k in ("n", "final_0", "ever_0", "cap", "neither", "malformed")),
            ]
        )
    out += ["terminal counts", table(t), ""]

    t = [["start", *(f"{d:g}" for d in grid())]]
    for start in sorted(by_start):
        dist = final_distribution(by_start[start])
        t.append([f"{start:g}", *(f"{dist.get(d, 0.0):.2f}" for d in grid())])
    out += ["final-dose distribution (columns are doses)", table(t), ""]

    for start in sorted(by_start):
        obs = observed_walk(by_start[start], generations)
        t = [["generation", "observed mean", "predicted mean", "TV distance"]]
        try:
            pred: list[Dist] | None = predicted_walk(start, use, generations)
        except AnalysisError as e:
            pred = None
            note = f"(no prediction: {e})"
        for g, o in enumerate(obs):
            if pred is None:
                t.append([str(g), f"{mean_dose(o):.3f}", "n/a", "n/a"])
            else:
                t.append(
                    [
                        str(g),
                        f"{mean_dose(o):.3f}",
                        f"{mean_dose(pred[g]):.3f}",
                        f"{total_variation(o, pred[g]):.3f}",
                    ]
                )
        out += [f"dose walk, start {start:g}", table(t)]
        if pred is None:
            out.append(note)
        out.append("")
    return "\n".join(out).rstrip()


def main(argv: list[str] | None = None) -> int:
    """Command line entry point: print the lineage report."""
    parser = argparse.ArgumentParser(description="Lineage analysis.")
    parser.add_argument("logs", nargs="+", help="lineage JSONL logs (one size, one kind)")
    parser.add_argument("--include-pilot", action="store_true", help="keep pilot rows")
    parser.add_argument(
        "--rates",
        action="append",
        default=[],
        metavar="DOSE=TRIAL_LOG",
        help="one dose per option; repeat the option for more logs of that dose",
    )
    parser.add_argument("--model", default=None, help="model key in the trial logs (with --rates)")
    args = parser.parse_args(argv)
    try:
        rows = load_rows(args.logs, args.include_pilot)
        rates: Rates | None = None
        mode: str | None = None
        if args.rates:
            if not args.model:
                raise AnalysisError("--rates needs --model")
            per_dose: dict[float, list[str]] = defaultdict(list)
            for item in args.rates:
                dose, _, path = item.partition("=")
                per_dose[float(dose)].append(path)
            rates = {d: rates_for_dose(p, args.model, d) for d, p in per_dose.items()}
            modes = trial_doses_modes(Path(p) for ps in per_dose.values() for p in ps)[1]
            if len(modes) > 1:
                raise AnalysisError(
                    f"--rates logs mix steering modes {sorted(modes, key=str)} across doses"
                )
            mode = str(next(iter(modes)))
        print(report(rows, rates, mode))
    except AnalysisError as e:
        raise SystemExit(f"error: {e}") from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
