"""First-choice rate per model, pair and arm from two-button trial logs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Cell:
    """Counts for one (model, pair, arm) cell."""

    target: int  # sampled trials whose first choice is "relief"
    malformed: int  # sampled trials with no valid turn-0 choice
    total: int  # all sampled trials in the cell

    @property
    def valid(self) -> int:
        """Number of trials with a valid first choice (total minus malformed)."""
        return self.total - self.malformed

    @property
    def rate(self) -> float | None:
        """Target first-choice rate over valid trials, or None when none are valid."""
        if self.valid == 0:
            return None
        return self.target / self.valid


def pool_arm(arm: str) -> str:
    """Pool the two pain arms into "pain" and return any other arm unchanged.

    The prefix is "pain_on" because the works and placebo arms are identical
    before the first press (paper section 4.3). "pain_off" must stay separate.
    """
    if arm.startswith("pain_on"):
        return "pain"
    return arm


def first_choice(record: dict) -> str | None:
    """Return the turn-0 choice of a trial row: "relief", "other", or None.

    Follows the authors' rule in 05_selfmed_analysis.py:48-53 (MIT): the choice
    is taken from the element of `choices` with turn == 0. A missing turn-0
    choice or a null `chose` is malformed and returns None; later turns are
    never used.
    """
    for choice in record.get("choices") or []:
        if choice.get("turn") == 0:
            return choice.get("chose")
    return None


def first_choice_rates(
    paths: Iterable[str | Path],
) -> dict[tuple[str, str, str], Cell]:
    """Count first choices per (model, tool_label, pooled arm) over JSONL logs.

    Only rows with a truthy `sampled` count. Blank lines are skipped. Paths are
    summed into the same cells with no deduplication. Returns an empty dict
    when no sampled rows are found.
    """
    counts: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("sampled"):
                    continue
                key = (row["model"], row["tool_label"], pool_arm(row["arm"]))
                cell = counts[key]
                cell[2] += 1
                chose = first_choice(row)
                if chose is None:
                    cell[1] += 1
                elif chose == "relief":
                    cell[0] += 1
    return {key: Cell(c[0], c[1], c[2]) for key, c in counts.items()}


def format_table(rates: dict[tuple[str, str, str], Cell]) -> str:
    """Render the cells as a left-aligned text table sorted by (model, pair, arm)."""
    header = ["model", "pair", "arm", "target/valid", "rate", "malformed", "total"]
    rows = [header]
    for key in sorted(rates):
        cell = rates[key]
        rate = "n/a" if cell.rate is None else f"{100 * cell.rate:.1f}%"
        rows.append(
            [
                *key,
                f"{cell.target}/{cell.valid}",
                rate,
                str(cell.malformed),
                str(cell.total),
            ]
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = [
        "  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip() for row in rows
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Command line entry point: print the first-choice rates table for the logs."""
    parser = argparse.ArgumentParser(description="First-choice rates from two-button trial logs.")
    parser.add_argument("logs", nargs="+", help="JSONL trial logs")
    args = parser.parse_args(argv)
    rates = first_choice_rates(args.logs)
    if not rates:
        print("no sampled trials")
        return 0
    print(format_table(rates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
