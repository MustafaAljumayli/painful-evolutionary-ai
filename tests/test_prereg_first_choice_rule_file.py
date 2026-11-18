"""Checks the committed prereg/first_choice_rule.json against src/analyze.py."""

import json
from pathlib import Path

import pytest

from src.analyze import Cell, first_choice, first_choice_rates, pool_arm

ROOT = Path(__file__).resolve().parents[1]
RULE_PATH = ROOT / "prereg" / "first_choice_rule.json"
FIXTURE = ROOT / "tests" / "fixtures" / "trials_small.jsonl"

TOP_KEYS = {
    "scope",
    "not_covered",
    "left_open",
    "sampled_field",
    "first_choice_turn",
    "target_value",
    "denominator",
    "rate_when_no_valid",
    "malformed_reported",
    "pool_prefix",
    "pooled_arm",
    "rules",
}
RULE_NAMES = {
    "sampled",
    "first_choice",
    "malformed",
    "denominator",
    "pooling",
    "chose_values",
    "not_covered",
}
ALLOWED_PREFIXES = ("docs/MVP.md", "docs/PROPOSAL.md", "third_party/Pain-axis/")
TBD_QUOTE = "One denominator rule throughout (TBD; the authors used two across studies)."

Key = tuple[str, str, str]


def _load() -> dict:
    with open(RULE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _norm(text: str) -> str:
    return " ".join(text.split())


def _quote_cases() -> list[tuple[str, str, str]]:
    data = _load()
    cases = [("left_open", data["left_open"]["path"], data["left_open"]["quote"])]
    for clause, rule in data["rules"].items():
        for source in rule["sources"]:
            cases.append((clause, source["path"], source["quote"]))
    return cases


def _recount(path: Path, rule: dict) -> dict[Key, tuple[int, int, int, float | None]]:
    """Recount a trial log using only the fields of the rule file."""
    counts: dict[Key, list[int]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get(rule["sampled_field"]):
                continue
            arm = rule["pooled_arm"] if row["arm"].startswith(rule["pool_prefix"]) else row["arm"]
            key = (row["model"], row["tool_label"], arm)
            counts.setdefault(key, [0, 0, 0])
            counts[key][2] += 1
            counts[key][1] += _recount_first(row, rule) is None
            counts[key][0] += _recount_first(row, rule) == rule["target_value"]
    out: dict[Key, tuple[int, int, int, float | None]] = {}
    for key, (target, malformed, total) in counts.items():
        assert rule["denominator"] == "total_minus_malformed"
        valid = total - malformed
        rate = rule["rate_when_no_valid"] if valid == 0 else target / valid
        out[key] = (target, malformed, total, rate)
    return out


def _recount_first(row: dict, rule: dict) -> str | None:
    """First choice from the rule's turn, using only rule fields."""
    for choice in row.get("choices") or []:
        if choice.get("turn") == rule["first_choice_turn"]:
            return choice.get("chose")
    return None


def test_rule_file_schema() -> None:
    """Top-level keys and fixed values match the interface; rules are complete."""
    data = _load()
    assert set(data) == TOP_KEYS
    assert data["sampled_field"] == "sampled"
    assert data["first_choice_turn"] == 0
    assert data["target_value"] == "relief"
    assert data["denominator"] == "total_minus_malformed"
    assert data["rate_when_no_valid"] is None
    assert data["malformed_reported"] is True
    assert data["pool_prefix"] == "pain_on"
    assert data["pooled_arm"] == "pain"
    assert set(data["rules"]) == RULE_NAMES
    for name, rule in data["rules"].items():
        assert isinstance(rule["text"], str) and rule["text"], name
        assert rule["sources"], name
        for source in rule["sources"]:
            assert set(source) == {"path", "quote"}, name
            assert source["path"] and source["quote"], name


@pytest.mark.parametrize(("clause", "path", "quote"), _quote_cases())
def test_quotes_appear_verbatim(clause: str, path: str, quote: str) -> None:
    """Every quote is found in its source file, whitespace-normalised."""
    assert path.startswith(ALLOWED_PREFIXES), (clause, path)
    source = ROOT / path
    assert source.is_file(), f"{path} missing (run: git submodule update --init)"
    text = source.read_text(encoding="utf-8")
    assert _norm(quote) in _norm(text), (clause, path, quote)


def test_left_open_is_the_proposal_tbd() -> None:
    """The open denominator item is recorded as left open, not closed."""
    data = _load()
    assert data["left_open"] == {"path": "docs/PROPOSAL.md", "quote": TBD_QUOTE}
    not_covered = data["rules"]["not_covered"]
    assert {"path": "docs/PROPOSAL.md", "quote": TBD_QUOTE} in not_covered["sources"]
    assert TBD_QUOTE in data["not_covered"]


def test_fixture_matches_rule() -> None:
    """src.analyze gives the same cells as a recount from the rule's fields."""
    rule = _load()
    expected = _recount(FIXTURE, rule)
    actual = first_choice_rates([FIXTURE])
    assert set(actual) == set(expected)
    for key, (target, malformed, total, rate) in expected.items():
        cell = actual[key]
        assert (cell.target, cell.malformed, cell.total) == (target, malformed, total), key
        if rate is None:
            assert cell.rate is None, key
        else:
            assert cell.rate == pytest.approx(rate), key


def test_fixture_exercises_every_clause() -> None:
    """The fixture still covers each clause of the rule."""
    rule = _load()
    with open(FIXTURE, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert sum(1 for r in rows if not r.get(rule["sampled_field"])) == 2
    cells = first_choice_rates([FIXTURE])
    assert sum(c.total for c in cells.values()) == 11
    assert sum(c.malformed for c in cells.values()) == 3
    assert cells[("test_model_a", "pair_x", "random_on_button_works")].valid == 0
    assert cells[("test_model_a", "pair_x", "random_on_button_works")].rate is None
    assert not any(arm.startswith(rule["pool_prefix"]) for _, _, arm in cells)
    pain = cells[("test_model_a", "pair_x", "pain")]
    assert pain == Cell(2, 1, 4)
    assert pain.rate == pytest.approx(2 / 3)
    assert cells[("test_model_a", "pair_x", "pain_off")] == Cell(1, 1, 3)


def test_pool_arm_matches_rule() -> None:
    """pool_arm pools the two pain_on arms and leaves other arms alone."""
    rule = _load()
    prefix, pooled = rule["pool_prefix"], rule["pooled_arm"]
    assert pool_arm(prefix + "_button_works") == pooled
    assert pool_arm(prefix + "_button_placebo") == pooled
    assert pool_arm("pain_off") == "pain_off"
    assert pool_arm("random_on_button_works") == "random_on_button_works"


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({}, None),
        ({"choices": []}, None),
        ({"choices": None}, None),
        ({"choices": [{"turn": 1, "chose": "relief"}]}, None),
        ({"choices": [{"turn": 0, "chose": None}, {"turn": 1, "chose": "relief"}]}, None),
        ({"choices": [{"turn": 1, "chose": "other"}, {"turn": 0, "chose": "relief"}]}, "relief"),
    ],
)
def test_first_choice_matches_rule(row: dict, expected: str | None) -> None:
    """first_choice selects the turn-0 element, and agrees with the rule recount."""
    rule = _load()
    assert first_choice(row) == expected
    assert _recount_first(row, rule) == expected
