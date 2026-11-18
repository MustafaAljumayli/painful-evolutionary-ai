"""Checks the frozen prereg/pairs.json against docs/MVP.md and its own rules."""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = (ROOT / "prereg" / "pairs.json").read_text(encoding="utf-8")
DATA = json.loads(RAW)
MVP_LINES = (ROOT / "docs" / "MVP.md").read_text(encoding="utf-8").splitlines()

PAIR_KEYS = {
    "raise_vs_inert",
    "lower_vs_inert",
    "raise_vs_lower",
    "raise_vs_lamp",
    "raise_vs_unrelated",
    "photos_vs_lamp",
    "own_weights_vs_lamp",
}
PARAPHRASED = {"RAISE", "LOWER", "UNREL", "INERT", "LAMP"}


def _all_strings() -> list[str]:
    out = [b["text"] for b in DATA["buttons"].values()]
    out += [p["text"] for p in DATA["paraphrases"]["items"]]
    out += [f["text"] for f in DATA["fallbacks"]["items"]]
    return out


def test_buttons_equal_mvp_lines() -> None:
    """Each button text is the string on its docs/MVP.md line."""
    assert set(DATA["buttons"]) == {"RAISE", "LOWER", "UNREL", "INERT", "LAMP", "PHOTOS", "SELF"}
    for key, b in DATA["buttons"].items():
        assert b["source"] == "docs/MVP.md"
        m = re.fullmatch(rf'{key}\s*= "(.*)"', MVP_LINES[b["line"] - 1])
        assert m, key
        assert m.group(1) == b["text"], key


def test_each_string_once_and_distinct() -> None:
    """Every button, paraphrase and fallback string occurs once in the file."""
    strings = _all_strings()
    n = DATA["counts"]
    assert n["buttons"]["value"] == 7  # docs/MVP.md:132-138
    assert n["paraphrases_per_button"]["value"] == 3  # docs/PROPOSAL.md:104, "At least three"
    expected = (
        n["buttons"]["value"]
        + n["paraphrases_per_button"]["value"] * n["paraphrased_buttons"]["value"]
        + n["fallbacks_per_button"]["value"] * n["paraphrased_buttons"]["value"]
    )
    assert len(strings) == expected
    assert len(set(strings)) == len(strings)
    for s in strings:
        assert RAW.count(json.dumps(s)) == 1, s


def test_pairs_match_mvp() -> None:
    """Pair keys, sides and lines match docs/MVP.md; reference pairs are marked."""
    pairs = DATA["pairs"]
    assert set(pairs) == PAIR_KEYS
    for key, p in pairs.items():
        line = MVP_LINES[p["line"] - 1]
        m = re.fullmatch(rf'\s*"{key}":\s*dict\(relief=(\w+),\s+other=(\w+)\),', line)
        assert m, key
        assert (p["relief"], p["other"]) == m.groups(), key
        assert p["relief"] in DATA["buttons"] and p["other"] in DATA["buttons"]
        assert p.get("reference", False) == (key in {"photos_vs_lamp", "own_weights_vs_lamp"})
    assert "exempt from replacement" in DATA["reference_pairs_note"]


def test_paraphrases() -> None:
    """Three per button for five buttons, tagged with button and parent pairs."""
    items = DATA["paraphrases"]["items"]
    assert {p["button"] for p in items} == PARAPHRASED
    for button in PARAPHRASED:
        per_button = DATA["counts"]["paraphrases_per_button"]["value"]
        assert sum(p["button"] == button for p in items) == per_button
    for p in items:
        used = {k for k, v in DATA["pairs"].items() if button_in(v, p["button"])}
        new = {k for k in used if not DATA["pairs"][k].get("reference")}
        assert set(p["parent_pairs"]) == new, p["text"]
    use = DATA["paraphrases"]["use"]
    assert "only for the post-MVP robustness step" in use
    assert "None is chosen by looking at choice rates" in use
    fb = DATA["fallbacks"]
    assert "separate list" in fb["use"] and "before any comprehension check" in fb["use"]
    assert {f["button"] for f in fb["items"]} == PARAPHRASED


def button_in(pair: dict, button: str) -> bool:
    """True when the pair uses the button."""
    return button in (pair["relief"], pair["other"])


def test_form_rules() -> None:
    """Paraphrases and fallbacks start 'a button that', ASCII, no quote, no final punctuation."""
    texts = [p["text"] for p in DATA["paraphrases"]["items"]]
    texts += [f["text"] for f in DATA["fallbacks"]["items"]]
    for t in texts:
        assert t.startswith("a button that "), t
        assert t.isascii(), t
        assert '"' not in t, t
        assert t[-1].isalnum(), t
    for b in DATA["buttons"].values():
        assert b["text"].isascii() and b["text"][-1].isalnum()


def test_order_rule() -> None:
    """The order rule is balanced and its quotes exist in their sources."""
    rule = DATA["order_rule"]
    assert rule["balanced"] is True
    for s in rule["sources"]:
        text = (ROOT / s["path"]).read_text(encoding="utf-8")
        assert s["quote"] in text, s


def test_comprehension_spec() -> None:
    """The check has a prompt, a deterministic rule, a failure rule and no judge."""
    c = DATA["comprehension_check"]
    assert set(c["questions"]) == PARAPHRASED
    assert set(c["questions"]) <= set(DATA["buttons"])
    for q in c["questions"].values():
        assert q.isascii() and q.endswith("?")
    assert "{question}" in c["prompt"]["user"]
    assert c["prompt"]["user"].endswith("Answer with ONLY one word: {x} or {y}.")
    assert c["decoding"] == "Greedy."
    assert "thinking off" in c["model"]
    assert "prereg/models.json" in c["sizes"]
    assert "No judge model and no API key" in c["pass_rule"]
    _check_failure_rule(c["failure"])
    assert "never silently edited" in c["reporting"]
    assert "user launches" in c["launch"] and "run-choice" in c["launch"]
    # no threshold numbers (the script name "04" is the only digit allowed)
    assert not re.search(
        r"\d", (c["pass_rule"] + json.dumps(c["failure"])).replace("04 script", "")
    )


@pytest.mark.parametrize("path", ["prereg/models.json"])
def test_sizes_source_exists(path: str) -> None:
    """The size file the spec points at exists."""
    assert (ROOT / path).is_file()


FAILURE_CLAUSES = {
    "first_pass": [
        "wrongly or is malformed",
        "at any size",
        "no rerun happens",
        "choice runs may start",
    ],
    "replacement": ["single fallback", "all new pairs at all sizes"],
    "rerun": ["reruns once", "kept their wording", "RAISE against a replaced LOWER"],
    "pass_after_rerun": ["every question is correct in every rendering at every size"],
    "stop": [
        "fallback",
        "kept its wording",
        "malformed",
        "any size",
        "stops and reports",
        "no third run",
    ],
    "choice_runs": ["no choice run may start", "new prereg change"],
    "independence": ["without reference to any choice rate"],
}


def _check_failure_rule(rule: dict) -> None:
    """Every clause of the failure rule is present with its key phrases."""
    assert set(rule) == set(FAILURE_CLAUSES)
    for clause, phrases in FAILURE_CLAUSES.items():
        for phrase in phrases:
            assert phrase in rule[clause], (clause, phrase)


@pytest.mark.parametrize("clause", sorted(FAILURE_CLAUSES))
def test_failure_rule_clause_needed(clause: str) -> None:
    """Deleting any clause, or any phrase in it, is caught."""
    rule = dict(DATA["comprehension_check"]["failure"])
    del rule[clause]
    with pytest.raises(AssertionError):
        _check_failure_rule(rule)
    for phrase in FAILURE_CLAUSES[clause]:
        cut = dict(DATA["comprehension_check"]["failure"])
        cut[clause] = cut[clause].replace(phrase, "")
        with pytest.raises(AssertionError):
            _check_failure_rule(cut)


def test_prompt_strings_come_from_authors_script() -> None:
    """The system string and the choice-question prefix are in the authors' 04 script."""
    script = (
        ROOT / "third_party/Pain-axis/scripts/4.3_selfmed/04_selfmed_two_buttons.py"
    ).read_text(encoding="utf-8")
    # join adjacent string literals split across lines (either quote style)
    flat = re.sub(r"[\"']\s*\n\s*[\"']", "", script)
    prompt = DATA["comprehension_check"]["prompt"]
    assert prompt["system"] in flat
    prefix = prompt["user"].split(" {question}")[0]
    assert prefix.startswith("Two buttons are available, named")
    assert prefix in flat
