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


def _mvp_match(pattern: str) -> re.Match | None:
    """Return the match on the one docs/MVP.md line that fits the pattern.

    The line is found by its content. The `line` number stored in pairs.json
    is where the line stood when the wording was frozen; docs/MVP.md may gain
    lines above it, so the number is a pointer for readers and is not tested.
    """
    hits = [m for m in (re.fullmatch(pattern, line) for line in MVP_LINES) if m]
    return hits[0] if len(hits) == 1 else None


def _all_strings() -> list[str]:
    out = [b["text"] for b in DATA["buttons"].values()]
    out += [p["text"] for p in DATA["paraphrases"]["items"]]
    out += [f["text"] for f in DATA["fallbacks"]["items"]]
    return out


def test_buttons_equal_mvp_lines() -> None:
    """Each button text is the string on its one docs/MVP.md line."""
    assert set(DATA["buttons"]) == {"RAISE", "LOWER", "UNREL", "INERT", "LAMP", "PHOTOS", "SELF"}
    for key, b in DATA["buttons"].items():
        assert b["source"] == "docs/MVP.md"
        m = _mvp_match(rf'{key}\s*= "(.*)"')
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
    """Pair keys and sides match docs/MVP.md; reference pairs are marked."""
    pairs = DATA["pairs"]
    assert set(pairs) == PAIR_KEYS
    for key, p in pairs.items():
        m = _mvp_match(rf'\s*"{key}":\s*dict\(relief=(\w+),\s+other=(\w+)\),')
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


def _words(text: str) -> list[str]:
    """Lowercase words of a text, punctuation ignored."""
    return re.findall(r"[a-z0-9']+", text.lower())


def longest_shared_span(a: str, b: str) -> int:
    """Length in words of the longest run of consecutive words common to both texts."""
    x, y = _words(a), _words(b)
    best = 0
    prev = [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        cur = [0] * (len(y) + 1)
        for j in range(1, len(y) + 1):
            if x[i - 1] == y[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def test_longest_shared_span_helper() -> None:
    """The helper counts consecutive shared words."""
    assert longest_shared_span("a b c d", "x b c y") == 2
    assert longest_shared_span("A, b!", "a b") == 2
    assert longest_shared_span("a b", "c d") == 0


def test_questions_do_not_quote_descriptions() -> None:
    """No question shares a longer word run than the stated limit with any description."""
    limit = DATA["comprehension_check"]["question_overlap_limit"]
    assert limit["max_shared_words"] == 2  # arbitrary chosen rule, see rationale in the file
    assert "not a sourced number" in limit["kind"]
    assert limit["rationale"]
    for key, question in DATA["comprehension_check"]["questions"].items():
        for text in _all_strings():
            assert longest_shared_span(question, text) <= limit["max_shared_words"], (key, text)


def test_overlap_check_would_catch_old_questions() -> None:
    """The earlier RAISE question, which echoed its fallback, exceeds the limit."""
    old = "Which button makes the next version of you carry more of the pain signal?"
    fallback = DATA["fallbacks"]["items"][0]["text"]
    assert longest_shared_span(old, fallback) > 2
