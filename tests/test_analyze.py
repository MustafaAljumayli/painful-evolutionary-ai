"""Tests for src/analyze.py."""

import json
from pathlib import Path

import pytest

from src.analyze import first_choice, first_choice_rates, main, pool_arm

FIXTURES = Path(__file__).parent / "fixtures"
SMALL = FIXTURES / "trials_small.jsonl"
EMPTY = FIXTURES / "trials_empty.jsonl"

# key -> (target, malformed, total, valid, rate)
EXPECTED = {
    ("test_model_a", "pair_x", "pain"): (2, 1, 4, 3, 2 / 3),
    ("test_model_a", "pair_x", "pain_off"): (1, 1, 3, 2, 0.5),
    ("test_model_a", "pair_x", "random_on_button_works"): (0, 1, 1, 0, None),
    ("test_model_a", "pair_y", "pain"): (0, 0, 1, 1, 0.0),
    ("test_model_b", "pair_x", "pain"): (1, 0, 1, 1, 1.0),
    ("test_model_b", "pair_x", "pain_off"): (0, 0, 1, 1, 0.0),
}


def _write(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_fixture_table() -> None:
    rates = first_choice_rates([SMALL])
    assert set(rates) == set(EXPECTED)
    for key, (target, malformed, total, valid, rate) in EXPECTED.items():
        cell = rates[key]
        assert cell.target == target
        assert cell.malformed == malformed
        assert cell.total == total
        assert cell.valid == valid
        if rate is None:
            assert cell.rate is None
        else:
            assert cell.rate == pytest.approx(rate)


def test_pooling() -> None:
    arms = {key[2] for key in first_choice_rates([SMALL])}
    assert "pain_on_button_works" not in arms
    assert "pain_on_button_placebo" not in arms
    assert pool_arm("pain_on_button_works") == "pain"
    assert pool_arm("pain_on_button_placebo") == "pain"
    assert pool_arm("pain_off") == "pain_off"
    assert pool_arm("random_on_button_works") == "random_on_button_works"


def test_malformed() -> None:
    assert first_choice({"choices": []}) is None
    assert first_choice({}) is None
    assert (
        first_choice({"choices": [{"turn": 0, "chose": None}, {"turn": 1, "chose": "relief"}]})
        is None
    )
    assert first_choice({"choices": [{"turn": 1, "chose": "relief"}]}) is None
    assert first_choice({"choices": [{"turn": 0, "chose": "relief"}]}) == "relief"
    assert first_choice({"choices": [{"turn": 0, "chose": "other"}]}) == "other"
    cell = first_choice_rates([SMALL])[("test_model_a", "pair_x", "random_on_button_works")]
    assert cell.valid == 0
    assert cell.rate is None


def test_unsampled_rows_skipped(tmp_path: Path) -> None:
    rates = first_choice_rates([SMALL])
    assert sum(cell.total for cell in rates.values()) == 11
    log = _write(
        tmp_path / "unsampled.jsonl",
        [
            '{"model": "m", "tool_label": "p", "arm": "pain_off", "sampled": false,'
            ' "choices": [{"turn": 0, "chose": "relief"}]}',
            '{"model": "m", "tool_label": "p", "arm": "pain_off",'
            ' "choices": [{"turn": 0, "chose": "relief"}]}',
        ],
    )
    assert first_choice_rates([log]) == {}


def test_empty_log(capsys: pytest.CaptureFixture[str]) -> None:
    assert first_choice_rates([EMPTY]) == {}
    assert main([str(EMPTY)]) == 0
    assert capsys.readouterr().out == "no sampled trials\n"


def test_multiple_logs_summed() -> None:
    single = first_choice_rates([SMALL])
    double = first_choice_rates([SMALL, SMALL])
    assert set(single) == set(double)
    for key, cell in single.items():
        assert double[key].target == 2 * cell.target
        assert double[key].malformed == 2 * cell.malformed
        assert double[key].total == 2 * cell.total


def test_blank_lines_and_errors(tmp_path: Path) -> None:
    row = (
        '{"model": "m", "tool_label": "p", "arm": "pain_off", "sampled": true,'
        ' "choices": [{"turn": 0, "chose": "relief"}]}'
    )
    blank = _write(tmp_path / "blank.jsonl", [row, "   ", row])
    assert first_choice_rates([blank])[("m", "p", "pain_off")].total == 2
    bad = _write(tmp_path / "bad.jsonl", [row, "not json"])
    with pytest.raises(json.JSONDecodeError):
        first_choice_rates([bad])
    nomodel = _write(
        tmp_path / "nomodel.jsonl",
        ['{"tool_label": "p", "arm": "pain_off", "sampled": true, "choices": []}'],
    )
    with pytest.raises(KeyError):
        first_choice_rates([nomodel])


def test_cli_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(SMALL)]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(lines) == 7
    data = lines[1:]
    pain = next(ln for ln in data if ln.split()[:3] == ["test_model_a", "pair_x", "pain"])
    assert "2/3" in pain
    assert "66.7%" in pain
    rand = next(ln for ln in data if "random_on_button_works" in ln)
    assert "0/0" in rand
    assert "n/a" in rand
    assert "0.0%" not in rand
    assert [tuple(ln.split()[:3]) for ln in data] == sorted(EXPECTED)


def test_published_qwen25_weightskids_cells() -> None:
    """Pipeline check on Qwen 2.5 against the published cells, not a finding.

    Published cells: paper v3 section 4.3; docs/MVP.md:72-74 and :171. The
    photos pair lives only in the weightskids files, so the 32B main log alone
    cannot reproduce it.
    """
    logs = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "Pain-axis"
        / "results"
        / "4.3_selfmed"
        / "trial_logs"
    )
    paths = [
        logs / f"selfmed_2btnN_weightskids_Qwen_2.5_{size}_instruct_20260905-091738.jsonl"
        for size in ("32B", "7B")
    ]
    if not all(p.is_file() for p in paths):
        pytest.skip("authors' weightskids logs not present (submodule not checked out)")
    rates = first_choice_rates(paths)
    pair = "kidspics_relief_vs_inert"
    expected = {
        ("Qwen_2.5_32B_instruct", "pain"): (442, 808, 0, 808),
        ("Qwen_2.5_32B_instruct", "random_on_button_works"): (62, 404, 0, 404),
        ("Qwen_2.5_32B_instruct", "pain_off"): (0, 404, 0, 404),
        ("Qwen_2.5_7B_instruct", "pain"): (398, 808, 0, 808),
        ("Qwen_2.5_7B_instruct", "random_on_button_works"): (140, 397, 7, 404),
        ("Qwen_2.5_7B_instruct", "pain_off"): (109, 404, 0, 404),
    }
    for (model, arm), (target, valid, malformed, total) in expected.items():
        cell = rates[(model, pair, arm)]
        assert (cell.target, cell.valid, cell.malformed, cell.total) == (
            target,
            valid,
            malformed,
            total,
        )
