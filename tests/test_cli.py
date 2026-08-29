from __future__ import annotations

import json

import pytest

from prompt_compiler.cli import main
from prompt_compiler.errors import MissingCredentialsError, ProviderError

from .conftest import FakeProvider


def test_prints_optimized_prompt(capsys, provider):
    exit_code = main(["organize my downloads folder"], provider=provider)
    out = capsys.readouterr().out

    assert exit_code == 0
    assert out.strip() == "Write a Python script that ..."


def test_show_analysis_adds_a_block(capsys, provider):
    main(["organize my downloads folder", "--show-analysis"], provider=provider)
    out = capsys.readouterr().out

    assert "ANALYSIS" in out
    assert "code generation" in out
    assert "which categories the user wants" in out
    assert "Confidence:  0.72" in out
    assert "[from: my downloads folder]" in out
    assert out.rstrip().endswith("Write a Python script that ...")


def test_json_output_is_machine_readable(capsys, provider):
    main(["organize my downloads folder", "--json"], provider=provider)
    payload = json.loads(capsys.readouterr().out)

    assert payload["analysis"]["original_prompt"] == "organize my downloads folder"
    assert payload["optimized_prompt"] == "Write a Python script that ..."
    assert payload["analysis"]["task_type"] == "code generation"


@pytest.mark.parametrize(
    "error",
    [
        MissingCredentialsError("No Anthropic API key found. Set ANTHROPIC_API_KEY ..."),
        ProviderError("Could not reach the Anthropic API."),
    ],
)
def test_expected_failures_report_cleanly(capsys, payload, error):
    failing = FakeProvider(payload=payload, error=error)

    exit_code = main(["explain recursion"], provider=failing)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "Traceback" not in captured.err


def test_empty_prompt_reports_cleanly(capsys, provider):
    exit_code = main(["   "], provider=provider)

    assert exit_code == 1
    assert "empty" in capsys.readouterr().err


def test_missing_argument_exits_with_usage_error(provider):
    with pytest.raises(SystemExit) as excinfo:
        main([], provider=provider)

    assert excinfo.value.code == 2


def test_non_ascii_output_survives_a_legacy_console(monkeypatch, payload):
    """Regression: a cp1252 console must not crash on model output.

    capsys captures text without encoding it, so this stands in for a real Windows
    console by giving the CLI a genuine cp1252 text stream.
    """
    import io
    import sys

    buffer = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(buffer, encoding="cp1252"))
    failing_on_cp1252 = FakeProvider(payload=payload, optimized="An em dash — and a quote ’.")

    exit_code = main(["explain recursion", "--show-analysis"], provider=failing_on_cp1252)
    sys.stdout.flush()

    assert exit_code == 0
    assert "em dash" in buffer.getvalue().decode("utf-8")


def test_bad_configuration_is_reported_cleanly(capsys, monkeypatch):
    """Regression: a typo in an env var used to produce a traceback."""
    monkeypatch.setenv("PROMPT_COMPILER_MAX_TOKENS", "lots")

    exit_code = main(["explain recursion"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err.startswith("error: ")
    assert "PROMPT_COMPILER_MAX_TOKENS" in captured.err
    assert "Traceback" not in captured.err


def _render(case_name: str) -> str:
    from .corpus import BY_NAME

    case = BY_NAME[case_name]
    provider = FakeProvider(payload=case.payload, optimized=case.optimized)
    exit_code = main([case.prompt, "--show-analysis"], provider=provider)
    assert exit_code == 0
    return case, provider


def test_ambiguities_and_questions_are_rendered(capsys):
    _render("ambiguous")
    out = capsys.readouterr().out

    assert "Ambiguity:   HIGH" in out
    assert "Ambiguities:" in out
    assert "[HIGH] unclear_scope:" in out
    assert "[MEDIUM] unclear_output_format:" in out
    assert "Worth asking before answering:" in out
    assert "1. What should the app do, and for whom?" in out


def test_a_clear_prompt_renders_no_ambiguity_section(capsys):
    _render("very_simple")
    out = capsys.readouterr().out

    assert "Ambiguity:" not in out
    assert "Ambiguities:" not in out
    assert "Worth asking" not in out


def test_medium_ambiguities_are_shown_but_not_asked(capsys):
    _render("coding")
    out = capsys.readouterr().out

    assert "Ambiguity:   MEDIUM" in out
    assert "[MEDIUM] vague_terminology:" in out
    assert "Worth asking" not in out


def test_the_chosen_structure_is_reported(capsys):
    _render("coding")
    out = capsys.readouterr().out

    assert "Structure:   GOAL, REQUIREMENTS, CONSTRAINTS, OUTPUT FORMAT, FACTUALITY RULES" in out


def test_a_prose_result_says_so(capsys):
    _render("very_simple")
    out = capsys.readouterr().out

    assert "Structure:   none - write a short prompt in plain prose" in out


def test_json_output_carries_the_structure(capsys, provider):
    main(["organize my downloads folder", "--json"], provider=provider)
    payload = json.loads(capsys.readouterr().out)

    assert payload["sections"] == ["GOAL", "CONTEXT", "REQUIREMENTS", "CONSTRAINTS",
                                   "OUTPUT FORMAT", "FACTUALITY RULES"]
    assert payload["analysis"]["ambiguities"][0]["severity"] == "medium"


def test_risks_and_safeguards_are_rendered(capsys):
    _render("constraints")
    out = capsys.readouterr().out

    assert "Hallucination risks:" in out
    assert "[HIGH] unavailable_information (unknown):" in out
    assert "Factuality rules added:" in out
    assert "say what is missing instead of filling it in" in out


def test_a_clean_prompt_shows_no_factuality_noise(capsys):
    _render("explicit_requirements")
    out = capsys.readouterr().out

    assert "Hallucination risks:" not in out
    assert "Factuality rules added:" not in out


def test_quality_is_reported_as_a_heuristic(capsys):
    """The label travels with the number. A score presented bare would read as a
    measurement, which is exactly what it is not."""
    _render("coding")
    out = capsys.readouterr().out

    assert "Quality:" in out
    assert "heuristic estimates, not measurements" in out


def test_the_two_score_subjects_are_labelled_separately(capsys):
    """The user must be able to tell "your request was vague" from "the rewrite dropped
    something" - the whole reason there is no single quality figure."""
    _render("coding")
    out = capsys.readouterr().out

    assert "Quality of the prompt you wrote (heuristic):" in out
    assert "Quality of the rewrite (heuristic):" in out
    assert "clarity" in out
    assert "requirement_coverage" in out


def test_each_score_is_shown_with_the_evidence_behind_it(capsys):
    _render("ambiguous")
    out = capsys.readouterr().out

    assert "severity ambiguities" in out
    assert "gaps blocking the answer" in out
    assert "for a complex request" in out


def test_json_output_carries_the_quality_report(capsys, provider):
    main(["organize my downloads folder", "--json"], provider=provider)
    payload = json.loads(capsys.readouterr().out)

    assert payload["quality"]["heuristic"] is True
    assert {item["name"] for item in payload["quality"]["dimensions"]} == {
        "clarity",
        "specificity",
        "completeness",
        "requirement_coverage",
        "risk_coverage",
        "token_efficiency",
    }
    assert all(item["basis"] for item in payload["quality"]["dimensions"])
