"""The ambiguity engine: the taxonomy, and the policy on what is worth asking.

Detection is the model's judgement and cannot be tested offline. The policy built on top
of it is ordinary code, and it is the part that enforces "do not ask unnecessary
questions" - so it is tested exhaustively here.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from prompt_compiler.analyzer.ambiguity import (
    MAX_QUESTIONS,
    by_severity,
    clarification_questions,
    needs_clarification,
    worst_severity,
)
from prompt_compiler.analyzer.models import Ambiguity, AnalysisPayload

from .corpus import BY_NAME, CASES


def amb(severity: str, question: str = "", kind: str = "unclear_scope") -> Ambiguity:
    return Ambiguity(
        kind=kind, text=f"a {severity} ambiguity", severity=severity, clarifying_question=question
    )


# ------------------------------------------------------------------------------- model


@pytest.mark.parametrize(
    "kind",
    [
        "vague_terminology",
        "undefined_terminology",
        "missing_requirement",
        "missing_technical_constraint",
        "conflicting_instructions",
        "unclear_scope",
        "unclear_output_format",
        "unclear_audience",
    ],
)
def test_every_kind_from_the_brief_is_representable(kind):
    assert Ambiguity(kind=kind, text="something", severity="low").kind == kind


@pytest.mark.parametrize("bad", ["confusing", "", "VAGUE_TERMINOLOGY"])
def test_the_kind_set_is_closed(bad):
    with pytest.raises(ValidationError):
        Ambiguity(kind=bad, text="something", severity="low")


@pytest.mark.parametrize("bad", ["HIGH", "critical", "", "none"])
def test_the_severity_set_is_closed(bad):
    with pytest.raises(ValidationError):
        Ambiguity(kind="unclear_scope", text="something", severity=bad)


@pytest.mark.parametrize("blank", ["", "   "])
def test_an_ambiguity_must_say_what_is_ambiguous(blank):
    with pytest.raises(ValidationError):
        Ambiguity(kind="unclear_scope", text=blank, severity="high")


def test_a_clarifying_question_is_optional():
    assert Ambiguity(kind="unclear_scope", text="x", severity="low").clarifying_question == ""


# ------------------------------------------------------------------------------ policy


def test_worst_severity_of_nothing_is_nothing():
    assert worst_severity([]) is None


@pytest.mark.parametrize(
    ("levels", "expected"),
    [
        (["low"], "low"),
        (["low", "medium"], "medium"),
        (["medium", "high", "low"], "high"),
        (["low", "low", "low"], "low"),
    ],
)
def test_worst_severity_reports_the_most_severe_present(levels, expected):
    assert worst_severity([amb(level) for level in levels]) == expected


def test_ordering_is_worst_first_and_stable_within_a_level():
    first_medium = amb("medium")
    second_medium = amb("medium")
    items = [amb("low"), first_medium, amb("high"), second_medium]

    ordered = by_severity(items)

    assert [item.severity for item in ordered] == ["high", "medium", "medium", "low"]
    assert ordered[1] is first_medium and ordered[2] is second_medium


@pytest.mark.parametrize(
    ("levels", "expected"),
    [([], False), (["low"], False), (["low", "medium"], False), (["medium", "high"], True)],
)
def test_clarification_is_needed_only_for_high_severity(levels, expected):
    assert needs_clarification([amb(level) for level in levels]) is expected


def test_only_high_severity_ambiguities_earn_a_question():
    """The core rule: a prompt with loose edges is not a prompt worth interrupting."""
    items = [
        amb("low", "should this be asked?"),
        amb("medium", "and this?"),
        amb("high", "what should the app do?"),
    ]

    assert clarification_questions(items) == ["what should the app do?"]


def test_no_high_severity_ambiguity_means_no_questions():
    assert clarification_questions([amb("low", "q"), amb("medium", "q")]) == []


def test_a_high_severity_ambiguity_without_a_question_invents_none():
    """An unanswered ambiguity is still reported; a question is never fabricated for it."""
    items = [amb("high"), amb("high", "the real question")]

    assert clarification_questions(items) == ["the real question"]


def test_questions_are_capped():
    items = [amb("high", f"question {n}") for n in range(10)]

    questions = clarification_questions(items)

    assert len(questions) == MAX_QUESTIONS
    assert questions == [f"question {n}" for n in range(MAX_QUESTIONS)]


def test_the_cap_is_configurable_per_call():
    items = [amb("high", f"question {n}") for n in range(5)]

    assert clarification_questions(items, limit=1) == ["question 0"]


# ------------------------------------------------------------------- against the corpus


def test_a_clear_question_is_never_interrogated():
    """The brief's rule: "What is photosynthesis?" needs no clarification at all."""
    case = BY_NAME["very_simple"]

    assert case.payload.ambiguities == []
    assert clarification_questions(case.payload.ambiguities) == []
    assert needs_clarification(case.payload.ambiguities) is False


def test_a_wide_open_prompt_does_get_asked():
    case = BY_NAME["ambiguous"]

    questions = clarification_questions(case.payload.ambiguities)

    assert needs_clarification(case.payload.ambiguities)
    assert 1 <= len(questions) <= MAX_QUESTIONS


def test_a_common_reading_is_not_treated_as_a_blocker():
    """"organize my downloads folder" has a usual meaning; proceed, do not interrogate."""
    case = BY_NAME["coding"]

    assert case.payload.ambiguities, "the vagueness is still recorded"
    assert clarification_questions(case.payload.ambiguities) == []


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_no_case_asks_more_than_the_cap(case):
    assert len(clarification_questions(case.payload.ambiguities)) <= MAX_QUESTIONS


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_questions_are_only_attached_to_high_severity_items(case):
    for ambiguity in case.payload.ambiguities:
        if ambiguity.severity != "high":
            assert not ambiguity.clarifying_question


def test_ambiguities_survive_a_json_round_trip():
    case = BY_NAME["ambiguous"]

    restored = AnalysisPayload.model_validate_json(case.payload.model_dump_json())

    assert restored.ambiguities == case.payload.ambiguities
    assert clarification_questions(restored.ambiguities) == clarification_questions(
        case.payload.ambiguities
    )
