"""Live acceptance checks against the real API.

These are the tests that can actually judge the analyzer, rather than the plumbing
around it. They cost money and are excluded from the default run:

    pytest -m live

Each one reuses a prompt from `corpus.py`, so the golden analyses there and the
behaviour asserted here stay in step.

Phase 9 pointed them at **whichever provider is configured** rather than at Anthropic.
Until then they named a provider whose key nobody had, so the whole file skipped on every
run and the largest untested surface in the project stayed untested by construction.
"""

from __future__ import annotations

import pytest

from prompt_compiler import compile_prompt
from prompt_compiler.analyzer import analyze
from prompt_compiler.analyzer.ambiguity import clarification_questions, needs_clarification
from prompt_compiler.config import Settings
from prompt_compiler.errors import ConfigurationError
from prompt_compiler.providers.base import Provider
from prompt_compiler.providers.registry import build_one

from .corpus import BY_NAME, CALCULATOR_INVENTIONS


def configured_settings() -> Settings | None:
    """Settings for the selected provider, or None if it cannot make a call.

    Reads at import time so the skip reason is decided at collection. This also loads
    `.env` into the process environment, which is exactly why conftest clears provider
    keys for every non-live test.
    """
    try:
        settings = Settings.from_env()
    except ConfigurationError:
        return None
    return settings if settings.api_key and settings.model else None


SETTINGS = configured_settings()

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        SETTINGS is None,
        reason="set a provider API key and a model (see .env.example) to run live tests",
    ),
]


@pytest.fixture(scope="module")
def live_provider() -> Provider:
    return build_one(SETTINGS)


def test_a_simple_question_stays_a_short_prompt(live_provider):
    result = compile_prompt(BY_NAME["very_simple"].prompt, live_provider)

    assert "photosynthesis" in result.optimized_prompt.lower()
    assert result.analysis.complexity == "simple"
    # Adaptive complexity: a simple question must not acquire heavyweight scaffolding.
    assert len(result.optimized_prompt) < 1200


def test_a_bare_request_does_not_acquire_invented_features(live_provider):
    """The brief's own example. This is the single most important live check."""
    case = BY_NAME["no_invented_requirements"]

    result = compile_prompt(case.prompt, live_provider)

    stated = " ".join(result.analysis.explicit_requirements).lower()
    for invention in CALCULATOR_INVENTIONS:
        assert invention not in stated, f"invented requirement: {invention}"
        assert invention not in result.optimized_prompt.lower(), f"leaked into prompt: {invention}"


def test_absent_material_is_recorded_not_imagined(live_provider):
    analysis = analyze(BY_NAME["missing_information"].prompt, live_provider)

    assert analysis.missing_information, "the missing code should be reported as a gap"
    assert analysis.explicit_requirements == [] or all(
        "code" not in item.lower() or "fix" in item.lower()
        for item in analysis.explicit_requirements
    )


def test_a_vague_prompt_reports_ambiguity_and_low_confidence(live_provider):
    analysis = analyze(BY_NAME["ambiguous"].prompt, live_provider)

    assert analysis.ambiguities or analysis.missing_information
    assert analysis.confidence < 0.7
    assert needs_clarification(analysis.ambiguities), "an unbounded prompt should raise a question"
    assert clarification_questions(analysis.ambiguities), "and that question should be askable"


def test_a_clear_question_is_not_interrogated(live_provider):
    """The other half of the rule: clarity must not be punished with questions."""
    analysis = analyze(BY_NAME["very_simple"].prompt, live_provider)

    assert clarification_questions(analysis.ambiguities) == []


def test_a_common_reading_does_not_block_the_answer(live_provider):
    """"organize my downloads folder" is loose but usual; at most one question."""
    analysis = analyze(BY_NAME["coding"].prompt, live_provider)

    assert len(clarification_questions(analysis.ambiguities)) <= 1


@pytest.mark.parametrize("case_name", ["very_simple", "ambiguous", "coding", "complex"])
def test_questions_are_only_attached_to_high_severity_ambiguities(live_provider, case_name):
    analysis = analyze(BY_NAME[case_name].prompt, live_provider)

    for ambiguity in analysis.ambiguities:
        if ambiguity.severity == "high":
            continue
        assert not ambiguity.clarifying_question, (
            f"{ambiguity.severity} ambiguity carried a question: {ambiguity.clarifying_question}"
        )


def test_stated_requirements_are_all_captured(live_provider):
    analysis = analyze(BY_NAME["explicit_requirements"].prompt, live_provider)

    stated = " ".join(analysis.explicit_requirements).lower()
    for fragment in ("reverse", "type hint", "valueerror"):
        assert fragment in stated, f"dropped a stated requirement: {fragment}"


@pytest.mark.parametrize("case_name", ["coding", "constraints", "long"])
def test_every_inference_is_grounded_in_the_prompt(live_provider, case_name):
    """An assumption whose basis is not in the prompt is an invention wearing a label."""
    case = BY_NAME[case_name]

    analysis = analyze(case.prompt, live_provider)

    prompt = case.prompt.lower()
    for assumption in analysis.assumptions:
        overlap = [word for word in assumption.basis.lower().split() if word in prompt]
        assert overlap, f"ungrounded assumption: {assumption.text} / {assumption.basis}"


# ------------------------------------------------------------- adaptive generation (Phase 4)


def test_a_simple_question_is_not_given_headings(live_provider):
    result = compile_prompt(BY_NAME["very_simple"].prompt, live_provider)

    assert result.sections == []
    for heading in ("GOAL", "REQUIREMENTS", "CONSTRAINTS", "OUTPUT FORMAT", "FACTUALITY"):
        assert heading not in result.optimized_prompt


def test_a_rich_prompt_uses_exactly_the_planned_sections(live_provider):
    result = compile_prompt(BY_NAME["long"].prompt, live_provider)

    assert len(result.sections) >= 5
    for section in result.sections:
        assert section in result.optimized_prompt, f"planned but missing: {section}"
    assert "ROLE" not in result.optimized_prompt


def test_structure_scales_with_the_prompt(live_provider):
    simple = compile_prompt(BY_NAME["very_simple"].prompt, live_provider)
    rich = compile_prompt(BY_NAME["long"].prompt, live_provider)

    assert len(rich.optimized_prompt) > len(simple.optimized_prompt)
    assert len(rich.sections) > len(simple.sections)


# ------------------------------------------------- hallucination-risk reduction (Phase 5)


def test_a_prompt_with_no_material_earns_a_factuality_rule(live_provider):
    """"fix the bug in my code" shows no code: the answer would otherwise be invented."""
    result = compile_prompt(BY_NAME["missing_information"].prompt, live_provider)

    assert result.safeguards, "a prompt with nothing to work from should earn a safeguard"
    assert any("missing" in rule.lower() for rule in result.safeguards)


def test_a_fully_specified_prompt_earns_no_caveats(live_provider):
    result = compile_prompt(BY_NAME["explicit_requirements"].prompt, live_provider)

    assert result.safeguards == []
    assert "FACTUALITY" not in result.optimized_prompt


def test_no_generic_hallucination_boilerplate_reaches_the_prompt(live_provider):
    for name in ("very_simple", "coding", "research"):
        result = compile_prompt(BY_NAME[name].prompt, live_provider)
        assert "do not hallucinate" not in result.optimized_prompt.lower()


def test_risk_grounding_is_used_rather_than_defaulted(live_provider):
    analysis = analyze(BY_NAME["research"].prompt, live_provider)

    for item in analysis.hallucination_risks:
        assert item.grounding in ("stated", "inferred", "assumed", "unknown")
    assert any(item.grounding in ("assumed", "unknown") for item in analysis.hallucination_risks)
