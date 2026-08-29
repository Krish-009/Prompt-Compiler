"""The section policy: how much structure a prompt earns.

This is the deterministic half of adaptive generation. The model writes the prose; the
rules here decide whether it writes prose at all.
"""

from __future__ import annotations

import pytest

from prompt_compiler import compile_prompt
from prompt_compiler.analyzer.models import Ambiguity, HallucinationRisk, PromptAnalysis
from prompt_compiler.optimizer.sections import (
    MIN_CONTENT_SECTIONS,
    SECTION_ORDER,
    describe_plan,
    plan_sections,
)

from .conftest import FakeProvider
from .corpus import BY_NAME, CASES


def analysis(**overrides) -> PromptAnalysis:
    base = {
        "original_prompt": "a prompt",
        "task_type": "code generation",
        "primary_goal": "Do the thing.",
        "complexity": "moderate",
        "confidence": 0.8,
    }
    return PromptAnalysis(**{**base, **overrides})


def rich(**overrides) -> PromptAnalysis:
    """An analysis that already clears the guard and the floor."""
    base = {
        "explicit_requirements": ["do the thing"],
        "constraints": ["Python"],
        "context": ["some background"],
    }
    return analysis(**{**base, **overrides})


# ---------------------------------------------------------------------------- the floor


def test_an_empty_analysis_gets_no_structure():
    assert plan_sections(analysis()) == []


def test_a_prompt_that_states_nothing_stays_prose_however_many_gaps_it_has():
    """Regression: gaps alone used to earn OUTPUT FORMAT and FACTUALITY RULES, so the
    emptiest prompts came out the most heavily scaffolded."""
    plan = plan_sections(
        analysis(
            complexity="complex",
            expected_output="something",
            missing_information=["everything", "the platform", "the language"],
            hallucination_risks=[
                HallucinationRisk(
                    kind="unavailable_information",
                    text="nothing was shown",
                    grounding="unknown",
                    severity="high",
                )
            ],
            ambiguities=[
                Ambiguity(kind="unclear_scope", text="unbounded", severity="high"),
                Ambiguity(kind="unclear_output_format", text="unknown shape", severity="high"),
            ],
        )
    )

    assert plan == []


def test_below_the_floor_stays_prose():
    plan = plan_sections(analysis(explicit_requirements=["one thing"]))

    assert plan == []


def test_at_the_floor_the_prompt_gets_structure():
    plan = plan_sections(
        analysis(
            explicit_requirements=["one thing"],
            constraints=["Python"],
            missing_information=["a gap"],
        )
    )

    assert len(plan) == MIN_CONTENT_SECTIONS + 1  # the content sections, plus GOAL
    assert plan[0] == "GOAL"


# -------------------------------------------------------------------------- the triggers


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("context", ["background"], "CONTEXT"),
        ("explicit_requirements", ["a stated requirement"], "REQUIREMENTS"),
        ("constraints", ["Python"], "CONSTRAINTS"),
        ("expected_output", "a JSON object", "OUTPUT FORMAT"),
        ("missing_information", ["a gap"], "FACTUALITY RULES"),
        (
            "hallucination_risks",
            [HallucinationRisk(kind="fabrication_prone", text="a risk",
                              grounding="assumed", severity="medium")],
            "FACTUALITY RULES",
        ),
    ],
)
def test_each_section_is_earned_by_its_own_evidence(field, value, expected):
    without = plan_sections(rich())
    with_evidence = plan_sections(rich(**{field: value}))

    assert expected in with_evidence
    if expected not in ("CONTEXT", "REQUIREMENTS", "CONSTRAINTS"):
        assert expected not in without


def test_an_unclear_output_format_earns_the_output_section_without_a_stated_one():
    plan = plan_sections(
        rich(
            ambiguities=[
                Ambiguity(kind="unclear_output_format", text="shape unknown", severity="medium")
            ]
        )
    )

    assert "OUTPUT FORMAT" in plan


def test_process_needs_both_complexity_and_something_to_order():
    assert "PROCESS" in plan_sections(rich(complexity="complex"))
    assert "PROCESS" not in plan_sections(rich(complexity="moderate"))
    # Complex but nothing stated to order: the guard sends it to prose entirely.
    assert plan_sections(analysis(complexity="complex", constraints=["Python"])) == []


def test_role_is_never_planned():
    """An unearned role line is the boilerplate this project exists to avoid."""
    for case in CASES:
        plan = plan_sections(PromptAnalysis(original_prompt=case.prompt, **case.payload.model_dump()))
        assert "ROLE" not in plan


# ---------------------------------------------------------------------------- the shape


def test_sections_come_out_in_a_fixed_order_without_duplicates():
    plan = plan_sections(
        rich(
            complexity="complex",
            expected_output="a script",
            missing_information=["a gap"],
        )
    )

    assert plan == [section for section in SECTION_ORDER if section in plan]
    assert len(plan) == len(set(plan))
    assert plan[0] == "GOAL"


def test_describe_plan_names_the_shape():
    assert describe_plan([]).startswith("none")
    assert describe_plan(["GOAL", "REQUIREMENTS"]) == "GOAL, REQUIREMENTS"


# -------------------------------------------------------------------- against the corpus


@pytest.mark.parametrize(
    "case_name", ["very_simple", "very_short", "missing_information", "ambiguous", "research"]
)
def test_prompts_with_little_stated_content_stay_prose(case_name):
    case = BY_NAME[case_name]

    plan = plan_sections(PromptAnalysis(original_prompt=case.prompt, **case.payload.model_dump()))

    assert plan == [], f"{case_name} should not be scaffolded"


@pytest.mark.parametrize("case_name", ["coding", "explicit_requirements", "complex", "long"])
def test_prompts_with_real_content_get_structure(case_name):
    case = BY_NAME[case_name]

    plan = plan_sections(PromptAnalysis(original_prompt=case.prompt, **case.payload.model_dump()))

    assert len(plan) >= MIN_CONTENT_SECTIONS + 1
    assert plan[0] == "GOAL"


def test_the_richest_prompt_gets_the_most_structure():
    plans = {
        case.name: plan_sections(
            PromptAnalysis(original_prompt=case.prompt, **case.payload.model_dump())
        )
        for case in CASES
    }

    assert len(plans["long"]) == max(len(plan) for plan in plans.values())
    assert len(plans["long"]) > len(plans["coding"]) > len(plans["very_simple"])


# ------------------------------------------------------------------------ through the pipeline


def test_the_plan_reaches_the_generator_and_the_result():
    case = BY_NAME["coding"]
    provider = FakeProvider(payload=case.payload, optimized="rewritten")

    result = compile_prompt(case.prompt, provider)

    assert result.sections == ["GOAL", "REQUIREMENTS", "CONSTRAINTS", "OUTPUT FORMAT", "FACTUALITY RULES"]
    assert "Structure to use: GOAL, REQUIREMENTS" in provider.calls[1]["user"]


def test_a_prose_plan_tells_the_generator_to_write_prose():
    case = BY_NAME["very_simple"]
    provider = FakeProvider(payload=case.payload, optimized="rewritten")

    result = compile_prompt(case.prompt, provider)

    assert result.sections == []
    assert "Structure to use: none - write a short prompt in plain prose" in provider.calls[1]["user"]


# ------------------------------------------------ integration with the ambiguity engine


def test_a_high_severity_ambiguity_earns_factuality_rules():
    """Regression: Phase 3 detected that the answer could go materially wrong, and
    Phase 4 discarded the signal - a prompt with conflicting instructions got no rule
    about which reading to take."""
    plan = plan_sections(
        rich(
            ambiguities=[
                Ambiguity(kind="conflicting_instructions", text="x and not x", severity="high")
            ]
        )
    )

    assert "FACTUALITY RULES" in plan


@pytest.mark.parametrize("severity", ["medium", "low"])
def test_a_softer_ambiguity_does_not_earn_factuality_rules_on_its_own(severity):
    plan = plan_sections(
        rich(ambiguities=[Ambiguity(kind="vague_terminology", text="loose", severity=severity)])
    )

    assert "FACTUALITY RULES" not in plan


def test_the_generator_will_not_silently_assume_prose():
    """Regression: an omitted plan used to mean "prose", hiding a forgotten plan."""
    from prompt_compiler.optimizer.generator import generate

    case = BY_NAME["coding"]
    provider = FakeProvider(payload=case.payload, optimized="x")
    analysis_obj = PromptAnalysis(original_prompt=case.prompt, **case.payload.model_dump())

    with pytest.raises(TypeError):
        generate(case.prompt, analysis_obj, provider)


def test_sections_survive_a_json_round_trip():
    """Section names contain spaces; they cross a JSON boundary in --json output."""
    from prompt_compiler.models import CompiledPrompt

    case = BY_NAME["long"]
    provider = FakeProvider(payload=case.payload, optimized="rewritten")
    result = compile_prompt(case.prompt, provider)

    restored = CompiledPrompt.model_validate_json(result.model_dump_json())

    assert restored.sections == result.sections
    assert "OUTPUT FORMAT" in restored.sections
