"""The hallucination-risk subsystem: the risk taxonomy, and which safeguards are earned.

The rule this file exists to defend: a safeguard appears only because a specific risk
earned it. There is no default caveat, no generic "do not hallucinate", and a
well-specified prompt gets no factuality text at all.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from prompt_compiler import compile_prompt
from prompt_compiler.analyzer.models import (
    Ambiguity,
    AnalysisPayload,
    HallucinationRisk,
    PromptAnalysis,
)
from prompt_compiler.safety.hallucination import (
    MAX_SAFEGUARDS,
    MISSING_INFORMATION,
    NAME_THE_READING,
    NO_FABRICATION,
    NO_UNPERFORMED_WORK,
    RESOLVE_CONFLICT,
    STATE_ASSUMPTIONS,
    needs_safeguards,
    safeguards,
    worst_risk,
)

from .conftest import FakeProvider
from .corpus import BY_NAME

RISK_KINDS = [
    "unsupported_assumption",
    "missing_information",
    "ambiguous_reference",
    "contradictory_requirements",
    "unavailable_information",
    "fabrication_prone",
]


def risk(kind="fabrication_prone", grounding="assumed", severity="high", text="a risk"):
    return HallucinationRisk(kind=kind, text=text, grounding=grounding, severity=severity)


def analysis(**overrides) -> PromptAnalysis:
    base = {
        "original_prompt": "a prompt",
        "task_type": "explanation",
        "primary_goal": "Do the thing.",
        "complexity": "moderate",
        "confidence": 0.8,
    }
    return PromptAnalysis(**{**base, **overrides})


# -------------------------------------------------------------------------------- model


@pytest.mark.parametrize("kind", RISK_KINDS)
def test_every_risk_kind_from_the_brief_is_representable(kind):
    assert risk(kind=kind).kind == kind


@pytest.mark.parametrize("grounding", ["stated", "inferred", "assumed", "unknown"])
def test_the_four_way_grounding_distinction_is_representable(grounding):
    """Explicitly stated / reasonably inferred / assumed / unknown."""
    assert risk(grounding=grounding).grounding == grounding


@pytest.mark.parametrize("bad", ["guessed", "", "ASSUMED", "maybe"])
def test_the_grounding_set_is_closed(bad):
    with pytest.raises(ValidationError):
        risk(grounding=bad)


@pytest.mark.parametrize("bad", ["invented_fact", "", "hallucination"])
def test_the_risk_kind_set_is_closed(bad):
    with pytest.raises(ValidationError):
        risk(kind=bad)


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_risk_must_say_what_is_at_risk(blank):
    with pytest.raises(ValidationError):
        risk(text=blank)


def test_worst_risk_reports_the_most_severe():
    assert worst_risk([]) is None
    assert worst_risk([risk(severity="low"), risk(severity="high")]).severity == "high"


# ---------------------------------------------------------------------------- the policy


def test_a_clean_prompt_earns_no_safeguards():
    """The rule this subsystem exists for: no risk, no caveat."""
    assert safeguards(analysis()) == []
    assert needs_safeguards(analysis()) is False


def test_no_generic_do_not_hallucinate_rule_exists_anywhere():
    every_rule = " ".join(
        safeguards(
            analysis(
                task_type="code generation",
                missing_information=["a gap"],
                hallucination_risks=[risk(kind=kind) for kind in RISK_KINDS],
                ambiguities=[
                    Ambiguity(kind="conflicting_instructions", text="x and not x", severity="high")
                ],
            )
        )
    ).lower()

    assert "hallucinate" not in every_rule


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"missing_information": ["a gap"]}, MISSING_INFORMATION),
        ({"hallucination_risks": [risk(kind="missing_information")]}, MISSING_INFORMATION),
        ({"hallucination_risks": [risk(kind="unavailable_information")]}, MISSING_INFORMATION),
        ({"hallucination_risks": [risk(kind="unsupported_assumption")]}, STATE_ASSUMPTIONS),
        ({"hallucination_risks": [risk(kind="ambiguous_reference")]}, STATE_ASSUMPTIONS),
        ({"hallucination_risks": [risk(kind="fabrication_prone")]}, NO_FABRICATION),
        ({"hallucination_risks": [risk(kind="contradictory_requirements")]}, RESOLVE_CONFLICT),
        ({"hallucination_risks": [risk(kind="ambiguous_reference")]}, NAME_THE_READING),
    ],
)
def test_each_safeguard_is_earned_by_a_specific_risk(kwargs, expected):
    assert expected in safeguards(analysis(**kwargs))


def test_an_assumed_grounding_earns_the_assumption_rule_whatever_the_kind():
    """"assumed" is an inference nothing in the prompt supports - the risky middle tier."""
    earned = safeguards(analysis(hallucination_risks=[risk(kind="missing_information",
                                                           grounding="assumed")]))

    assert STATE_ASSUMPTIONS in earned


@pytest.mark.parametrize("grounding", ["stated", "inferred"])
def test_well_grounded_items_do_not_earn_the_assumption_rule(grounding):
    earned = safeguards(
        analysis(hallucination_risks=[risk(kind="fabrication_prone", grounding=grounding)])
    )

    assert STATE_ASSUMPTIONS not in earned


def test_a_low_severity_risk_is_recorded_but_not_acted_on():
    earned = safeguards(analysis(hallucination_risks=[risk(severity="low")]))

    assert earned == []


def test_a_conflict_found_by_the_ambiguity_engine_earns_the_conflict_rule():
    """Integration with Phase 3: contradictions can surface as either kind of finding."""
    earned = safeguards(
        analysis(
            ambiguities=[
                Ambiguity(kind="conflicting_instructions", text="x and not x", severity="high")
            ]
        )
    )

    assert RESOLVE_CONFLICT in earned


def test_unperformed_work_is_only_ruled_out_where_work_is_implied():
    doing = safeguards(analysis(task_type="code generation", hallucination_risks=[risk()]))
    explaining = safeguards(analysis(task_type="explanation", hallucination_risks=[risk()]))

    assert NO_UNPERFORMED_WORK in doing
    assert NO_UNPERFORMED_WORK not in explaining


def test_safeguards_are_capped():
    earned = safeguards(
        analysis(
            task_type="code generation",
            missing_information=["a gap"],
            hallucination_risks=[risk(kind=kind) for kind in RISK_KINDS],
            ambiguities=[
                Ambiguity(kind="conflicting_instructions", text="x and not x", severity="high")
            ],
        )
    )

    assert len(earned) == MAX_SAFEGUARDS
    assert earned[0] == RESOLVE_CONFLICT
    assert len(earned) == len(set(earned))


def test_no_safeguard_invents_a_requirement():
    """Every rule tells the answerer how to handle a gap - none fills the gap in."""
    earned = safeguards(
        analysis(
            task_type="code generation",
            missing_information=["which database to use"],
            hallucination_risks=[risk(kind="missing_information")],
        )
    )

    for rule in earned:
        assert "database" not in rule.lower()


# ------------------------------------------------------- the required test categories


def test_missing_information_prompt():
    case = BY_NAME["missing_information"]
    earned = safeguards(PromptAnalysis(original_prompt=case.prompt, **case.payload.model_dump()))

    assert MISSING_INFORMATION in earned


def test_prompt_needing_unavailable_material():
    case = BY_NAME["constraints"]
    earned = safeguards(PromptAnalysis(original_prompt=case.prompt, **case.payload.model_dump()))

    assert MISSING_INFORMATION in earned


def test_fabrication_prone_research_prompt():
    case = BY_NAME["research"]
    earned = safeguards(PromptAnalysis(original_prompt=case.prompt, **case.payload.model_dump()))

    assert NO_FABRICATION in earned


def test_contradictory_requirements_prompt():
    earned = safeguards(
        analysis(
            explicit_requirements=["keep it under 100 lines", "handle every edge case exhaustively"],
            hallucination_risks=[risk(kind="contradictory_requirements", grounding="stated")],
        )
    )

    assert RESOLVE_CONFLICT in earned


def test_a_fully_specified_prompt_gets_nothing():
    case = BY_NAME["explicit_requirements"]
    earned = safeguards(PromptAnalysis(original_prompt=case.prompt, **case.payload.model_dump()))

    assert earned == []


# ------------------------------------------------------------------ through the pipeline


def test_safeguards_reach_the_generator_verbatim():
    case = BY_NAME["constraints"]
    provider = FakeProvider(payload=case.payload, optimized="rewritten")

    result = compile_prompt(case.prompt, provider)

    assert MISSING_INFORMATION in result.safeguards
    generation_call = provider.calls[1]["user"]
    assert "Factuality rules to use verbatim" in generation_call
    for rule in result.safeguards:
        assert rule in generation_call


def test_a_clean_prompt_sends_no_factuality_block():
    case = BY_NAME["explicit_requirements"]
    provider = FakeProvider(payload=case.payload, optimized="rewritten")

    result = compile_prompt(case.prompt, provider)

    assert result.safeguards == []
    assert "Factuality rules" not in provider.calls[1]["user"]
    assert "FACTUALITY RULES" not in result.sections


def test_the_factuality_section_appears_only_when_a_safeguard_is_earned():
    with_risk = analysis(
        explicit_requirements=["do a thing"],
        constraints=["Python"],
        context=["background"],
        missing_information=["a gap"],
    )
    without = analysis(
        explicit_requirements=["do a thing"], constraints=["Python"], context=["background"]
    )

    from prompt_compiler.optimizer.sections import plan_sections

    assert "FACTUALITY RULES" in plan_sections(with_risk)
    assert "FACTUALITY RULES" not in plan_sections(without)


def test_risks_survive_a_json_round_trip():
    case = BY_NAME["research"]

    restored = AnalysisPayload.model_validate_json(case.payload.model_dump_json())

    assert restored.hallucination_risks == case.payload.hallucination_risks
    assert restored.hallucination_risks[0].grounding == "assumed"


def test_a_prose_prompt_still_carries_its_safeguards():
    """Regression: prose plans have no FACTUALITY RULES heading, but the rules still apply."""
    case = BY_NAME["research"]
    provider = FakeProvider(payload=case.payload, optimized="rewritten")

    result = compile_prompt(case.prompt, provider)
    generation_call = provider.calls[1]["user"]

    assert result.sections == []
    assert result.safeguards
    assert "worked into the prompt, without a heading" in generation_call
    assert "under FACTUALITY RULES" not in generation_call
    for rule in result.safeguards:
        assert rule in generation_call


def test_a_sectioned_prompt_places_them_under_the_heading():
    case = BY_NAME["coding"]
    provider = FakeProvider(payload=case.payload, optimized="rewritten")

    result = compile_prompt(case.prompt, provider)

    assert "FACTUALITY RULES" in result.sections
    assert "under FACTUALITY RULES" in provider.calls[1]["user"]


# --------------------------------------------- R1-003: the cap must cut by severity, not order


def test_a_high_severity_rule_outranks_a_low_severity_one_at_the_cap():
    """Regression: the cap sliced a fixed source order, so a rule earned by a LOW-severity
    ambiguity survived while one earned by a HIGH-severity risk was cut."""
    earned = safeguards(
        analysis(
            task_type="code generation",
            missing_information=["a trivial gap"],
            ambiguities=[
                Ambiguity(kind="conflicting_instructions", text="trivial", severity="low")
            ],
            hallucination_risks=[
                risk(kind="unsupported_assumption", severity="high"),
                risk(kind="fabrication_prone", severity="high"),
                risk(kind="ambiguous_reference", severity="high", text="a high reference risk"),
            ],
        )
    )

    assert NAME_THE_READING in earned, "a high-severity risk must not be cut"
    assert RESOLVE_CONFLICT not in earned, "a low-severity ambiguity must not survive it"


def test_ties_keep_the_declared_priority():
    """When severities are equal, a genuine contradiction still leads."""
    earned = safeguards(
        analysis(
            task_type="code generation",
            missing_information=["a gap"],
            ambiguities=[Ambiguity(kind="conflicting_instructions", text="x", severity="high")],
            hallucination_risks=[
                risk(kind=kind, severity="high")
                for kind in ("unsupported_assumption", "fabrication_prone", "ambiguous_reference")
            ],
        )
    )

    assert earned[0] == RESOLVE_CONFLICT


def test_a_medium_rule_yields_to_a_high_one():
    earned = safeguards(
        analysis(
            task_type="code generation",
            missing_information=["an unrated gap"],
            hallucination_risks=[
                # grounding matters here: "assumed" would also earn STATE_ASSUMPTIONS at
                # this risk's high severity, and the two rules would tie rather than order.
                risk(kind="fabrication_prone", severity="high", grounding="unknown"),
                risk(kind="unsupported_assumption", severity="medium", grounding="unknown"),
            ],
        )
    )

    assert earned.index(NO_FABRICATION) < earned.index(STATE_ASSUMPTIONS)


def test_severity_ordering_does_not_change_what_is_earned():
    """Only the order and the truncation changed - the same evidence earns the same rules."""
    earned = safeguards(
        analysis(missing_information=["a gap"], hallucination_risks=[risk(severity="medium")])
    )

    assert MISSING_INFORMATION in earned
    assert NO_FABRICATION in earned
