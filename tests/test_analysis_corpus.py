"""Every prompt category, run through the pipeline against its golden analysis.

Two things are being checked, and they are different:

* **Contract** - the pipeline carries any well-formed analysis through without altering
  it, whatever the category. That is testable here, deterministically.
* **Judgement** - whether a real model classifies information correctly. That is not
  testable without the API; the golden analyses document what correct looks like, and
  the live tests reuse the same prompts.
"""

from __future__ import annotations

import pytest

from prompt_compiler import compile_prompt
from prompt_compiler.analyzer.models import AnalysisPayload, PromptAnalysis
from prompt_compiler.cli import _format_analysis

from .conftest import FakeProvider
from .corpus import BY_NAME, CALCULATOR_INVENTIONS, CASES

CASE_IDS = [case.name for case in CASES]


def _compile(case):
    provider = FakeProvider(payload=case.payload, optimized=case.optimized)
    return compile_prompt(case.prompt, provider), provider


# --------------------------------------------------------------------------- contract


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_every_category_compiles(case):
    result, _ = _compile(case)

    assert result.original_prompt == case.prompt
    assert result.optimized_prompt == case.optimized
    assert result.analysis.task_type == case.payload.task_type


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_the_pipeline_adds_nothing_to_the_analysis(case):
    """Intent preservation, mechanically: our code is a courier, not an author."""
    result, _ = _compile(case)

    carried = result.analysis.model_dump(exclude={"original_prompt"})
    assert carried == case.payload.model_dump()


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_the_prompt_reaches_both_calls_exactly_once_as_data(case):
    _, provider = _compile(case)

    assert len(provider.calls) == 2
    for call in provider.calls:
        assert f"<prompt>\n{case.prompt}\n</prompt>" in call["user"]
        assert call["user"].count("<prompt>") == 1
    # The prompt travels in the <prompt> block only; the analysis must not carry a copy.
    # (Counting occurrences of the prompt itself would misfire on a one-word prompt that
    # the analysis legitimately quotes back, such as "recursion".)
    assert "original_prompt" not in provider.calls[1]["user"]


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_every_category_renders(case):
    result, _ = _compile(case)

    rendered = _format_analysis(result)

    assert case.payload.task_type in rendered
    assert "\n\n" not in rendered, "blank lines mean an empty list leaked into the output"
    for line in rendered.splitlines():
        assert line.strip(), "no blank or whitespace-only lines"


# ------------------------------------------------------------------- corpus invariants


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_gaps_are_never_also_requirements(case):
    """A gap that has been promoted to a requirement is an invented requirement."""
    requirements = {item.lower() for item in case.payload.explicit_requirements}
    gaps = {item.lower() for item in case.payload.missing_information}

    assert requirements.isdisjoint(gaps)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_every_inference_is_grounded(case):
    """An assumption whose basis is not in the prompt is an invention wearing a label."""
    prompt = case.prompt.lower()

    for assumption in case.payload.assumptions:
        assert assumption.basis.lower() in prompt


# ------------------------------------------------------------ category-specific checks


def test_a_simple_question_stays_simple():
    case = BY_NAME["very_simple"]

    assert case.payload.complexity == "simple"
    assert case.payload.missing_information == []
    assert case.payload.ambiguities == []
    assert case.payload.assumptions == [], "a clear question needs no assumptions"


def test_an_ambiguous_prompt_records_gaps_rather_than_filling_them():
    case = BY_NAME["ambiguous"]

    assert case.payload.ambiguities
    assert case.payload.missing_information
    assert case.payload.explicit_requirements == []
    assert case.payload.confidence < 0.5


def test_a_prompt_with_no_material_records_the_hallucination_risk():
    for name in ("missing_information", "constraints"):
        payload = BY_NAME[name].payload
        assert payload.missing_information
        assert payload.hallucination_risks


def test_stated_requirements_are_captured_verbatim_enough():
    case = BY_NAME["explicit_requirements"]

    assert len(case.payload.explicit_requirements) == 3
    assert case.payload.missing_information == []
    for fragment in ("type hints", "ValueError"):
        assert any(fragment in item for item in case.payload.explicit_requirements)


def test_stated_constraints_and_output_format_are_captured():
    constraints = BY_NAME["constraints"].payload
    assert "exactly three bullet points" in constraints.constraints

    output = BY_NAME["expected_output_format"].payload
    assert output.expected_output
    assert "JSON output" in output.constraints


def test_a_long_prompt_keeps_its_context_and_constraints():
    case = BY_NAME["long"]

    assert case.payload.complexity == "complex"
    assert len(case.payload.context) >= 3
    assert "Python 3.11" in case.payload.constraints
    assert len(case.payload.explicit_requirements) >= 4


def test_a_bare_request_does_not_acquire_features():
    """The brief's own example: "Make a Python calculator." must not grow a GUI."""
    case = BY_NAME["no_invented_requirements"]
    stated = " ".join(case.payload.explicit_requirements + case.payload.constraints).lower()

    for invention in CALCULATOR_INVENTIONS:
        assert invention not in stated

    # What is unknown is recorded as unknown, not decided for the user.
    assert any("operations" in gap for gap in case.payload.missing_information)


def test_uncertain_prompts_carry_lower_confidence_than_clear_ones():
    clear = BY_NAME["explicit_requirements"].payload.confidence
    for vague in ("ambiguous", "very_short", "missing_information"):
        assert BY_NAME[vague].payload.confidence < clear


# ------------------------------------------------------------------------ invalid input


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t ", " "])
def test_blank_prompts_are_rejected_before_any_call(prompt):
    from prompt_compiler.errors import InvalidInputError

    provider = FakeProvider(payload=BY_NAME["very_simple"].payload)

    with pytest.raises(InvalidInputError):
        compile_prompt(prompt, provider)

    assert provider.calls == []


def test_an_analysis_cannot_be_bound_to_an_empty_prompt():
    from pydantic import ValidationError

    payload = BY_NAME["very_simple"].payload

    with pytest.raises(ValidationError):
        PromptAnalysis(original_prompt="   ", **payload.model_dump())


def test_a_payload_survives_a_json_round_trip():
    """The analysis crosses a JSON boundary twice: from the model, and into --json."""
    for case in CASES:
        restored = AnalysisPayload.model_validate_json(case.payload.model_dump_json())
        assert restored == case.payload
