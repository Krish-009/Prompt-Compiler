"""Post-generation validation (R1-002).

Before this existed, nothing checked that the rewritten prompt actually contained the
requirements the analysis found: the guard inside `tighten()` compares generation's output
against its own tightened version, so an item the model never wrote is missing from both
sides and the guard passes trivially.

Two properties matter here and pull against each other, so both are tested hard: a
genuinely dropped requirement must be caught, and a legitimately reworded one must not be.
"""

from __future__ import annotations

import pytest

from prompt_compiler import compile_prompt
from prompt_compiler.analyzer.models import AnalysisPayload
from prompt_compiler.optimizer.generator import GeneratedPrompt
from prompt_compiler.providers.base import Provider
from prompt_compiler.validation import PRESENCE_THRESHOLD, is_present, unverified

from .corpus import BY_NAME

REQUIREMENTS = [
    "reverse the input string",
    "include type hints",
    "raise ValueError on non-string input",
    "support unicode input",
]

PAYLOAD = AnalysisPayload(
    task_type="code generation",
    primary_goal="Write a Python function that reverses a string.",
    explicit_requirements=REQUIREMENTS,
    constraints=["Python"],
    complexity="moderate",
    confidence=0.9,
)

COMPLETE = (
    "GOAL\nWrite a Python function that reverses a string.\n\n"
    "REQUIREMENTS\n- reverse the input string\n- include type hints\n"
    "- raise ValueError on non-string input\n- support unicode input\n\n"
    "CONSTRAINTS\n- Python"
)

DROPS_UNICODE = (
    "GOAL\nWrite a Python function that reverses a string.\n\n"
    "REQUIREMENTS\n- reverse the input string\n- include type hints\n"
    "- raise ValueError on non-string input\n\nCONSTRAINTS\n- Python"
)

REWORDED = (
    "GOAL\nWrite a Python function that reverses a given string.\n\n"
    "REQUIREMENTS\n- The function reverses whatever string it is handed\n"
    "- Type hints are included on every parameter and on the return\n"
    "- A ValueError is raised when the input is not a string\n"
    "- Unicode inputs, including combining characters, are supported\n\n"
    "CONSTRAINTS\n- Written in Python"
)


class Generates(Provider):
    """A provider whose generation step returns exactly the text it was given."""

    name = "fake"
    model = "fake-model"

    def __init__(self, text: str, payload: AnalysisPayload = PAYLOAD) -> None:
        self._text = text
        self._payload = payload
        self.calls: list[type] = []

    def structured(self, *, system: str, user: str, schema):
        self.calls.append(schema)
        if schema is AnalysisPayload:
            return self._payload
        return GeneratedPrompt(optimized_prompt=self._text)


# ------------------------------------------------------------------------ the check itself


def test_a_verbatim_item_is_present():
    assert is_present("- include type hints", "include type hints")


def test_a_reworded_item_is_still_present():
    """The analysis and the rewrite come from different calls; paraphrase is expected."""
    assert is_present("Type hints are included on every parameter", "include type hints")
    assert is_present("reverses whatever string it is handed", "reverse the input string")


def test_inflection_does_not_hide_an_item():
    assert is_present("the function reverses the string", "reverse the string")


def test_an_absent_item_is_absent():
    assert not is_present("reverses a string with type hints", "support unicode input")


def test_an_item_of_only_stopwords_is_not_flagged():
    """Nothing identifying to look for means nothing can be proven missing."""
    assert is_present("anything at all", "of the and")


def test_the_threshold_is_a_fraction_not_all_or_nothing():
    """One dropped adjective out of four content words must not raise an alarm."""
    assert 0 < PRESENCE_THRESHOLD < 1
    assert is_present(
        "write a summary of the quarterly revenue report",
        "write a short summary of the quarterly revenue report",
    )


def test_unverified_reports_in_order_and_only_what_is_missing():
    assert unverified(DROPS_UNICODE, REQUIREMENTS) == ["support unicode input"]
    assert unverified(COMPLETE, REQUIREMENTS) == []


# ---------------------------------------------------------------- through compile_prompt()


def test_a_dropped_requirement_is_reported_on_the_result():
    """R1-002: the model wrote three of four stated requirements and nothing noticed."""
    result = compile_prompt("Reverse a string, typed, ValueError, unicode.", Generates(DROPS_UNICODE))

    assert result.unverified_requirements == ["support unicode input"]


def test_a_faithful_rewrite_reports_nothing():
    result = compile_prompt("Reverse a string, typed, ValueError, unicode.", Generates(COMPLETE))

    assert result.unverified_requirements == []


def test_a_reworded_rewrite_is_not_flagged():
    """The check must not cry wolf: a check that fires on good output gets ignored."""
    result = compile_prompt("Reverse a string, typed, ValueError, unicode.", Generates(REWORDED))

    assert result.unverified_requirements == []


def test_constraints_are_checked_as_well_as_requirements():
    payload = PAYLOAD.model_copy(update={"constraints": ["Python 3.11", "Windows only"]})
    text = COMPLETE  # mentions Python, but neither 3.11 nor Windows

    result = compile_prompt("x", Generates(text, payload))

    assert "Windows only" in result.unverified_requirements


def test_the_check_runs_before_tightening_so_the_two_causes_are_distinguishable():
    """A requirement the model never wrote is a different failure from one tightening
    removed, and must not be masked by the tightening pass."""
    provider = Generates(DROPS_UNICODE)

    result = compile_prompt("x", provider)

    assert result.unverified_requirements == ["support unicode input"]
    assert result.optimized_prompt  # the compile still succeeds and returns usable output


def test_a_dropped_requirement_does_not_fail_the_compile():
    """Deliberately reported rather than raised: this cannot tell "dropped" from
    "reworded past recognition", and hard-failing a good compile is the worse error."""
    result = compile_prompt("x", Generates(DROPS_UNICODE))

    assert result.optimized_prompt
    assert result.analysis.explicit_requirements == REQUIREMENTS


def test_the_finding_survives_a_json_round_trip():
    from prompt_compiler.models import CompiledPrompt

    result = compile_prompt("x", Generates(DROPS_UNICODE))

    restored = CompiledPrompt.model_validate_json(result.model_dump_json())

    assert restored.unverified_requirements == ["support unicode input"]


@pytest.mark.parametrize("case_name", ["coding", "explicit_requirements", "long", "constraints"])
def test_the_corpus_cases_pass_their_own_validation(case_name):
    """A generated prompt that does contain every stated item must never be flagged."""
    case = BY_NAME[case_name]
    payload = case.payload
    stated = [*payload.explicit_requirements, *payload.constraints]
    text = "REQUIREMENTS\n" + "\n".join(f"- {item}" for item in stated)

    assert unverified(text, stated) == []
