"""Token measurement and redundancy removal.

The load-bearing tests are the preservation ones. Everything here is pure and local -
no provider, no key, no network - which is itself part of the contract: the optimizer
must not care which provider produced the text.
"""

from __future__ import annotations

import pytest

from prompt_compiler import compile_prompt
from prompt_compiler.optimizer.token_optimizer import (
    COUNT_METHOD,
    FILLER_PHRASES,
    Redundancy,
    TokenReport,
    count_tokens,
    find_redundancies,
    measure,
    missing_from,
    tighten,
)

from .conftest import FakeProvider
from .corpus import BY_NAME


# ------------------------------------------------------------------------------ counting


def test_counting_is_deterministic():
    text = "Write a Python function that reverses a string."

    assert count_tokens(text) == count_tokens(text)


def test_empty_text_costs_nothing():
    assert count_tokens("") == 0
    assert count_tokens("   \n  ") == 0


def test_longer_text_costs_more():
    short = count_tokens("Explain recursion.")
    long = count_tokens("Explain recursion with two worked examples and a diagram.")

    assert long > short


def test_counting_is_in_the_right_ballpark():
    """Not exact - but a wildly wrong estimate would make every comparison meaningless."""
    text = " ".join(["word"] * 100)

    assert 80 <= count_tokens(text) <= 220


def test_the_method_is_labelled_so_a_stored_measurement_is_recognisable():
    report = measure("a b c", "a b c d")

    assert report.estimated is True
    assert report.method == COUNT_METHOD


# --------------------------------------------------------------------------- the report


def test_a_shorter_prompt_reports_a_saving():
    report = TokenReport(original_tokens=100, optimized_tokens=70)

    assert report.tokens_saved == 30
    assert report.percent_change == -30.0
    assert "shorter" in report.summary()


def test_a_longer_prompt_reports_honestly_rather_than_hiding_it():
    """Making a vague request precise usually costs tokens. Saying so is the point."""
    report = TokenReport(original_tokens=10, optimized_tokens=40)

    assert report.tokens_saved == -30
    assert report.percent_change == 300.0
    assert "longer" in report.summary()


def test_an_empty_original_does_not_divide_by_zero():
    assert TokenReport(original_tokens=0, optimized_tokens=5).percent_change == 0.0


# ------------------------------------------------------------------------- detection


def test_a_duplicate_line_is_found_and_removable():
    findings = find_redundancies("do the thing\ndo the thing")

    assert [f.kind for f in findings] == ["duplicate_line"]
    assert findings[0].removable


def test_a_repeated_requirement_is_named_as_such():
    findings = find_redundancies(
        "- include type hints\n- include type hints", requirements=["include type hints"]
    )

    assert findings[0].kind == "repeated_requirement"


def test_a_repeated_constraint_is_named_as_such():
    findings = find_redundancies("- Python 3.11\n- Python 3.11", constraints=["Python 3.11"])

    assert findings[0].kind == "repeated_constraint"


@pytest.mark.parametrize("phrase", FILLER_PHRASES)
def test_verbosity_is_reported_but_never_removed_automatically(phrase):
    """Cutting words out of a sentence is a rewrite, and rewrites carry meaning."""
    findings = find_redundancies(f"Do the thing. {phrase} it matters.")

    filler = [f for f in findings if f.kind == "filler_phrase"]
    assert filler and not any(f.removable for f in filler)


def test_excess_whitespace_and_empty_bullets_are_found():
    kinds = {f.kind for f in find_redundancies("a\n\n\n\nb\n-  \nc   \n")}

    assert "excess_whitespace" in kinds
    assert "needless_formatting" in kinds


def test_a_clean_prompt_has_nothing_to_report():
    assert find_redundancies("Explain recursion with one worked example.") == []


# -------------------------------------------------------------------------- tightening


def test_duplicates_collapse_to_one():
    text = "- do the thing\n- do the thing\n- do the thing"

    tightened, _ = tighten(text)

    assert tightened.count("do the thing") == 1


def test_bullet_style_does_not_hide_a_duplicate():
    tightened, _ = tighten("- do the thing\n* do the thing.\n1. Do the thing")

    assert tightened.count("do the thing") == 1


def test_formatting_noise_is_removed():
    tightened, _ = tighten("GOAL\n\n\n\nDo the thing.   \n-  \n")

    assert "\n\n\n" not in tightened
    assert not any(line.endswith(" ") for line in tightened.splitlines())
    assert "-  " not in tightened


def test_nothing_is_touched_when_there_is_nothing_to_remove():
    text = "GOAL\nWrite a Python function that reverses a string."

    tightened, findings = tighten(text)

    assert tightened == text
    assert findings == []


# -------------------------------------------------------- preservation (the load-bearing part)


REQUIREMENTS = [
    "read the CSV with pandas",
    "drop duplicate rows",
    "write an xlsx file with one sheet per segment",
    "raise ValueError on non-string input",
]
CONSTRAINTS = ["Python 3.11", "only the standard library plus pandas and openpyxl", "Windows"]


def test_multiple_requirements_all_survive():
    text = "REQUIREMENTS\n" + "\n".join(f"- {item}" for item in REQUIREMENTS)

    tightened, _ = tighten(text, REQUIREMENTS, CONSTRAINTS)

    assert missing_from(tightened, REQUIREMENTS) == []


def test_important_constraints_all_survive():
    text = "CONSTRAINTS\n" + "\n".join(f"- {item}" for item in CONSTRAINTS)

    tightened, _ = tighten(text, REQUIREMENTS, CONSTRAINTS)

    assert missing_from(tightened, CONSTRAINTS) == []


def test_technical_specifications_survive_verbatim():
    specs = ["Python 3.11", "pandas>=2.0", "UTF-8 encoding", "exit code 2 on bad input"]
    text = "CONSTRAINTS\n" + "\n".join(f"- {item}" for item in specs) + "\n\n\n- Python 3.11"

    tightened, _ = tighten(text, constraints=specs)

    assert missing_from(tightened, specs) == []


def test_output_format_requirements_survive():
    formats = ["a JSON object", "exactly three bullet points", "one sheet per segment"]
    text = "OUTPUT FORMAT\n" + "\n".join(f"- {item}" for item in formats) * 2

    tightened, _ = tighten(text, requirements=formats)

    assert missing_from(tightened, formats) == []


def test_emphasis_by_repetition_keeps_the_requirement_once():
    """Repetition is collapsed, not discarded - the requirement still stands."""
    text = (
        "REQUIREMENTS\n- do not overwrite existing files\n"
        "- do not overwrite existing files\n- do not overwrite existing files"
    )

    tightened, findings = tighten(text, ["do not overwrite existing files"])

    assert tightened.count("do not overwrite existing files") == 1
    assert missing_from(tightened, ["do not overwrite existing files"]) == []
    assert len([f for f in findings if f.removable]) == 2


def test_long_context_survives():
    case = BY_NAME["long"]
    payload = case.payload
    text = (
        "CONTEXT\n"
        + "\n".join(f"- {item}" for item in payload.context)
        + "\n\n\nREQUIREMENTS\n"
        + "\n".join(f"- {item}" for item in payload.explicit_requirements)
        + "\n\nCONSTRAINTS\n"
        + "\n".join(f"- {item}" for item in payload.constraints)
    )

    tightened, _ = tighten(text, payload.explicit_requirements, payload.constraints)

    assert missing_from(tightened, payload.explicit_requirements) == []
    assert missing_from(tightened, payload.constraints) == []
    assert missing_from(tightened, payload.context) == []


def test_ambiguous_language_is_left_alone():
    """Vagueness is the analyzer's problem, not the token optimizer's."""
    text = "GOAL\nOrganize the folder somehow, make it nice."

    tightened, _ = tighten(text)

    assert tightened == text


def test_coding_requirements_survive():
    reqs = ["reverse a string", "include type hints", "raise ValueError on non-string input"]
    text = "REQUIREMENTS\n" + "\n".join(f"- {item}" for item in reqs) + "\n- include type hints"

    tightened, _ = tighten(text, reqs)

    assert missing_from(tightened, reqs) == []


def test_multi_step_instructions_keep_their_steps_and_order():
    steps = ["1. Load the CSV", "2. Drop duplicates", "3. Write the xlsx"]
    text = "PROCESS\n" + "\n".join(steps)

    tightened, _ = tighten(text)

    assert [line for line in tightened.splitlines() if line[0].isdigit()] == steps


def test_two_phrasings_of_one_requirement_collapse_without_loss():
    """"Handle errors" and "handle errors!" are the same requirement, so collapsing them
    loses nothing - the check is that the requirement is still there afterwards."""
    text = "- Handle errors\n- handle errors!"

    tightened, _ = tighten(text, ["Handle errors", "handle errors!"])

    assert tightened.lower().count("handle errors") == 1
    assert missing_from(tightened, ["Handle errors", "handle errors!"]) == []


def test_a_pass_that_would_lose_a_requirement_is_abandoned_entirely(monkeypatch):
    """The failure mode must be "no saving", never "quietly weaker prompt".

    By construction the current removals - exact duplicates, empty bullets, whitespace -
    cannot drop a requirement. This drives the guard directly, so it still holds if a
    future removal rule turns out to be less careful than today's.
    """
    import prompt_compiler.optimizer.token_optimizer as module

    monkeypatch.setattr(module, "_apply", lambda text: "everything went missing")
    text = "- keep this requirement\n- keep this requirement"

    tightened, findings = module.tighten(text, ["keep this requirement"])

    assert tightened == text, "the destructive pass must be discarded"
    assert not any(f.removable for f in findings)


def test_missing_from_finds_what_was_dropped():
    assert missing_from("only this one", ["only this one", "not this one"]) == ["not this one"]


# ------------------------------------------------------ provider independence and pipeline


def test_optimization_imports_no_provider():
    """The engine must not care whose API produced the text - checked on the imports
    themselves, since the prose legitimately discusses providers."""
    import ast

    import prompt_compiler.optimizer.token_optimizer as module

    tree = ast.parse(open(module.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")

    assert imported == {"__future__", "re", "collections.abc", "math", "typing", "pydantic"}


def test_the_same_text_measures_the_same_whichever_provider_produced_it():
    case = BY_NAME["coding"]
    text = "GOAL\nDo the thing.\n- a requirement\n- a requirement"

    from_gemini = FakeProvider(payload=case.payload, optimized=text, model="gemini-model")
    from_groq = FakeProvider(payload=case.payload, optimized=text, model="groq-model")

    first = compile_prompt(case.prompt, from_gemini)
    second = compile_prompt(case.prompt, from_groq)

    assert first.optimized_prompt == second.optimized_prompt
    assert first.tokens.model_dump() == second.tokens.model_dump()


def test_a_compile_reports_its_token_measurement():
    case = BY_NAME["coding"]
    provider = FakeProvider(
        payload=case.payload, optimized="GOAL\nDo it.\n- a thing\n- a thing\n\n\n- "
    )

    result = compile_prompt(case.prompt, provider)

    assert result.tokens.original_tokens == count_tokens(case.prompt)
    assert result.tokens.optimized_tokens == count_tokens(result.optimized_prompt)
    assert result.tokens.redundancy_removed > 0
    assert result.optimized_prompt.count("a thing") == 1


def test_requirements_survive_a_whole_compile():
    case = BY_NAME["long"]
    generated = (
        "GOAL\nClean the billing CSV.\n\nREQUIREMENTS\n"
        + "\n".join(f"- {item}" for item in case.payload.explicit_requirements)
        + "\n"
        + "\n".join(f"- {item}" for item in case.payload.explicit_requirements)  # duplicated
        + "\n\nCONSTRAINTS\n"
        + "\n".join(f"- {item}" for item in case.payload.constraints)
    )
    provider = FakeProvider(payload=case.payload, optimized=generated)

    result = compile_prompt(case.prompt, provider)

    assert missing_from(result.optimized_prompt, case.payload.explicit_requirements) == []
    assert missing_from(result.optimized_prompt, case.payload.constraints) == []
    assert result.tokens.redundancy_removed > 0


def test_the_measurement_survives_a_json_round_trip():
    from prompt_compiler.models import CompiledPrompt

    case = BY_NAME["coding"]
    result = compile_prompt(case.prompt, FakeProvider(payload=case.payload, optimized="Do it."))

    restored = CompiledPrompt.model_validate_json(result.model_dump_json())

    assert restored.tokens.original_tokens == result.tokens.original_tokens
    assert restored.tokens.method == COUNT_METHOD


def test_findings_are_carried_on_the_result():
    case = BY_NAME["coding"]
    provider = FakeProvider(payload=case.payload, optimized="- a thing\n- a thing")

    result = compile_prompt(case.prompt, provider)

    assert any(isinstance(f, Redundancy) for f in result.tokens.findings)


def test_tightening_never_empties_a_prompt():
    """Regression: generate() guards its own output, but this pass runs after it. A
    prompt that is almost entirely formatting must survive, not vanish."""
    text = "-  \n-\n\n\n"

    tightened, findings = tighten(text)

    assert tightened == text
    assert not any(f.removable for f in findings)


def test_a_compile_never_yields_an_empty_prompt():
    case = BY_NAME["very_simple"]
    provider = FakeProvider(payload=case.payload, optimized="-")

    result = compile_prompt(case.prompt, provider)

    assert result.optimized_prompt.strip()


def test_safeguards_are_protected_from_the_tightening_pass():
    """Phase 5's earned rules must survive Phase 6, not be treated as spare text."""
    case = BY_NAME["constraints"]
    rule = "If information you need is missing, say what is missing instead of filling it in."
    provider = FakeProvider(
        payload=case.payload,
        optimized=f"GOAL\nSummarize the article.\n\nFACTUALITY RULES\n- {rule}\n- {rule}",
    )

    result = compile_prompt(case.prompt, provider)

    assert rule in result.optimized_prompt
    assert result.optimized_prompt.count(rule) == 1


# ------------------------------------------- R1-001: punctuation must not fuse distinct items


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Timeout of 5.5 seconds", "Timeout of 55 seconds"),          # decimal point
        ("Operate at -5 degrees", "Operate at 5 degrees"),            # sign
        ("requires v1.2", "requires v12"),                            # version string
        ("under 2.5% overhead", "under 25% overhead"),                # percentage
        ("costs $1.50", "costs $150"),                                # price
        ("retry 3 times", "retry 3.0 times"),                         # trailing precision
    ],
)
def test_two_items_differing_only_by_punctuation_both_survive(first, second):
    """Regression: normalisation deleted punctuation, so "5.5" and "55" became one key.
    The duplicate collapse then removed a real constraint, and the preservation check -
    sharing the same normalisation - reported nothing missing."""
    text = "CONSTRAINTS\n" + f"- {first}\n- {second}"

    tightened, _ = tighten(text, constraints=[first, second])

    assert first in tightened
    assert second in tightened
    assert missing_from(tightened, [first, second]) == []


def test_the_preservation_check_is_not_blinded_by_punctuation():
    """The guard must be able to see a loss that normalisation used to hide."""
    assert missing_from("- Timeout of 5.5 seconds", ["Timeout of 55 seconds"]) == [
        "Timeout of 55 seconds"
    ]
    assert missing_from("- Operate at -5 degrees", ["Operate at 5 degrees"]) == [
        "Operate at 5 degrees"
    ]


def test_cosmetic_differences_still_collapse():
    """The fix must not go too far the other way: bullet style, case and trailing
    punctuation are still cosmetic."""
    tightened, _ = tighten("- do the thing\n* Do the thing.\n1. do the thing!")

    assert tightened.lower().count("do the thing") == 1


# ---------------------------------------------- R1-004: identifiers are words, not punctuation


def test_an_underscored_identifier_costs_what_its_length_says():
    """Regression: str.isalnum() is False for anything containing an underscore, so every
    snake_case identifier was charged one token however long it was."""
    assert count_tokens("max_retry_count_value") >= 5


def test_underscores_do_not_change_the_price_of_a_name():
    with_underscores = count_tokens("max_retry_count_value")
    without = count_tokens("maxretrycountvalue")

    assert abs(with_underscores - without) <= 2


def test_a_code_signature_is_counted_like_code():
    signature = "def __init__(self, max_retry_count=5, api_base_url=None):"

    assert count_tokens(signature) >= 18


def test_the_ballpark_check_covers_identifiers_too():
    """The original ballpark test used only underscore-free words, so the bias was
    invisible to the suite's own self-check."""
    text = " ".join(["retry_count_value"] * 50)

    assert 150 <= count_tokens(text) <= 350
