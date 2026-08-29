"""Heuristic quality scoring (Phase 7).

Three properties carry most of the weight here.

**The two subjects must not leak into each other.** The analysis describes the prompt the
user wrote; the rewrite is a different artefact. If a change to the rewrite could move a
prompt-side score, the report would be quietly passing off input quality as output quality
- the exact dishonesty the split exists to prevent.

**Size must not be scored.** `token_efficiency` counts waste, never length. A test pins
that, because "shorter is better" is the single easiest rule to reintroduce by accident,
and the project rejects it.

**A safeguard that never landed must be caught.** Nothing before this phase verified that
the factuality rules handed to the generator actually reached the prompt: `tighten()`
protects only text already present, so a rule the model never wrote is absent from both
sides of its comparison and the guard passes trivially. Same blind spot as R1-002, one
layer over.
"""

from __future__ import annotations

import pytest

from prompt_compiler import compile_prompt
from prompt_compiler.analyzer.models import (
    Ambiguity,
    AnalysisPayload,
    Assumption,
    HallucinationRisk,
    PromptAnalysis,
)
from prompt_compiler.models import CompiledPrompt
from prompt_compiler.optimizer.generator import GeneratedPrompt
from prompt_compiler.providers.base import Provider
from prompt_compiler.safety.hallucination import (
    MAX_SAFEGUARDS,
    MISSING_INFORMATION,
    NO_FABRICATION,
    earned_rules,
    safeguards,
)
from prompt_compiler.scoring import (
    BANDS,
    SCORE_METHOD,
    QualityReport,
    band,
    score,
)

from .corpus import CASES

CLEAN = "GOAL\nExplain something clearly."


def analysis(**overrides) -> PromptAnalysis:
    """A minimal analysis, overridden field by field so each test states its own evidence."""
    fields: dict = {
        "task_type": "explanation",
        "primary_goal": "Explain something.",
        "complexity": "simple",
        "confidence": 0.8,
    }
    fields.update(overrides)
    return PromptAnalysis(original_prompt="explain something", **fields)


def ambiguity(severity: str) -> Ambiguity:
    return Ambiguity(kind="unclear_scope", text="how much is wanted", severity=severity)


class Generates(Provider):
    """A provider whose generation step returns exactly the text it was given."""

    name = "fake"
    model = "fake-model"

    def __init__(self, text: str, payload: AnalysisPayload) -> None:
        self._text = text
        self._payload = payload

    def structured(self, *, system: str, user: str, schema):
        if schema is AnalysisPayload:
            return self._payload
        return GeneratedPrompt(optimized_prompt=self._text)


# ------------------------------------------------------------------- the heuristic contract


def test_the_report_says_it_is_a_heuristic():
    """Rule from CLAUDE.md: a quality claim is measured or it is labelled unverified."""
    report = score(analysis(), CLEAN)

    assert report.heuristic is True
    assert report.method == SCORE_METHOD
    assert "heuristic" in report.summary()
    assert "not measurements" in report.summary()


def test_every_dimension_carries_the_evidence_it_was_computed_from():
    """A bare number is an assertion; a number naming its evidence can be argued with."""
    report = score(analysis(), CLEAN)

    assert len(report.dimensions) == 6
    for dimension in report.dimensions:
        assert dimension.basis.strip()


def test_there_is_no_grand_total():
    """Deliberate: averaging "your request was vague" with "the rewrite kept your
    constraints" produces a figure that means nothing and lets each fact hide the other."""
    assert not hasattr(QualityReport, "overall")

    report = score(analysis(), CLEAN)
    assert isinstance(report.prompt_score, int)
    assert isinstance(report.rewrite_score, int)


def test_bands_are_ordered_and_cover_every_score():
    floors = [floor for floor, _ in BANDS]
    assert floors == sorted(floors, reverse=True)
    assert floors[-1] == 0
    for value in range(0, 101):
        assert band(value) in {name for _, name in BANDS}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_every_corpus_case_scores_in_range(case):
    bound = PromptAnalysis(original_prompt=case.prompt, **case.payload.model_dump())
    rules = safeguards(bound)
    stated = [*bound.explicit_requirements, *bound.constraints]
    text = "\n".join([bound.primary_goal, *(f"- {item}" for item in stated), *rules])

    report = score(bound, text, safeguards=rules, unverified_requirements=[])

    assert len(report.dimensions) == 6
    for dimension in report.dimensions:
        assert 0 <= dimension.score <= 100


# --------------------------------------------------------------- the two subjects stay apart


def test_rewriting_the_text_cannot_move_a_prompt_side_score():
    """The separation property. A prompt-side score describes what the user wrote, and
    nothing this tool produces afterwards may change it."""
    subject = analysis(
        explicit_requirements=["do the thing"],
        ambiguities=[ambiguity("high")],
        missing_information=["which thing"],
    )

    good = score(subject, "GOAL\nDo the thing, precisely.")
    bad = score(subject, "x\nx\nx\n\n\n\nin order to")

    assert good.prompt_score == bad.prompt_score
    assert [d.score for d in good.about("prompt")] == [d.score for d in bad.about("prompt")]
    assert good.rewrite_score != bad.rewrite_score  # the rewrite side did notice


def test_the_dimensions_are_split_across_the_two_subjects():
    report = score(analysis(), CLEAN)

    assert [d.name for d in report.about("prompt")] == [
        "clarity",
        "specificity",
        "completeness",
    ]
    assert [d.name for d in report.about("rewrite")] == [
        "requirement_coverage",
        "risk_coverage",
        "token_efficiency",
    ]


# --------------------------------------------------------------------------------- clarity


def test_an_unambiguous_prompt_is_fully_clear():
    assert score(analysis(), CLEAN).of("clarity").score == 100


def test_severity_orders_the_penalty():
    high = score(analysis(ambiguities=[ambiguity("high")]), CLEAN).of("clarity").score
    medium = score(analysis(ambiguities=[ambiguity("medium")]), CLEAN).of("clarity").score
    low = score(analysis(ambiguities=[ambiguity("low")]), CLEAN).of("clarity").score

    assert high < medium < low < 100


def test_three_high_severity_ambiguities_land_in_the_lowest_band():
    """Anchored to MAX_QUESTIONS: three is where clarification_questions() stops asking,
    because more than that is an interrogation rather than a prompt."""
    subject = analysis(ambiguities=[ambiguity("high") for _ in range(3)])

    assert score(subject, CLEAN).of("clarity").band == "weak"


def test_clarity_floors_at_zero():
    subject = analysis(ambiguities=[ambiguity("high") for _ in range(20)])

    assert score(subject, CLEAN).of("clarity").score == 0


# ----------------------------------------------------------------------------- specificity


def test_a_prompt_that_states_nothing_is_not_specific():
    assert score(analysis(), CLEAN).of("specificity").score == 0


def test_specificity_is_relative_to_what_the_request_needs():
    """The same three items are specific for a simple ask and thin for a complex one."""
    stated = {"explicit_requirements": ["a", "b"], "constraints": ["c"]}

    simple = score(analysis(complexity="simple", **stated), CLEAN).of("specificity")
    complex_ = score(analysis(complexity="complex", **stated), CLEAN).of("specificity")

    assert simple.score > complex_.score


def test_specificity_is_capped_rather_than_unbounded():
    subject = analysis(explicit_requirements=[f"item {n}" for n in range(50)])

    assert score(subject, CLEAN).of("specificity").score == 100


def test_inference_and_gaps_do_not_count_as_specificity():
    """Only what the user stated counts. Crediting an assumption would reward the analysis
    for filling in a blank, which is precisely what the three-way split exists to stop."""
    inferred = analysis(
        assumptions=[Assumption(text="they mean Python", basis="the word script")],
        missing_information=["which language"],
    )

    assert score(inferred, CLEAN).of("specificity").score == 0


# ---------------------------------------------------------------------------- completeness


def test_a_prompt_missing_nothing_is_complete():
    assert score(analysis(), CLEAN).of("completeness").score == 100


def test_each_blocking_gap_costs_and_the_score_floors_at_zero():
    scores = [
        score(analysis(missing_information=[f"gap {n}" for n in range(count)]), CLEAN)
        .of("completeness")
        .score
        for count in range(6)
    ]

    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 100
    assert scores[-1] == 0


def test_completeness_and_clarity_are_different_axes():
    """The taxonomy separates them by definition: a gap stops a responder proceeding, an
    ambiguity only misdirects them. Scoring must keep them separate too."""
    gapped = score(analysis(missing_information=["which file", "which language"]), CLEAN)
    unclear = score(analysis(ambiguities=[ambiguity("high")]), CLEAN)

    assert gapped.of("clarity").score == 100
    assert gapped.of("completeness").score < 100
    assert unclear.of("completeness").score == 100
    assert unclear.of("clarity").score < 100


# --------------------------------------------------------------------- requirement coverage


def test_nothing_to_carry_forward_says_so_rather_than_implying_a_check_ran():
    """A vacuous 100 must explain itself, or it reads as a verification that happened."""
    dimension = score(analysis(), CLEAN).of("requirement_coverage")

    assert dimension.score == 100
    assert "no stated requirement" in dimension.basis


def test_coverage_is_proportional_to_what_survived():
    subject = analysis(explicit_requirements=["a", "b", "c"], constraints=["d"])

    full = score(subject, CLEAN, unverified_requirements=[])
    half = score(subject, CLEAN, unverified_requirements=["a", "b"])
    none = score(subject, CLEAN, unverified_requirements=["a", "b", "c", "d"])

    assert full.of("requirement_coverage").score == 100
    assert half.of("requirement_coverage").score == 50
    assert none.of("requirement_coverage").score == 0


def test_an_unverified_item_that_was_never_stated_cannot_drive_the_score_negative():
    subject = analysis(explicit_requirements=["a"])

    dimension = score(subject, CLEAN, unverified_requirements=["z", "y", "x"]).of(
        "requirement_coverage"
    )

    assert dimension.score == 100


# --------------------------------------------------------------------------- risk coverage


def test_no_actionable_risk_means_nothing_was_owed():
    dimension = score(analysis(), CLEAN).of("risk_coverage")

    assert dimension.score == 100
    assert "no actionable risk" in dimension.basis


def test_a_safeguard_the_model_never_wrote_is_caught():
    """The gap this dimension exists to close: generate() is handed the rules but nothing
    checked they arrived, and tighten()'s guard cannot see a rule that was never there."""
    subject = analysis(missing_information=["which file"])
    rules = safeguards(subject)
    assert rules  # the analysis really did earn one

    dimension = score(subject, "GOAL\nFix it.", safeguards=rules).of("risk_coverage")

    assert dimension.score == 0


def test_a_safeguard_that_landed_is_recognised():
    subject = analysis(missing_information=["which file"])
    rules = safeguards(subject)

    dimension = score(subject, f"GOAL\nFix it.\n\n{rules[0]}", safeguards=rules).of(
        "risk_coverage"
    )

    assert dimension.score == 100


def test_a_reworded_safeguard_still_counts():
    """Presence is judged the way validation.py judges it - a check that fires on good
    output gets ignored."""
    subject = analysis(missing_information=["which file"])

    text = (
        "GOAL\nFix it.\n\nIf information you need is missing, say what is missing rather "
        "than filling it in yourself."
    )
    dimension = score(subject, text, safeguards=[MISSING_INFORMATION]).of("risk_coverage")

    assert dimension.score == 100


def test_rules_cut_by_the_cap_are_reported():
    """The cap is a real coverage loss and must be visible, not silently absorbed."""
    subject = analysis(
        task_type="code generation",
        missing_information=["which file"],
        ambiguities=[
            Ambiguity(kind="conflicting_instructions", text="a vs b", severity="high")
        ],
        hallucination_risks=[
            HallucinationRisk(
                kind=kind, text=f"risk about {kind}", grounding="assumed", severity="high"
            )
            for kind in (
                "unsupported_assumption",
                "fabrication_prone",
                "ambiguous_reference",
                "contradictory_requirements",
            )
        ],
    )
    earned = earned_rules(subject)
    rules = safeguards(subject)
    assert len(earned) > len(rules) == MAX_SAFEGUARDS

    text = "GOAL\nDo it.\n\n" + "\n".join(rules)
    dimension = score(subject, text, safeguards=rules).of("risk_coverage")

    assert dimension.score < 100
    assert "cut by the cap" in dimension.basis


def test_splitting_out_earned_rules_kept_severity_ordering_intact():
    """R1-003: the cap must cut the least severe evidence, not the last-declared rule.
    earned_rules() was split out of safeguards() for this phase, so the ordering that
    property depends on is re-pinned here against concrete output."""
    subject = analysis(
        missing_information=["which file"],  # unrated, treated as medium
        hallucination_risks=[
            HallucinationRisk(
                kind="fabrication_prone",
                text="invents versions",
                grounding="unknown",
                severity="high",
            )
        ],
    )

    assert earned_rules(subject) == [NO_FABRICATION, MISSING_INFORMATION]
    assert safeguards(subject) == earned_rules(subject)


def test_more_rules_than_the_cap_allows_are_still_scored_against_what_was_earned():
    """A caller that hands over more rules than were earned gets a capped score, not a
    crash: failing a good compile inside the scorer would be the worse outcome."""
    subject = analysis()
    dimension = score(subject, CLEAN, safeguards=[MISSING_INFORMATION] * 5).of(
        "risk_coverage"
    )

    assert dimension.score == 100


# ------------------------------------------------------------------------ token efficiency


def test_a_clean_rewrite_wastes_nothing():
    assert score(analysis(), CLEAN).of("token_efficiency").score == 100


def test_growth_is_not_penalised():
    """CLAUDE.md: shorter is not the goal. A precise prompt is usually longer than the
    vague one it came from, and that is the transformation working, not a cost."""
    subject = analysis(explicit_requirements=["reverse the string"])
    short = "Reverse it."
    long = (
        "GOAL\nWrite a Python function that reverses a string.\n\n"
        "REQUIREMENTS\n- reverse the string\n- handle unicode correctly\n"
        "- raise ValueError when the input is not a string\n\n"
        "OUTPUT FORMAT\nA single function with type hints and a docstring."
    )
    assert len(long) > len(short) * 10

    assert score(subject, short).of("token_efficiency").score == 100
    assert score(subject, long).of("token_efficiency").score == 100


def test_redundancy_left_in_the_rewrite_is_penalised():
    duplicated = "GOAL\nDo the thing.\n- step one\n- step one"

    dimension = score(analysis(), duplicated).of("token_efficiency")

    assert dimension.score < 100
    assert "redundanc" in dimension.basis


def test_filler_is_penalised_even_though_the_tightener_leaves_it_alone():
    """Reported-not-removed is right for tighten(), where cutting words is a rewrite. It is
    still padding the generator should not have produced."""
    padded = "GOAL\nIn order to do this, it is important to note that you should begin."

    dimension = score(analysis(), padded).of("token_efficiency")

    assert dimension.score < 100
    assert "filler" in dimension.basis


def test_unnecessary_content_carried_into_the_rewrite_is_penalised():
    subject = analysis(unnecessary_content=["thanks so much in advance"])

    kept = score(subject, "GOAL\nDo it.\n\nThanks so much in advance!")
    dropped = score(subject, "GOAL\nDo it.")

    assert kept.of("token_efficiency").score < dropped.of("token_efficiency").score
    assert dropped.of("token_efficiency").score == 100


def test_token_efficiency_floors_at_zero():
    dupes = "\n".join(["- same line"] * 30)

    assert score(analysis(), dupes).of("token_efficiency").score == 0


# ------------------------------------------------------------------ through compile_prompt()


PAYLOAD = AnalysisPayload(
    task_type="code generation",
    primary_goal="Write a Python function that reverses a string.",
    explicit_requirements=["reverse the input string"],
    constraints=["Python"],
    missing_information=["which Python version"],
    complexity="moderate",
    confidence=0.9,
)


def test_a_compile_carries_a_quality_report():
    result = compile_prompt("reverse a string", Generates("GOAL\nReverse a string.", PAYLOAD))

    assert result.quality.heuristic is True
    assert len(result.quality.dimensions) == 6


def test_the_report_survives_a_json_round_trip():
    result = compile_prompt("reverse a string", Generates("GOAL\nReverse a string.", PAYLOAD))

    restored = CompiledPrompt.model_validate_json(result.model_dump_json())

    assert restored.quality.of("clarity").score == result.quality.of("clarity").score
    assert restored.quality.summary() == result.quality.summary()


def test_a_compile_that_drops_its_safeguards_scores_badly_on_risk_coverage():
    """End to end: the analysis earns a rule, the model ignores it, and the report says so
    instead of the omission passing silently through the whole pipeline."""
    result = compile_prompt("reverse a string", Generates("Reverse the input string.", PAYLOAD))

    assert result.safeguards  # a rule was earned and handed to generation
    assert result.quality.of("risk_coverage").score == 0


def test_a_compile_that_keeps_its_safeguards_scores_well():
    bound = PromptAnalysis(original_prompt="reverse a string", **PAYLOAD.model_dump())
    text = "GOAL\nReverse the input string in Python.\n\n" + "\n".join(safeguards(bound))

    result = compile_prompt("reverse a string", Generates(text, PAYLOAD))

    assert result.quality.of("risk_coverage").score == 100
    assert result.quality.of("requirement_coverage").score == 100


def test_scoring_imports_no_provider():
    """Scoring is core: it must be importable and runnable with no network and no key.
    Checked on the imports themselves, since the prose legitimately discusses providers."""
    import ast

    import prompt_compiler.scoring as module

    tree = ast.parse(open(module.__file__, encoding="utf-8").read())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")

    assert not any("provider" in name for name in imported)
    assert imported == {
        "__future__",
        "collections",
        "collections.abc",
        "typing",
        "pydantic",
        "analyzer.models",
        "optimizer.token_optimizer",
        "safety.hallucination",
        "validation",
    }


def test_scoring_adds_no_provider_call():
    """One LLM call per prompt wherever practical; Phase 7 adds none. The pipeline still
    makes exactly the two it made before - analyze, then generate."""

    class Counting(Generates):
        def __init__(self, text, payload):
            super().__init__(text, payload)
            self.calls = 0

        def structured(self, *, system, user, schema):
            self.calls += 1
            return super().structured(system=system, user=user, schema=schema)

    provider = Counting("GOAL\nReverse a string.", PAYLOAD)
    compile_prompt("reverse a string", provider)

    assert provider.calls == 2
