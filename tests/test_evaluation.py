"""Evaluation and benchmarking (Phase 8).

The harness is an instrument, so its own correctness matters more than usual: a bug here
does not crash anything, it produces a plausible number that gets believed. The checks
below are mostly about refusing to report something the data does not support - a constant
series correlating with nothing, a run where everything failed, a threshold that never
fires.

The sweep is exercised on a reduced grid. The full grid is ~46,000 points and takes about
twenty seconds, which does not belong in a suite that runs on every change; what needs
pinning is that the machinery is right, and a small grid does that.
"""

from __future__ import annotations

import pytest

from prompt_compiler.analyzer.models import AnalysisPayload
from prompt_compiler.errors import ProviderError, RateLimitError
from prompt_compiler.evaluation.benchmark import BenchmarkReport, Case, Outcome, run
from prompt_compiler.evaluation.metrics import (
    NARROW_RANGE,
    REDUNDANT_CORRELATION,
    binding_rate,
    correlation,
    degeneracies,
    distribution,
)
from prompt_compiler.evaluation.sweep import (
    DIMENSION_NAMES,
    Axes,
    cap_pressure,
    points,
    prose_inversions,
    series,
)
from prompt_compiler.optimizer.generator import GeneratedPrompt
from prompt_compiler.providers.base import Provider

SMALL = Axes(
    task_type=("code generation",),
    complexity=("simple", "complex"),
    requirements=(0, 1, 4),
    constraints=(0, 2),
    context=(0,),
    expected_output=(False, True),
    missing=(0, 2),
    ambiguity=("none", "one_high"),
    risk=("none", "many_high"),
    # The full fidelity axis, not a subset. Trimming it to two values left
    # requirement_coverage with nothing to vary it, and the sweep then reported a healthy
    # dimension as dead - the grid's defect reading as the instrument's.
    fidelity=("faithful", "drops_requirement", "drops_safeguards", "wasteful"),
)


@pytest.fixture(scope="module")
def swept():
    return list(points(SMALL))


# ------------------------------------------------------------------------------- metrics


def test_a_distribution_reports_shape_without_judging_it():
    shape = distribution("scores", [0, 50, 50, 100])

    assert (shape.minimum, shape.maximum, shape.median) == (0, 100, 50)
    assert shape.mean == 50
    assert shape.distinct == 3
    assert shape.spread == 100
    assert not shape.is_constant


def test_an_empty_series_is_an_error_not_a_zero():
    """A mean of nothing is not 0, and reporting it as 0 would be a fabricated measurement."""
    with pytest.raises(ValueError):
        distribution("scores", [])


def test_a_constant_series_correlates_with_nothing_rather_than_zero():
    """None and 0.0 are different claims: "no relationship can be computed" is not
    "no relationship was found", and conflating them hides a dead dimension."""
    assert correlation([5, 5, 5, 5], [1, 2, 3, 4]) is None
    assert correlation([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert correlation([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_correlation_needs_matching_series():
    with pytest.raises(ValueError):
        correlation([1, 2, 3], [1, 2])


def test_a_dimension_that_cannot_move_is_reported():
    found = degeneracies({"dead": [50.0] * 10, "alive": [float(n * 10) for n in range(10)]})

    assert [f.kind for f in found if f.subject == "dead"] == ["constant"]
    assert not [f for f in found if f.subject == "alive" and f.kind == "constant"]


def test_two_dimensions_that_are_one_dimension_are_reported():
    rising = [float(n) for n in range(20)]
    found = degeneracies({"a": rising, "b": [value * 2 + 1 for value in rising]}, scale=20)

    assert any(f.kind == "redundant_pair" for f in found)


def test_a_narrow_dimension_is_reported_without_being_called_dead():
    found = degeneracies({"cramped": [50.0, 51.0, 52.0]})

    kinds = [f.kind for f in found if f.subject == "cramped"]
    assert kinds == ["narrow_range"]


def test_an_inverse_pair_is_as_redundant_as_a_direct_one():
    rising = [float(n) for n in range(20)]
    found = degeneracies({"a": rising, "b": [-value for value in rising]}, scale=20)

    assert any(f.kind == "redundant_pair" for f in found)


def test_the_redundancy_bar_is_near_identity_not_mere_correlation():
    """Dimensions drawn from related evidence should correlate. Only near-identity is a
    defect, or the check would demand independence the design never claimed."""
    assert 0.9 < REDUNDANT_CORRELATION < 1.0
    assert 0 < NARROW_RANGE < 1


def test_a_threshold_that_never_or_always_fires_is_not_a_threshold():
    assert binding_rate("cap", [False] * 10).kind == "never_binds"
    assert binding_rate("cap", [True] * 10).kind == "always_binds"
    assert binding_rate("cap", [True, False]) is None
    assert binding_rate("cap", []) is None


def test_a_metric_set_renders_its_distributions_and_findings():
    from prompt_compiler.evaluation.metrics import MetricSet

    rendered = MetricSet(
        distributions=[distribution("dead", [5.0] * 4)],
        findings=degeneracies({"dead": [5.0] * 4}),
    ).report()

    assert "dead: 5..5" in rendered
    assert "[constant]" in rendered


def test_an_empty_metric_set_renders_nothing_rather_than_a_headline():
    from prompt_compiler.evaluation.metrics import MetricSet

    assert MetricSet().report() == ""


# --------------------------------------------------------------------------------- sweep


def test_the_sweep_covers_its_whole_grid(swept):
    assert len(swept) == SMALL.size()


def test_the_sweep_needs_no_provider():
    """The point of the offline half: it runs with no key, no network, no adapter."""
    import ast

    import prompt_compiler.evaluation.sweep as module

    tree = ast.parse(open(module.__file__, encoding="utf-8").read())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")

    assert not any("provider" in name for name in imported)


def test_every_scored_dimension_moves_somewhere_in_the_sweep(swept):
    """The instrument check that matters: a dimension pinned at one value across the whole
    input space measures nothing, and its share of the group mean is dead weight."""
    data = series(swept)

    for name in DIMENSION_NAMES:
        assert not distribution(name, data[name]).is_constant, f"{name} never moved"


def test_no_two_dimensions_are_secretly_the_same_measurement(swept):
    data = series(swept)

    assert [f for f in degeneracies(data) if f.kind == "redundant_pair"] == []


def test_rewrite_fidelity_is_an_axis_so_the_rewrite_scores_can_vary(swept):
    """With a faithful rewrite every time, the three rewrite-side dimensions would sit at
    100 and read as dead when nothing was varying their input."""
    faithful = [p for p in swept if p.fidelity == "faithful"]
    dropped = [p for p in swept if p.fidelity == "drops_safeguards"]

    assert {p.scores["risk_coverage"] for p in dropped} != {
        p.scores["risk_coverage"] for p in faithful
    }


def test_the_safeguard_cap_is_reachable_and_discriminates(swept):
    pressure = cap_pressure(swept)

    assert any(pressure), "the cap never bound, so the sweep cannot say whether it works"
    assert not all(pressure)
    assert binding_rate("safeguard cap", pressure) is None


def test_a_swept_point_records_what_produced_it(swept):
    point = swept[0]

    assert point.stated_items >= 0
    assert set(point.scores) == set(DIMENSION_NAMES)
    assert point.emitted <= point.earned
    assert point.is_prose == (point.sections == 0)


# ----------------------------------------------------- the section floor (Phase 8 finding)


def test_many_items_in_few_categories_still_earn_structure():
    """Phase 8 regression. Four requirements and two constraints - six concrete items -
    clear only two section categories, and were written as unstructured prose while a
    single requirement in a complex task earned four headings."""
    from prompt_compiler.analyzer.models import PromptAnalysis
    from prompt_compiler.optimizer.sections import plan_sections

    analysis = PromptAnalysis(
        original_prompt="a prompt stating a great deal in two categories",
        task_type="code generation",
        primary_goal="Write the function.",
        complexity="simple",
        confidence=0.9,
        explicit_requirements=[
            "reverse the input string",
            "include type hints on every parameter",
            "raise ValueError on empty input",
            "support unicode characters throughout",
        ],
        constraints=["Python only", "no external libraries"],
    )

    plan = plan_sections(analysis)

    assert plan, "six stated items were written as prose"
    assert "REQUIREMENTS" in plan and "CONSTRAINTS" in plan


def test_the_volume_floor_does_not_scaffold_a_prompt_that_says_little():
    """The other half of the same rule. Raising the volume floor must not reintroduce the
    over-scaffolding the Phases 3-4 pass removed."""
    from prompt_compiler.analyzer.models import PromptAnalysis
    from prompt_compiler.optimizer.sections import plan_sections

    analysis = PromptAnalysis(
        original_prompt="thin",
        task_type="explanation",
        primary_goal="Explain it.",
        complexity="simple",
        confidence=0.9,
        explicit_requirements=["explain recursion"],
    )

    assert plan_sections(analysis) == []


def test_the_severe_inversion_is_gone_from_the_swept_space(swept):
    """No prompt in the grid states four or more items and is still written as prose."""
    buried = [p for p in swept if p.is_prose and p.stated_items >= 4]

    assert buried == []


def test_prose_inversions_are_counted_by_configuration_not_by_pair(swept):
    """The metric itself had a bug: pairing every prose point against every structured one
    is a cartesian product running to millions of rows that says nothing the distinct count
    does not say better."""
    inversions = prose_inversions(swept)

    assert len(inversions) <= len(swept)
    assert all(point.is_prose for point in inversions)


# ----------------------------------------------------------------------------- benchmark


PAYLOAD = AnalysisPayload(
    task_type="code generation",
    primary_goal="Write a Python function that reverses a string.",
    explicit_requirements=["reverse the input string"],
    constraints=["Python"],
    complexity="moderate",
    confidence=0.9,
)


class Fake(Provider):
    name = "fake"
    model = "fake-model"

    def __init__(self, error: Exception | None = None) -> None:
        self._error = error
        self.compiles = 0

    def structured(self, *, system: str, user: str, schema):
        if schema is AnalysisPayload:
            if self._error is not None:
                raise self._error
            self.compiles += 1
            return PAYLOAD
        return GeneratedPrompt(
            optimized_prompt="GOAL\nReverse the input string in Python."
        )


CASES = [
    Case(name="one", prompt="reverse a string"),
    Case(name="two", prompt="reverse a string with type hints"),
]


def test_a_benchmark_run_records_every_case():
    report = run(CASES, Fake())

    assert len(report.outcomes) == 2
    assert len(report.succeeded) == 2
    assert all(outcome.scores for outcome in report.succeeded)


def test_an_expected_failure_does_not_lose_the_cases_already_paid_for():
    """A rate limit on the last case must not discard the results already bought."""
    report = run(CASES, Fake(error=RateLimitError("slow down")))

    assert report.failed and not report.succeeded
    assert "slow down" in report.failed[0].error
    assert "0 of 2 cases compiled" in report.summary()


def test_an_unexpected_exception_still_propagates():
    """A bug in the pipeline is not a data point, and swallowing it would quietly turn a
    broken run into a report full of plausible numbers."""

    class Broken(Fake):
        def structured(self, *, system, user, schema):
            raise RuntimeError("this is a bug, not a provider failure")

    with pytest.raises(RuntimeError):
        run(CASES, Broken())


def test_provider_failures_are_recorded_as_data_not_raised():
    report = run(CASES, Fake(error=ProviderError("upstream is down")))

    assert [outcome.ok for outcome in report.outcomes] == [False, False]


def test_a_report_with_nothing_to_summarise_says_so_instead_of_raising():
    empty = BenchmarkReport(outcomes=[])

    assert empty.metrics().distributions == []
    assert "0 of 0" in empty.summary()


def test_a_mixed_provider_compile_is_visible_in_the_report():
    """Recorded rather than hidden: a case served partly by a fallback is not comparable
    with one served entirely by the primary."""
    outcome = Outcome(name="x", ok=True, models_used=["gemini-x", "groq-y"])

    assert outcome.mixed_providers
    assert not Outcome(name="x", ok=True, models_used=["gemini-x"]).mixed_providers


def test_a_missing_dimension_is_dropped_rather_than_filled_with_a_zero():
    """Phases 7-8 validation pass. Taking the union of score keys and defaulting a missing
    one to 0 invents a measurement, and a fabricated zero drags a mean down exactly like a
    real bad score - the worst way to be wrong in a report whose purpose is to be trusted."""
    report = BenchmarkReport(
        outcomes=[
            Outcome(name="a", ok=True, scores={"clarity": 10, "specificity": 20}),
            Outcome(name="b", ok=True, scores={"clarity": 90}),
        ]
    )

    scored = {shape.name for shape in report.metrics().distributions}

    assert "clarity" in scored
    assert "specificity" not in scored, "a dimension only one case reported was averaged in"


def test_an_empty_series_still_refuses_to_summarise_itself():
    """degeneracies() surfacing distribution()'s error is correct: the alternative is
    returning "no problems found" for data that does not exist."""
    with pytest.raises(ValueError):
        degeneracies({"nothing": []})


def test_correlation_is_skipped_when_there_is_only_one_case():
    """A correlation over a single point is not a weak result, it is not a result."""
    report = BenchmarkReport(outcomes=[Outcome(name="a", ok=True, scores={"clarity": 10})])

    assert report.metrics().findings == []


def test_a_report_carries_no_credentials():
    """Reports get written to disk. A key must never reach one."""
    report = run(CASES, Fake())

    assert "api_key" not in report.model_dump_json()
    assert "key" not in {field for outcome in report.outcomes for field in outcome.model_dump()}


# ----------------------------------------------------- phases 7 and 8 working together


def test_the_benchmark_records_the_same_dimensions_the_scorer_produces():
    """The integration seam between the two phases: Phase 8 reads Phase 7's report by
    dimension name, so a rename in one that is not made in the other would silently drop a
    dimension out of every benchmark instead of failing."""
    report = run(CASES, Fake())

    for outcome in report.succeeded:
        assert set(outcome.scores) == set(DIMENSION_NAMES)


def test_benchmark_metrics_survive_a_json_round_trip():
    """A report is written to disk and read back later; a run that cannot be stored is a
    run that has to be paid for twice."""
    report = run(CASES, Fake())

    restored = BenchmarkReport.model_validate_json(report.model_dump_json())

    assert [o.scores for o in restored.succeeded] == [o.scores for o in report.succeeded]
    assert restored.summary() == report.summary()


def test_the_sweep_and_the_pipeline_agree_on_what_earns_structure(swept):
    """Phase 8 measures the same policy the pipeline runs - not a reimplementation of it.
    Every structured point must clear one of the two floors the module actually applies."""
    from prompt_compiler.optimizer.sections import MIN_CONTENT_ITEMS, MIN_CONTENT_SECTIONS

    for point in swept:
        if point.is_prose:
            continue
        items = point.n_requirements + point.n_constraints + point.n_context
        # GOAL is added after the floor is cleared, so the content count excludes it.
        assert items >= MIN_CONTENT_ITEMS or point.sections - 1 >= MIN_CONTENT_SECTIONS
