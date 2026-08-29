"""Systematic sweep of the deterministic layer's input space.

**Why a sweep rather than a corpus.** The obvious way to evaluate the policy layer is to
run a set of realistic prompts through it. That needs analyses, analyses need a model, and
the only analyses available offline are twelve fixtures written by the same hand that wrote
the policies - so any conclusion drawn from them would largely confirm its own assumptions,
on a sample far too small to show a distribution.

Enumerating the input space instead answers a narrower question completely rather than a
broad one badly: *given every combination of evidence an analysis can carry, how does the
policy layer behave?* That catches the defects which are properties of the machinery
itself - a score that cannot move, two scores that are secretly one, a threshold that never
fires - and it catches them exhaustively, with no model and no sampling error.

**What it cannot do.** It says nothing about whether the model produces good analyses, nor
whether a higher-scoring prompt yields a better answer. Those need a live provider and wait
for Phase 9. Nothing here should be read as evidence that compiling a prompt helps.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import product
from typing import Literal

from pydantic import BaseModel, Field

from ..analyzer.models import (
    Ambiguity,
    Complexity,
    HallucinationRisk,
    PromptAnalysis,
    Severity,
)
from ..optimizer.sections import plan_sections
from ..safety.hallucination import MAX_SAFEGUARDS, earned_rules, safeguards
from ..scoring import DimensionName, score
from ..validation import unverified

Fidelity = Literal["faithful", "drops_requirement", "drops_safeguards", "wasteful"]
"""How well the simulated rewrite carried the analysis. An axis rather than a constant:
with a perfect rewrite every time, the three rewrite-side dimensions would sit at 100 and
look dead when the truth is that nothing was varying their input."""

#: Real-ish wording rather than "item 1", so the presence and redundancy checks behave the
#: way they will on genuine text instead of on strings that defeat their stemming.
_REQUIREMENTS = (
    "reverse the input string",
    "include type hints on every parameter",
    "raise ValueError on empty input",
    "support unicode characters throughout",
)
_CONSTRAINTS = ("Python only", "no external libraries")
_CONTEXT = ("the script runs on the user's own laptop",)

AMBIGUITY_PROFILES: dict[str, tuple[Severity, ...]] = {
    "none": (),
    "one_low": ("low",),
    "one_medium": ("medium",),
    "one_high": ("high",),
    "two_high_one_medium": ("high", "high", "medium"),
}

#: (kind, grounding, severity). Chosen to reach every branch of the safeguard policy,
#: including enough simultaneous rules to make the cap bind.
RISK_PROFILES: dict[str, tuple[tuple[str, str, Severity], ...]] = {
    "none": (),
    "one_medium": (("missing_information", "unknown", "medium"),),
    "many_high": (
        ("fabrication_prone", "unknown", "high"),
        ("unsupported_assumption", "assumed", "high"),
        ("contradictory_requirements", "stated", "high"),
        ("ambiguous_reference", "assumed", "high"),
    ),
}

#: "code generation" is one of the task types that can earn NO_UNPERFORMED_WORK, which is
#: the sixth rule and therefore the one that makes a cap of four reachable at all.
TASK_TYPES = ("explanation", "code generation")


class Axes(BaseModel):
    """The grid to enumerate. Every field is a list of values to cross with the others."""

    task_type: tuple[str, ...] = TASK_TYPES
    complexity: tuple[Complexity, ...] = ("simple", "moderate", "complex")
    requirements: tuple[int, ...] = (0, 1, 2, 4)
    constraints: tuple[int, ...] = (0, 2)
    context: tuple[int, ...] = (0, 1)
    expected_output: tuple[bool, ...] = (False, True)
    missing: tuple[int, ...] = (0, 1, 2, 4)
    ambiguity: tuple[str, ...] = tuple(AMBIGUITY_PROFILES)
    risk: tuple[str, ...] = tuple(RISK_PROFILES)
    fidelity: tuple[Fidelity, ...] = (
        "faithful",
        "drops_requirement",
        "drops_safeguards",
        "wasteful",
    )

    def size(self) -> int:
        total = 1
        for values in self.model_dump().values():
            total *= len(values)
        return total


class SweepPoint(BaseModel):
    """One combination of evidence, and everything the deterministic layer did with it."""

    task_type: str
    complexity: Complexity
    n_requirements: int
    n_constraints: int
    n_context: int
    expected_output: bool
    n_missing: int
    ambiguity: str
    risk: str
    fidelity: Fidelity

    stated_items: int
    """Requirements + constraints + context + a named output shape - what the user said."""
    sections: int
    is_prose: bool
    earned: int
    emitted: int
    scores: dict[str, int] = Field(default_factory=dict)

    @property
    def cap_bound(self) -> bool:
        return self.earned > self.emitted


def _analysis(
    *,
    task_type: str,
    complexity: Complexity,
    requirements: int,
    constraints: int,
    context: int,
    expected_output: bool,
    missing: int,
    ambiguity: str,
    risk: str,
) -> PromptAnalysis:
    return PromptAnalysis(
        original_prompt="a synthetic prompt for the sweep",
        task_type=task_type,
        primary_goal="Produce the thing that was asked for.",
        complexity=complexity,
        confidence=0.8,
        explicit_requirements=list(_REQUIREMENTS[:requirements]),
        constraints=list(_CONSTRAINTS[:constraints]),
        context=list(_CONTEXT[:context]),
        expected_output="a single file" if expected_output else "",
        missing_information=[f"missing fact number {n}" for n in range(missing)],
        ambiguities=[
            Ambiguity(
                kind="unclear_scope",
                text=f"an ambiguity of {severity} severity, number {index}",
                severity=severity,
                clarifying_question="Which reading did you mean?"
                if severity == "high"
                else "",
            )
            for index, severity in enumerate(AMBIGUITY_PROFILES[ambiguity])
        ],
        hallucination_risks=[
            HallucinationRisk(
                kind=kind, text=f"a risk about {kind}", grounding=grounding, severity=severity
            )
            for kind, grounding, severity in RISK_PROFILES[risk]
        ],
    )


def _rewrite(analysis: PromptAnalysis, rules: Sequence[str], fidelity: Fidelity) -> str:
    """A stand-in for what the generation model would return, at a chosen fidelity."""
    stated = [*analysis.explicit_requirements, *analysis.constraints]
    carried = stated[:-1] if fidelity == "drops_requirement" and stated else stated

    lines = ["GOAL", analysis.primary_goal]
    if carried:
        lines += ["", "REQUIREMENTS", *(f"- {item}" for item in carried)]
    if analysis.context:
        lines += ["", "CONTEXT", *(f"- {item}" for item in analysis.context)]
    if fidelity != "drops_safeguards" and rules:
        lines += ["", "FACTUALITY RULES", *rules]
    if fidelity == "wasteful":
        # A duplicated block and some padding: waste tighten() would have caught, standing
        # in for a compile whose tightening pass was abandoned.
        lines += ["", *(f"- {item}" for item in carried)]
        lines += ["", "In order to proceed, it is important to note that you should begin."]
    return "\n".join(lines)


def points(axes: Axes | None = None) -> Iterator[SweepPoint]:
    """Every combination in `axes`, run through the deterministic layer."""
    grid = axes or Axes()
    for (
        task_type,
        complexity,
        requirements,
        constraints,
        context,
        expected_output,
        missing,
        ambiguity,
        risk,
        fidelity,
    ) in product(
        grid.task_type,
        grid.complexity,
        grid.requirements,
        grid.constraints,
        grid.context,
        grid.expected_output,
        grid.missing,
        grid.ambiguity,
        grid.risk,
        grid.fidelity,
    ):
        analysis = _analysis(
            task_type=task_type,
            complexity=complexity,
            requirements=requirements,
            constraints=constraints,
            context=context,
            expected_output=expected_output,
            missing=missing,
            ambiguity=ambiguity,
            risk=risk,
        )
        rules = safeguards(analysis)
        text = _rewrite(analysis, rules, fidelity)
        stated = [*analysis.explicit_requirements, *analysis.constraints]
        report = score(
            analysis,
            text,
            safeguards=rules,
            unverified_requirements=unverified(text, stated),
        )
        plan = plan_sections(analysis)

        yield SweepPoint(
            task_type=task_type,
            complexity=complexity,
            n_requirements=requirements,
            n_constraints=constraints,
            n_context=context,
            expected_output=expected_output,
            n_missing=missing,
            ambiguity=ambiguity,
            risk=risk,
            fidelity=fidelity,
            stated_items=requirements + constraints + context + int(expected_output),
            sections=len(plan),
            is_prose=not plan,
            earned=len(earned_rules(analysis)),
            emitted=len(rules),
            scores={
                dimension.name: dimension.score for dimension in report.dimensions
            },
        )


DIMENSION_NAMES: tuple[DimensionName, ...] = (
    "clarity",
    "specificity",
    "completeness",
    "requirement_coverage",
    "risk_coverage",
    "token_efficiency",
)


def series(collected: Sequence[SweepPoint]) -> dict[str, list[float]]:
    """The scored dimensions as parallel series, ready for the metrics checks."""
    return {
        name: [float(point.scores[name]) for point in collected]
        for name in DIMENSION_NAMES
    }


def cap_pressure(collected: Sequence[SweepPoint]) -> list[bool]:
    """Where the safeguard cap actually discarded an earned rule."""
    return [point.cap_bound for point in collected]


def prose_inversions(collected: Sequence[SweepPoint]) -> list[SweepPoint]:
    """Prose points that stated *more* than the thinnest point which earned headings.

    The deferred question from the adversarial review: `plan_sections` counts section
    categories, not the volume of content in them, so a prompt with four requirements and
    two constraints clears fewer categories than one with a single item spread across
    three. If that inversion is reachable, it is here.

    Reported as the offending prose configurations rather than as every offending pair:
    pairing is a cartesian product that runs to millions of rows and says nothing the count
    of distinct configurations does not say better.
    """
    structured = [point for point in collected if not point.is_prose]
    if not structured:
        return []
    thinnest = min(point.stated_items for point in structured)

    seen: set[tuple] = set()
    inversions: list[SweepPoint] = []
    for point in collected:
        if not point.is_prose or point.stated_items <= thinnest:
            continue
        key = (
            point.complexity,
            point.n_requirements,
            point.n_constraints,
            point.n_context,
            point.expected_output,
        )
        if key not in seen:
            seen.add(key)
            inversions.append(point)
    return inversions
