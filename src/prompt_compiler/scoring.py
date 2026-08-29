"""Quality scoring: heuristic estimates, computed locally from evidence already gathered.

No LLM call. Every number here comes from the analysis, the compiled text, and the
safeguards - all of which the pipeline already holds by the time scoring runs.

**Two subjects, never one number.** The analysis describes the prompt the *user wrote*:
its ambiguities, its gaps, what it left unstated. The rewrite is a different artefact, and
what can be checked about it - did it carry the requirements forward, did the factuality
rules land, is it carrying waste - is checked against its own text. Averaging the two would
produce a single figure that means nothing: "your request was vague" and "the rewrite kept
your constraints" are not commensurable, and blending them would let a good rewrite of a
bad prompt hide either fact. So each dimension names its `subject`, the two groups score
separately, and there is deliberately no grand total.

**These are heuristics, not measurements.** Every threshold and weight below is reasoned
from the policies elsewhere in this package, not calibrated against outcomes: nothing here
has been checked against whether a higher-scoring prompt actually produces better answers.
That is Phase 8's work. Until then `QualityReport.heuristic` stays True, `summary()` says
so in words, and no score should be quoted as evidence that a transformation improved
anything.

**Size is not a dimension.** `token_efficiency` scores waste - redundancy, padding, input
noise carried forward - never length. Compiling a vague request into a precise one usually
costs tokens, and penalising that would smuggle "shorter is better" back into a project
that rejects it.

Clarity and ambiguity, listed separately in the phase plan, are one axis read in two
directions: both would be computed from the same `ambiguities` list. They are merged into
`clarity` rather than double-counted.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from .analyzer.models import Complexity, NonEmptyText, PromptAnalysis, Severity
from .optimizer.token_optimizer import find_redundancies
from .safety.hallucination import MAX_SAFEGUARDS, earned_rules
from .validation import is_present

DimensionName = Literal[
    "clarity",
    "specificity",
    "completeness",
    "requirement_coverage",
    "risk_coverage",
    "token_efficiency",
]

Subject = Literal["prompt", "rewrite"]
"""What a dimension describes: the prompt the user wrote, or the prompt this tool wrote."""

#: Bumped when the scoring method changes, so a stored score is recognisable as having
#: come from a different scorer. Mirrors `token_optimizer.COUNT_METHOD`.
SCORE_METHOD = "heuristic-v1"

#: Worst first, matching the rest of the package.
SEVERITIES: tuple[Severity, ...] = ("high", "medium", "low")

# --------------------------------------------------------------------------------- weights
#
# Reasoned, not measured. Each is anchored to a policy constant elsewhere in the package so
# that the two cannot drift apart silently; Phase 8 replaces the reasoning with evidence.

#: Deducted per ambiguity. Three high-severity ambiguities - the point at which
#: `clarification_questions()` stops asking, because more than that is an interrogation -
#: lands a prompt at 25, in the lowest band. A low-severity one is detail a competent
#: responder can choose sensibly, so it barely registers.
AMBIGUITY_PENALTY: dict[Severity, int] = {"high": 25, "medium": 10, "low": 3}

#: Deducted per entry in `missing_information`. That field means the responder cannot
#: proceed at all without the fact, so a quarter of the score per gap: one gap is a real
#: problem rather than a blemish, and four leave a prompt that cannot be answered at all.
#: Flat rather than complexity-scaled - a blocking gap blocks either way.
MISSING_INFORMATION_PENALTY = 25

#: How many stated items a prompt of each complexity carries when it is fully specific.
#: Anchored to `sections.MIN_CONTENT_SECTIONS`: three content sections is where structure
#: starts to pay, so a moderate request wants rather more than that, and a complex one more
#: again. A simple request is specific with very little.
SPECIFICITY_TARGET: dict[Complexity, int] = {"simple": 2, "moderate": 4, "complex": 6}

#: Deducted per redundancy still present in the compiled prompt. High, because `tighten()`
#: removes these unless it abandoned the pass, so finding one means something went wrong.
REDUNDANCY_PENALTY = 15

#: Deducted per filler phrase. Lower: the tightener reports filler and deliberately leaves
#: it alone, because cutting words out of a sentence is a rewrite and rewrites carry
#: meaning. It is still padding the generator should not have produced.
FILLER_PENALTY = 8

#: Deducted per item of `unnecessary_content` that survived into the rewrite. The weakest
#: signal here - presence is judged tolerantly, and a false positive costs 10 points - so
#: it is weighted to match its confidence.
UNNECESSARY_CONTENT_PENALTY = 10

#: Score floors for the coarse label. Bands, not the raw integer, are the resolution this
#: method actually supports; the integer is ordinal and exists so Phase 8 can rank.
BANDS: tuple[tuple[int, str], ...] = ((80, "strong"), (60, "good"), (40, "fair"), (0, "weak"))


def band(score: int) -> str:
    """The coarse label for a score. Read this rather than the number."""
    for floor, name in BANDS:
        if score >= floor:
            return name
    return BANDS[-1][1]


class Dimension(BaseModel):
    """One scored axis, carrying the evidence that produced it.

    `basis` is not decoration. A bare number is an assertion; a number that names what it
    was counted from can be checked, argued with, and recalibrated.
    """

    name: DimensionName
    subject: Subject
    score: int = Field(ge=0, le=100)
    basis: NonEmptyText

    @property
    def band(self) -> str:
        return band(self.score)


class QualityReport(BaseModel):
    """Six heuristic dimensions across two subjects. There is no overall figure."""

    dimensions: list[Dimension] = Field(default_factory=list)
    heuristic: bool = True
    """Always True. These are estimates from countable evidence, not measured outcomes."""
    method: str = SCORE_METHOD

    def of(self, name: DimensionName) -> Dimension:
        for dimension in self.dimensions:
            if dimension.name == name:
                return dimension
        raise KeyError(name)

    def about(self, subject: Subject) -> list[Dimension]:
        return [dimension for dimension in self.dimensions if dimension.subject == subject]

    def _mean(self, subject: Subject) -> int:
        scored = self.about(subject)
        if not scored:
            return 0
        # Equal weights, and deliberately so: any other weighting would claim to know which
        # dimension matters more, which nothing has established. Phase 8 may replace it.
        return round(sum(dimension.score for dimension in scored) / len(scored))

    @property
    def prompt_score(self) -> int:
        """How well specified the request was, before this tool touched it."""
        return self._mean("prompt")

    @property
    def rewrite_score(self) -> int:
        """How faithfully and economically the rewrite carried that request."""
        return self._mean("rewrite")

    def summary(self) -> str:
        return (
            f"prompt {self.prompt_score}/100 ({band(self.prompt_score)}), "
            f"rewrite {self.rewrite_score}/100 ({band(self.rewrite_score)}) "
            f"- heuristic estimates, not measurements"
        )


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _counted(items: Sequence[Severity]) -> str:
    counts = Counter(items)
    return ", ".join(f"{counts[level]} {level}" for level in SEVERITIES if counts[level])


# ------------------------------------------------------------------- the prompt as written


def _clarity(analysis: PromptAnalysis) -> Dimension:
    """How many ways the request can be read, weighted by how much the reading matters."""
    ambiguities = analysis.ambiguities
    penalty = sum(AMBIGUITY_PENALTY[item.severity] for item in ambiguities)
    basis = (
        "no ambiguity found"
        if not ambiguities
        else f"{_counted([item.severity for item in ambiguities])} severity "
        f"{'ambiguity' if len(ambiguities) == 1 else 'ambiguities'}"
    )
    return Dimension(
        name="clarity", subject="prompt", score=max(0, 100 - penalty), basis=basis
    )


def _specificity(analysis: PromptAnalysis) -> Dimension:
    """How much the request actually pins down, against what its complexity asks for.

    Counts stated items only - requirements, constraints, context, a named output shape.
    Assumptions and gaps are excluded on purpose: an inference is not something the user
    specified, and crediting it would reward the analysis for filling in blanks.
    """
    stated = (
        len(analysis.explicit_requirements)
        + len(analysis.constraints)
        + len(analysis.context)
        + (1 if analysis.expected_output else 0)
    )
    target = SPECIFICITY_TARGET[analysis.complexity]
    return Dimension(
        name="specificity",
        subject="prompt",
        score=min(100, round(100 * stated / target)),
        basis=f"{_plural(stated, 'stated item')}, against {target} "
        f"for a {analysis.complexity} request",
    )


def _completeness(analysis: PromptAnalysis) -> Dimension:
    """Whether anything needed is absent. Distinct from clarity by the taxonomy's own
    definition: a gap stops a responder proceeding, an ambiguity only misdirects them."""
    gaps = len(analysis.missing_information)
    return Dimension(
        name="completeness",
        subject="prompt",
        score=max(0, 100 - gaps * MISSING_INFORMATION_PENALTY),
        basis="nothing recorded as missing"
        if not gaps
        else f"{_plural(gaps, 'gap')} blocking the answer",
    )


# ----------------------------------------------------------------- the prompt this tool wrote


def _requirement_coverage(
    analysis: PromptAnalysis, unverified_requirements: Sequence[str]
) -> Dimension:
    """What proportion of the stated items survived the rewrite.

    Takes the finding rather than recomputing it, because `compile_prompt` checks against
    generation's raw output on purpose - so a requirement the model never wrote stays
    distinguishable from one the tightening pass removed.
    """
    stated = [*analysis.explicit_requirements, *analysis.constraints]
    if not stated:
        return Dimension(
            name="requirement_coverage",
            subject="rewrite",
            score=100,
            basis="no stated requirement or constraint to carry forward",
        )

    absent = set(unverified_requirements)
    missing = sum(1 for item in stated if item in absent)
    found = len(stated) - missing
    return Dimension(
        name="requirement_coverage",
        subject="rewrite",
        score=round(100 * found / len(stated)),
        basis=f"{found} of {_plural(len(stated), 'stated item')} found in the rewrite",
    )


def _risk_coverage(
    analysis: PromptAnalysis, safeguards: Sequence[str], optimized_prompt: str
) -> Dimension:
    """How much of the risk the analysis found is answered by a rule that actually landed.

    Two things can go wrong between a risk and a safeguard, and this catches both. The cap
    in `safeguards()` can cut an earned rule. And the generation model can simply not write
    a rule it was handed - which nothing else notices, because `tighten()` protects only
    text that is already present, so a rule that never arrived is missing from both sides
    of its before/after comparison and the guard passes trivially. That is the same blind
    spot `validation.py` was written to close, one layer over.
    """
    earned = earned_rules(analysis)
    if not earned:
        return Dimension(
            name="risk_coverage",
            subject="rewrite",
            score=100,
            basis="no actionable risk found, so no factuality rule was earned",
        )

    present = [rule for rule in safeguards if is_present(optimized_prompt, rule)]
    basis = (
        f"{len(present)} of {_plural(len(earned), 'earned factuality rule')} "
        f"present in the rewrite"
    )
    cut = len(earned) - len(safeguards)
    if cut > 0:
        basis += f" ({cut} cut by the cap of {MAX_SAFEGUARDS})"
    return Dimension(
        name="risk_coverage",
        subject="rewrite",
        # Clamped: the pipeline never passes more rules than were earned, but scoring is
        # public and Phase 8 will drive it directly. A caller that does must get a capped
        # score rather than a validation error - failing a good compile inside the scorer
        # would be a far worse outcome than an optimistic number.
        score=min(100, round(100 * len(present) / len(earned))),
        basis=basis,
    )


def _token_efficiency(analysis: PromptAnalysis, optimized_prompt: str) -> Dimension:
    """Tokens the rewrite spends on nothing.

    Measured on the compiled text directly rather than inferred from the tightening pass's
    report, which cannot distinguish "filler, deliberately left alone" from "duplicate the
    pass abandoned" - both come back marked non-removable. Waste is waste whatever the
    reason, and asking the text is both simpler and true by construction.

    Growth is not counted. A precise prompt is usually longer than the vague one it came
    from, and that is the transformation working, not a cost.
    """
    findings = find_redundancies(
        optimized_prompt, analysis.explicit_requirements, analysis.constraints
    )
    filler = [item for item in findings if item.kind == "filler_phrase"]
    structural = [item for item in findings if item.kind != "filler_phrase"]
    survived = [
        item for item in analysis.unnecessary_content if is_present(optimized_prompt, item)
    ]

    penalty = (
        REDUNDANCY_PENALTY * len(structural)
        + FILLER_PENALTY * len(filler)
        + UNNECESSARY_CONTENT_PENALTY * len(survived)
    )
    counted = [
        (len(structural), "unremoved redundancy", "unremoved redundancies"),
        (len(filler), "filler phrase", "filler phrases"),
        (len(survived), "unnecessary item carried forward", "unnecessary items carried forward"),
    ]
    parts = [
        f"{size} {singular if size == 1 else plural}"
        for size, singular, plural in counted
        if size
    ]
    return Dimension(
        name="token_efficiency",
        subject="rewrite",
        score=max(0, 100 - penalty),
        basis=", ".join(parts) if parts else "no redundancy or padding found in the rewrite",
    )


def score(
    analysis: PromptAnalysis,
    optimized_prompt: str,
    *,
    safeguards: Sequence[str] = (),
    unverified_requirements: Sequence[str] = (),
) -> QualityReport:
    """Score a compile. Local, deterministic, and honest about being a heuristic.

    Deliberately takes the pieces rather than a `CompiledPrompt`: scoring is testable
    without building a whole result, and `models.py` can depend on this module without a
    cycle.
    """
    return QualityReport(
        dimensions=[
            _clarity(analysis),
            _specificity(analysis),
            _completeness(analysis),
            _requirement_coverage(analysis, unverified_requirements),
            _risk_coverage(analysis, safeguards, optimized_prompt),
            _token_efficiency(analysis, optimized_prompt),
        ]
    )
