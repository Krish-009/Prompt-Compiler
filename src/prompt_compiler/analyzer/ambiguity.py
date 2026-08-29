"""Ambiguity detection: the taxonomy the model classifies against, and the policy that
decides which ambiguities are worth interrupting the user over.

Detection is judgement and belongs to the model. **Whether to ask a question is policy and
belongs here**, in code: it is deterministic, testable without the API, and it is where the
"do not ask unnecessary questions" rule is actually enforced. "Explain recursion." produces
no questions because no ambiguity in it is severe enough, not because a model felt restrained.
"""

from __future__ import annotations

from collections.abc import Iterable

from .models import Ambiguity, Severity

#: Worst first. Used for ordering and for reporting the overall level.
SEVERITY_ORDER: dict[Severity, int] = {"high": 0, "medium": 1, "low": 2}

#: Even a badly underspecified prompt should not turn into an interrogation.
MAX_QUESTIONS = 3

DETECTION_GUIDANCE = """\
- ambiguities: places where the prompt can be read more than one way, or is pinned down too \
loosely to act on. Each one has:
  - kind: one of vague_terminology (a word doing too much work), undefined_terminology (a term \
specific to the user's world, never defined), missing_requirement (a needed behaviour is named \
but not specified), missing_technical_constraint (language, platform, version or library left \
open where it matters), conflicting_instructions (two parts of the prompt cannot both be \
satisfied), unclear_scope (how much work is being asked for is not bounded), \
unclear_output_format (the shape of the answer is not determined), unclear_audience (who the \
answer is for changes how it should be written).
  - text: the phrase at issue and what it could mean instead.
  - severity: "high" if the two readings lead to materially different deliverables, so \
answering the wrong one wastes the work; "medium" if a wrong reading still produces something \
useful but costs real rework; "low" if the readings differ only in detail a competent \
responder can choose sensibly and note.
  - clarifying_question: for high severity only, the single question that would settle it. \
Leave it empty otherwise. One question per ambiguity, never a list.

Record each issue once. If a responder could proceed but might proceed in the wrong \
direction, it is an ambiguity. If they could not proceed at all without the fact, it is \
missing_information instead. Most prompts have no high-severity ambiguity at all."""


def worst_severity(ambiguities: Iterable[Ambiguity]) -> Severity | None:
    """The most severe level present, or None when there is nothing to report."""
    levels = [item.severity for item in ambiguities]
    if not levels:
        return None
    return min(levels, key=lambda level: SEVERITY_ORDER[level])


def by_severity(ambiguities: Iterable[Ambiguity]) -> list[Ambiguity]:
    """Worst first, preserving the model's order within a level."""
    return sorted(ambiguities, key=lambda item: SEVERITY_ORDER[item.severity])


def needs_clarification(ambiguities: Iterable[Ambiguity]) -> bool:
    return any(item.severity == "high" for item in ambiguities)


def clarification_questions(
    ambiguities: Iterable[Ambiguity], limit: int = MAX_QUESTIONS
) -> list[str]:
    """The questions actually worth asking.

    Only high-severity ambiguities earn one, at most `limit` of them, and only where the
    model supplied a question - an unanswered high-severity ambiguity is still reported
    as an ambiguity, but no question is invented for it.
    """
    questions = [
        item.clarifying_question
        for item in by_severity(ambiguities)
        if item.severity == "high" and item.clarifying_question
    ]
    return questions[:limit]
