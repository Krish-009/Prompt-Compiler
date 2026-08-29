"""Hallucination-risk reduction: what the model looks for, and which safeguards a prompt
has actually earned.

The same division as the rest of the pipeline. Spotting where an answer could drift into
invention is language reasoning and belongs to the model. **Deciding which safeguards to
add is policy and belongs here**, in code - because the failure mode this subsystem exists
to prevent is a compiler that staples "Do not hallucinate." onto every prompt it sees.

A safeguard is only emitted when a specific risk earns it, it is phrased here rather than
by the model, and there is no generic catch-all. A prompt with nothing at risk gets no
factuality text at all.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..analyzer.models import HallucinationRisk, PromptAnalysis

#: Even a badly grounded prompt should not end in a wall of caveats.
MAX_SAFEGUARDS = 4

#: Risks at this level or worse earn a safeguard. A low risk is recorded, not acted on.
ACTIONABLE = frozenset({"high", "medium"})

MISSING_INFORMATION = (
    "If information you need is missing, say what is missing instead of filling it in."
)
STATE_ASSUMPTIONS = (
    "State any assumption you make, and mark it clearly as an assumption rather than "
    "a fact from the request."
)
NO_FABRICATION = (
    "Do not invent specifics that were not provided - names, figures, dates, citations, "
    "APIs or file contents. Say when you do not know."
)
RESOLVE_CONFLICT = (
    "Two requirements conflict. Point out the conflict and ask which takes precedence "
    "rather than silently choosing one."
)
NAME_THE_READING = (
    "Where the request can be read more than one way, say which reading you answered."
)
NO_UNPERFORMED_WORK = (
    "Do not claim to have run, tested or verified anything you did not actually do."
)

GUIDANCE = """\
- hallucination_risks: places where answering could produce something the prompt does not \
support. Each one has:
  - kind: one of unsupported_assumption (the answer would rest on something nothing in the \
prompt supports), missing_information (a needed fact is absent), ambiguous_reference (a \
"this", "the file", "our system" with no referent), contradictory_requirements (two stated \
requirements cannot both hold), unavailable_information (the answer needs material the \
responder cannot see, such as code or a document that was not attached), fabrication_prone \
(the answer invites specifics that are easy to invent - citations, version numbers, \
statistics, API signatures).
  - text: what could be invented, and why the prompt does not settle it.
  - grounding: how well the prompt supports it - "stated" (in the prompt), "inferred" (not \
stated but safely implied), "assumed" (an inference nothing in the prompt supports), \
"unknown" (absent entirely). Only "assumed" and "unknown" are risky; a stated or inferred \
item rarely belongs here at all.
  - severity: "high" if the answer would likely contain something untrue or unusable, \
"medium" if it would contain something unsupported but recoverable, "low" if it is a \
detail worth noting only.

Report what the prompt makes likely, not every way an answer could theoretically go wrong. \
Most well-specified prompts have no risks at all, and an empty list is the correct answer \
for them."""


def _actionable(risks: Iterable[HallucinationRisk]) -> list[HallucinationRisk]:
    return [risk for risk in risks if risk.severity in ACTIONABLE]


#: Worst first, matching Severity's own ordering.
_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}

#: A gap recorded in `missing_information` carries no severity of its own. Medium: worth a
#: rule, but it should yield to a risk the analysis actually rated high.
_UNRATED = "medium"


def _worst(severities: Iterable[str]) -> str | None:
    ranked = sorted(severities, key=lambda level: _SEVERITY_RANK[level])
    return ranked[0] if ranked else None


def earned_rules(analysis: PromptAnalysis) -> list[str]:
    """Every rule the analysis earns, worst-rated evidence first, *before* the cap.

    Split out from `safeguards()` so that what a prompt earned and what it was given stay
    separately visible: the gap between the two is real coverage loss, and Phase 7's
    scoring reports it rather than re-deriving the risk-to-rule mapping and letting the two
    copies drift.
    """
    risks = _actionable(analysis.hallucination_risks)

    def rated(*kinds: str, groundings: tuple[str, ...] = ()) -> list[str]:
        return [
            risk.severity
            for risk in risks
            if risk.kind in kinds or (groundings and risk.grounding in groundings)
        ]

    candidates: list[tuple[str, list[str]]] = [
        (
            RESOLVE_CONFLICT,
            rated("contradictory_requirements")
            + [
                item.severity
                for item in analysis.ambiguities
                if item.kind == "conflicting_instructions"
            ],
        ),
        (
            MISSING_INFORMATION,
            rated("missing_information", "unavailable_information")
            + ([_UNRATED] if analysis.missing_information else []),
        ),
        (STATE_ASSUMPTIONS, rated("unsupported_assumption", groundings=("assumed",))),
        (NO_FABRICATION, rated("fabrication_prone")),
        (
            NAME_THE_READING,
            rated("ambiguous_reference")
            + [item.severity for item in analysis.ambiguities if item.severity == "high"],
        ),
        # Only where the request implies doing something, so "verified" is a claim that
        # could be made falsely. An explanation cannot claim to have run anything.
        (
            NO_UNPERFORMED_WORK,
            [risk.severity for risk in risks]
            if analysis.task_type.strip().lower()
            in {"code generation", "debugging", "data analysis"}
            else [],
        ),
    ]

    earned = [
        (rule, _worst(severities)) for rule, severities in candidates if _worst(severities)
    ]
    earned.sort(key=lambda pair: _SEVERITY_RANK[pair[1]])
    return [rule for rule, _ in earned]


def safeguards(analysis: PromptAnalysis) -> list[str]:
    """The factuality rules this prompt has earned, worst-rated risk first, capped.

    Each rule answers a risk the analysis actually found. There is no default rule and no
    generic "do not hallucinate" - an unearned caveat spends tokens and teaches the reader
    to skim the ones that matter.

    When more rules are earned than the cap allows, the ones cut are those whose evidence
    the analysis rated least severe. Ordering by source position instead would drop a rule
    earned by a high-severity risk in favour of one earned by a low-severity ambiguity,
    which is backwards precisely when a prompt is in most trouble. Ties keep the declared
    order, so a genuine contradiction still leads a field of equals.
    """
    return earned_rules(analysis)[:MAX_SAFEGUARDS]


def needs_safeguards(analysis: PromptAnalysis) -> bool:
    return bool(safeguards(analysis))


def worst_risk(risks: Iterable[HallucinationRisk]) -> HallucinationRisk | None:
    order = {"high": 0, "medium": 1, "low": 2}
    ranked = sorted(risks, key=lambda risk: order[risk.severity])
    return ranked[0] if ranked else None
