"""Structured representation of a prompt.

The three-way split is the point of this module:

* **Explicit** - the user stated it (`explicit_requirements`, `constraints`, `context`,
  `expected_output`).
* **Reasonable inference** - not stated, but a competent responder would assume it
  (`assumptions`, each carrying the words in the prompt that support it).
* **Missing** - needed and not safely inferable (`missing_information`), never filled in.

Later phases rely on it: only explicit items may become requirements in a generated
prompt, inferences must be surfaced as assumptions, and gaps must be asked about
rather than invented.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field, StringConstraints

Complexity = Literal["simple", "moderate", "complex"]


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _clean_items(value: Any) -> Any:
    """Strip list entries and drop blank ones.

    A blank entry is noise from the model rather than a semantic error, so it is
    dropped instead of failing the whole analysis - unlike a blank scalar field,
    which is meaningless and is rejected.
    """
    if isinstance(value, list):
        return [item.strip() for item in value if not isinstance(item, str) or item.strip()]
    return value


Text = Annotated[str, BeforeValidator(_strip)]
NonEmptyText = Annotated[str, BeforeValidator(_strip), StringConstraints(min_length=1)]
TextList = Annotated[list[NonEmptyText], BeforeValidator(_clean_items)]


AmbiguityKind = Literal[
    "vague_terminology",
    "undefined_terminology",
    "missing_requirement",
    "missing_technical_constraint",
    "conflicting_instructions",
    "unclear_scope",
    "unclear_output_format",
    "unclear_audience",
]
Severity = Literal["high", "medium", "low"]


class Ambiguity(BaseModel):
    """A place where the prompt can be read more than one way.

    `severity` is what separates a prompt that needs a question from one that merely has
    loose edges; the policy that acts on it lives in `ambiguity.py`, not here.
    """

    kind: AmbiguityKind
    text: NonEmptyText
    severity: Severity
    clarifying_question: Text = ""
    """Supplied for high-severity ambiguities only. Empty is normal."""


RiskKind = Literal[
    "unsupported_assumption",
    "missing_information",
    "ambiguous_reference",
    "contradictory_requirements",
    "unavailable_information",
    "fabrication_prone",
]

Grounding = Literal["stated", "inferred", "assumed", "unknown"]
"""How well the prompt supports a piece of information.

`stated` and `inferred` are safe - they correspond to the explicit fields and to
`assumptions`, which carry a basis. `assumed` is an inference nothing in the prompt
supports, and `unknown` is absent entirely; those two are where fabrication starts.
"""


class HallucinationRisk(BaseModel):
    """A place where answering could produce something the prompt does not support."""

    kind: RiskKind
    text: NonEmptyText
    grounding: Grounding
    severity: Severity


class Assumption(BaseModel):
    """A reasonable inference, tied to what in the prompt justifies it.

    `basis` exists to keep the category honest: an inference with nothing behind it is
    missing information, not an assumption. Both fields are required to be non-blank -
    that constraint is the whole mechanism, so it is enforced rather than requested.
    """

    text: NonEmptyText
    basis: NonEmptyText


class AnalysisPayload(BaseModel):
    """The part of the analysis the model produces.

    The original prompt is deliberately absent: making the model echo it back would
    spend output tokens on text we already hold, and give it an opportunity to alter
    the user's words. `analyze()` attaches it afterwards.
    """

    task_type: NonEmptyText
    primary_goal: NonEmptyText
    secondary_goals: TextList = Field(default_factory=list)
    context: TextList = Field(default_factory=list)
    explicit_requirements: TextList = Field(default_factory=list)
    constraints: TextList = Field(default_factory=list)
    expected_output: Text = ""
    assumptions: list[Assumption] = Field(default_factory=list)
    missing_information: TextList = Field(default_factory=list)
    ambiguities: list[Ambiguity] = Field(default_factory=list)
    hallucination_risks: list[HallucinationRisk] = Field(default_factory=list)
    unnecessary_content: TextList = Field(default_factory=list)
    complexity: Complexity
    confidence: float = Field(ge=0.0, le=1.0)
    """The model's own estimate of how well this analysis captures the prompt.
    A self-report, not a calibrated probability."""


class PromptAnalysis(AnalysisPayload):
    """A payload bound to the prompt it describes."""

    original_prompt: NonEmptyText
