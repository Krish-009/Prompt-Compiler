"""How much structure a prompt actually needs.

Every section must be *earned by evidence in the analysis*, never included by default.
The policy lives in code for the same reason the question policy does: "a simple request
produces a short prompt" is a rule, and a rule that a model is merely asked to follow is
not a rule.

ROLE is deliberately never planned. A role line the user did not ask for is exactly the
boilerplate the project forbids, and a role the user *did* specify already arrives as a
constraint. It stays in `Section` so a later phase - an explicit `--mode` in Phase 10 -
can add one on purpose rather than by habit.
"""

from __future__ import annotations

from typing import Literal, get_args

from ..analyzer.models import PromptAnalysis
from ..safety.hallucination import needs_safeguards

Section = Literal[
    "ROLE",
    "GOAL",
    "CONTEXT",
    "REQUIREMENTS",
    "CONSTRAINTS",
    "PROCESS",
    "OUTPUT FORMAT",
    "FACTUALITY RULES",
]

#: The order sections appear in a generated prompt.
SECTION_ORDER: tuple[Section, ...] = get_args(Section)

#: Below this many content sections, headings cost more than they carry and the prompt is
#: written as plain prose instead. Three is the point where a reader benefits from
#: scanning rather than reading: two facts are a sentence, three are a list.
MIN_CONTENT_SECTIONS = 3

#: A prompt can also earn structure on volume alone, whatever its category count.
#:
#: The category floor asks how many *kinds* of guidance a prompt carries, which is the
#: wrong question when it carries many items of one kind. Measured across the policy's
#: whole input space at Phase 8: four requirements and two constraints clear only two
#: categories and were written as unstructured prose, while a single requirement in a
#: complex task cleared three and earned four headings. Six concrete items are a list by
#: this module's own reasoning, so counting only the categories they fall into was
#: measuring the wrong thing.
#:
#: Four rather than three, deliberately. The category floor can already trigger on three
#: items, so three here would make the volume rule subsume it and flip a large block of
#: modestly-specified prompts into headings at once - reintroducing the over-scaffolding
#: the Phases 3-4 pass removed. Four is the smallest value that closes the inversion
#: without that, which the sweep confirms.
MIN_CONTENT_ITEMS = 4


def plan_sections(analysis: PromptAnalysis) -> list[Section]:
    """Choose the sections this prompt needs, or an empty plan meaning "plain prose"."""
    # A prompt that states no requirement, constraint or context has nothing to organise.
    # Its gaps would otherwise earn it OUTPUT FORMAT and FACTUALITY RULES, so the emptiest
    # prompts would come out the most heavily scaffolded - headings over nothing.
    if not (analysis.explicit_requirements or analysis.constraints or analysis.context):
        return []

    content: set[Section] = set()

    if analysis.context:
        content.add("CONTEXT")
    if analysis.explicit_requirements:
        content.add("REQUIREMENTS")
    if analysis.constraints:
        content.add("CONSTRAINTS")
    if analysis.complexity == "complex" and analysis.explicit_requirements:
        # Multi-component work is the only case where saying how to proceed earns its
        # space - and only once there are stated requirements to order.
        content.add("PROCESS")
    if analysis.expected_output or any(
        item.kind == "unclear_output_format" for item in analysis.ambiguities
    ):
        content.add("OUTPUT FORMAT")
    # Earned by the safeguard policy rather than by raw evidence: if no safeguard is
    # warranted there is nothing to put under the heading. The policy already folds in
    # missing information and high-severity ambiguities from Phase 3.
    if needs_safeguards(analysis):
        content.add("FACTUALITY RULES")

    # Either floor earns structure: enough kinds of guidance to be worth scanning, or
    # enough individual items that a paragraph would bury them. Only stated list content
    # counts towards the second - expected_output is a single line, and FACTUALITY RULES
    # is guidance this tool added rather than content the user supplied.
    stated_items = (
        len(analysis.explicit_requirements)
        + len(analysis.constraints)
        + len(analysis.context)
    )
    if len(content) < MIN_CONTENT_SECTIONS and stated_items < MIN_CONTENT_ITEMS:
        return []

    content.add("GOAL")
    return [section for section in SECTION_ORDER if section in content]


def describe_plan(sections: list[Section]) -> str:
    """One line naming the structure, for the model and for the CLI."""
    if not sections:
        return "none - write a short prompt in plain prose, with no headings"
    return ", ".join(sections)
