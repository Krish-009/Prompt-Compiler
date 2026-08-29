"""Post-generation validation: did the rewrite actually carry the analysis forward?

Until this module existed, nothing checked that `generate()`'s output contained the
requirements and constraints the analysis found. The structural half of the pipeline was
deterministic - code decides the sections and the safeguards - but the *content* half was
left entirely to the model's compliance with an instruction, unverified.

**Why this is not a literal check.** Requirement text comes from the analysis model;
the prompt text comes from the generation model, which is asked to rewrite rather than
transcribe. "reverse the input string" legitimately becomes "reverses the given string".
A substring test would flag every good paraphrase, and a check that cries wolf gets
ignored - so presence is judged on content-word overlap, tolerant of wording and
inflection, and strict about a requirement that simply is not there.

**Why it reports rather than raises.** The same reasoning: this cannot distinguish
"dropped" from "reworded past recognition" with certainty, and failing a whole compile on
a heuristic would trade a rare silent omission for a common false failure. The finding is
surfaced on `CompiledPrompt` and printed by the CLI, where a person can judge it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from .optimizer.token_optimizer import _contains

#: Words that carry no identifying content, so their absence proves nothing.
STOPWORDS = frozenset(
    """a an and are as at be by for from has have in into is it its of on or that the
    to with will should must can may do does using use when where which who whom this
    these those been being was were not no than then there their them they you your""".split()
)

#: Enough of an item's content words must appear for it to count as present. Two thirds
#: tolerates a dropped adjective or a reworded connector without tolerating a whole
#: requirement going missing.
PRESENCE_THRESHOLD = 0.66

#: Compared on a prefix so "reverse"/"reverses"/"reversing" match, without a stemmer.
_STEM_LENGTH = 5

_WORD_RE = re.compile(r"\w+")


def _content_words(text: str) -> list[str]:
    return [
        word.lower()
        for word in _WORD_RE.findall(text)
        if len(word) > 2 and word.lower() not in STOPWORDS
    ]


def _stems(words: Iterable[str]) -> set[str]:
    return {word[:_STEM_LENGTH] for word in words}


def is_present(text: str, item: str) -> bool:
    """Whether `item` appears in `text`, allowing for rewording."""
    if _contains(text, item):
        return True

    wanted = _content_words(item)
    if not wanted:
        return True  # nothing identifying to look for

    available = _stems(_content_words(text))
    found = sum(1 for word in wanted if word[:_STEM_LENGTH] in available)
    return found / len(wanted) >= PRESENCE_THRESHOLD


def unverified(text: str, items: Sequence[str]) -> list[str]:
    """Which of `items` cannot be found in `text`, in the order given.

    Named for what it can actually claim: these could not be verified as present, which
    is strong evidence they were dropped but is not a proof.
    """
    return [item for item in items if not is_present(text, item)]
