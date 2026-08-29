"""Token measurement and redundancy removal - local, deterministic, no LLM call.

Two deliberate limits shape this module.

**It counts, it does not guess at meaning.** Token counting is a local estimate with a
stated method, not a call to a provider's counting endpoint: a network round trip per
compile would break the one-call-per-prompt budget for a number that only needs to be
approximately right. Every count is labelled `estimated`.

**It removes only what it can prove is safe.** Deleting a duplicate line cannot change
meaning; rewriting a verbose sentence can. So duplicates and formatting noise are removed,
verbosity is *reported* and left alone, and every removal is checked against the
requirements and constraints the analysis found - if one would be lost, nothing is removed.
A shorter prompt is not automatically a better one.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from math import ceil
from typing import Literal

from pydantic import BaseModel, Field

RedundancyKind = Literal[
    "duplicate_line",
    "repeated_requirement",
    "repeated_constraint",
    "filler_phrase",
    "excess_whitespace",
    "needless_formatting",
]

#: Multi-word padding that adds tokens without adding meaning. Reported, never removed
#: automatically: cutting words out of a sentence is a rewrite, and rewrites carry meaning.
FILLER_PHRASES = (
    "it is important to note that",
    "it should be noted that",
    "please note that",
    "as previously mentioned",
    "at this point in time",
    "due to the fact that",
    "for the purpose of",
    "in order to",
    "needless to say",
    "it goes without saying",
)

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
#: Leading/trailing punctuation is cosmetic; punctuation *inside* a line is not.
_EDGE_PUNCT = re.compile(r"^[^\w]+|[^\w]+$")

#: Bumped when the counting method changes, so a stored measurement can be recognised as
#: having come from a different estimator.
COUNT_METHOD = "local-heuristic-v1"


def count_tokens(text: str) -> int:
    """Estimate the token count of `text`.

    Method: split into word and punctuation pieces; each piece costs one token per four
    characters, rounded up. Punctuation always matches as a single character, so it costs
    one. This tracks byte-pair encoders closely enough for comparing two prompts, which is
    all it is used for. It is not exact, and nothing here pretends otherwise - see
    `TokenReport.estimated`.

    The length rule replaces an `isalnum()` test that silently misclassified every
    snake_case identifier as punctuation - `str.isalnum()` is False for any string
    containing an underscore - and charged a 21-character name one token instead of six.
    That biased the estimate hardest on code prompts, where identifiers are dense.
    """
    return sum(max(1, ceil(len(piece) / 4)) for piece in _TOKEN_RE.findall(text))


class Redundancy(BaseModel):
    kind: RedundancyKind
    text: str
    removable: bool
    """True only where removal provably cannot change meaning."""


class TokenReport(BaseModel):
    """Before and after, in estimated tokens.

    `tokens_saved` is negative when the compiled prompt is longer than the original, which
    is the common case: making a vague request precise usually costs tokens. The number
    worth watching is `redundancy_removed`, which is what this module actually cut.
    """

    original_tokens: int
    optimized_tokens: int
    redundancy_removed: int = 0
    estimated: bool = True
    method: str = COUNT_METHOD
    findings: list[Redundancy] = Field(default_factory=list)

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.optimized_tokens

    @property
    def percent_change(self) -> float:
        """Negative means the compiled prompt is shorter than the original."""
        if self.original_tokens == 0:
            return 0.0
        return round(
            (self.optimized_tokens - self.original_tokens) / self.original_tokens * 100, 1
        )

    def summary(self) -> str:
        direction = "shorter" if self.tokens_saved > 0 else "longer"
        if self.tokens_saved == 0:
            direction = "the same length"
        return (
            f"{self.original_tokens} -> {self.optimized_tokens} tokens "
            f"({abs(self.percent_change):.1f}% {direction}, estimated); "
            f"{self.redundancy_removed} removed as redundant"
        )


def _normalise(line: str) -> str:
    """A form in which two lines that say the same thing compare equal.

    Only bullet markers, edge punctuation, case and whitespace are ignored - the
    differences that are genuinely cosmetic. Internal punctuation is kept, because it
    carries meaning: deleting it made "5.5" and "55", or "-5" and "5", normalise
    identically, so two genuinely different constraints looked like one duplicate.

    That mattered far more than a cosmetic nicety, because every duplicate test *and*
    every preservation check funnels through this one function: the collision deleted a
    real constraint and simultaneously blinded the guard meant to catch the deletion.
    """
    without_bullet = _BULLET_RE.sub("", line)
    collapsed = " ".join(without_bullet.split())
    return _EDGE_PUNCT.sub("", collapsed).lower()


def _contains(haystack: str, needle: str) -> bool:
    return _normalise(needle) in _normalise(haystack)


def find_redundancies(
    text: str,
    requirements: Sequence[str] = (),
    constraints: Sequence[str] = (),
) -> list[Redundancy]:
    """Report every redundancy found, whether or not it is safe to remove."""
    findings: list[Redundancy] = []
    lines = text.splitlines()

    seen: dict[str, str] = {}
    for line in lines:
        key = _normalise(line)
        if not key:
            continue
        if key in seen:
            kind: RedundancyKind = "duplicate_line"
            if any(_normalise(item) == key for item in requirements):
                kind = "repeated_requirement"
            elif any(_normalise(item) == key for item in constraints):
                kind = "repeated_constraint"
            findings.append(Redundancy(kind=kind, text=line.strip(), removable=True))
        else:
            seen[key] = line

    lowered = text.lower()
    for phrase in FILLER_PHRASES:
        if phrase in lowered:
            findings.append(Redundancy(kind="filler_phrase", text=phrase, removable=False))

    if re.search(r"\n{3,}", text):
        findings.append(
            Redundancy(kind="excess_whitespace", text="blank line runs", removable=True)
        )
    if re.search(r"[ \t]+\n", text):
        findings.append(
            Redundancy(kind="excess_whitespace", text="trailing whitespace", removable=True)
        )
    if any(_BULLET_RE.match(line) and not _normalise(line) for line in lines):
        findings.append(Redundancy(kind="needless_formatting", text="empty bullet", removable=True))

    return findings


def _apply(text: str) -> str:
    kept: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        key = _normalise(line)
        if _BULLET_RE.match(line) and not key:
            continue  # an empty bullet carries nothing
        if key:
            if key in seen:
                continue
            seen.add(key)
        kept.append(line.rstrip())

    collapsed = re.sub(r"\n{3,}", "\n\n", "\n".join(kept))
    return collapsed.strip()


def tighten(
    text: str,
    requirements: Sequence[str] = (),
    constraints: Sequence[str] = (),
    protected: Sequence[str] = (),
) -> tuple[str, list[Redundancy]]:
    """Remove what is provably safe to remove; report the rest.

    The pass is abandoned - text returned untouched - if it would drop a stated
    requirement, constraint or other protected line, or if it would leave nothing at all.
    Preserving meaning outranks saving tokens, so the failure mode is "no saving", never
    "quietly weaker prompt" and never "quietly empty prompt".
    """
    findings = find_redundancies(text, requirements, constraints)
    if not any(item.removable for item in findings):
        return text, findings

    tightened = _apply(text)

    def abandon() -> tuple[str, list[Redundancy]]:
        return text, [item.model_copy(update={"removable": False}) for item in findings]

    if text.strip() and not tightened.strip():
        # Removing formatting from a prompt that was almost entirely formatting must not
        # leave an empty prompt: generate() guards its own output, but this pass runs after.
        return abandon()

    for stated in (*requirements, *constraints, *protected):
        if _contains(text, stated) and not _contains(tightened, stated):
            return abandon()

    return tightened, findings


def missing_from(text: str, required: Iterable[str]) -> list[str]:
    """Which of `required` no longer appear in `text`. The preservation check."""
    return [item for item in required if not _contains(text, item)]


def measure(
    original: str,
    optimized: str,
    *,
    before_tightening: str | None = None,
    findings: Sequence[Redundancy] = (),
) -> TokenReport:
    removed = 0
    if before_tightening is not None:
        removed = max(0, count_tokens(before_tightening) - count_tokens(optimized))
    return TokenReport(
        original_tokens=count_tokens(original),
        optimized_tokens=count_tokens(optimized),
        redundancy_removed=removed,
        findings=list(findings),
    )
