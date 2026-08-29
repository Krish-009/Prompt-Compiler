"""Summary statistics for a set of measurements, and the checks that read them.

Deliberately stdlib-only. Pulling in numpy to take a mean would add a compiled dependency
to a laptop application for arithmetic Python already does, and `statistics.correlation`
has been in the standard library since 3.10 - well inside this project's floor of 3.12.

Nothing here decides whether a number is *good*. These functions report shape - spread,
constancy, how tightly two series move together - and the findings they raise are about the
**instrument**, not about prompt quality: a dimension that cannot vary is measuring nothing,
and two dimensions that move together perfectly are one dimension counted twice. Those are
defects detectable without any knowledge of outcomes, which is exactly what can be
established offline.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

FindingKind = Literal[
    "constant",
    "narrow_range",
    "redundant_pair",
    "never_binds",
    "always_binds",
]

#: A dimension using less than this fraction of its 0-100 range across the whole sweep is
#: reported. Not a failure - a dimension can be legitimately hard to move - but a scale
#: whose ends are unreachable is a scale with fewer usable values than it advertises.
NARROW_RANGE = 0.5

#: Above this, two series are close enough to being the same measurement that keeping both
#: double-weights whatever they share. Chosen high on purpose: dimensions drawn from
#: related evidence *should* correlate, and only near-identity is a defect.
REDUNDANT_CORRELATION = 0.95


class Distribution(BaseModel):
    """What a set of measurements looks like. No judgement, just shape."""

    name: str
    count: int
    minimum: float
    maximum: float
    mean: float
    median: float
    distinct: int

    @property
    def spread(self) -> float:
        return self.maximum - self.minimum

    @property
    def is_constant(self) -> bool:
        return self.distinct <= 1

    def summary(self) -> str:
        return (
            f"{self.name}: {self.minimum:g}..{self.maximum:g} "
            f"(mean {self.mean:.1f}, median {self.median:g}, "
            f"{self.distinct} distinct over {self.count})"
        )


class Finding(BaseModel):
    """Something about the instrument that a person should look at."""

    kind: FindingKind
    subject: str
    detail: str

    def summary(self) -> str:
        return f"[{self.kind}] {self.subject}: {self.detail}"


def distribution(name: str, values: Sequence[float]) -> Distribution:
    if not values:
        raise ValueError(f"{name}: nothing to summarise")
    return Distribution(
        name=name,
        count=len(values),
        minimum=min(values),
        maximum=max(values),
        mean=statistics.fmean(values),
        median=statistics.median(values),
        distinct=len(set(values)),
    )


def correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Pearson correlation, or None when either series is constant.

    None rather than 0.0: a constant series has no linear relationship to report, which is
    a different statement from "no relationship was found", and conflating the two would
    quietly turn a dead dimension into an apparently well-behaved independent one.
    """
    if len(left) != len(right):
        raise ValueError("series must be the same length")
    if len(left) < 2:
        return None
    try:
        return statistics.correlation(left, right)
    except statistics.StatisticsError:
        return None


def degeneracies(series: dict[str, Sequence[float]], scale: float = 100.0) -> list[Finding]:
    """Instrument defects: dimensions that cannot vary, and pairs that are one dimension.

    `scale` is the range a dimension is supposed to span, used only to judge whether an
    observed spread is narrow. Pass the real scale or this reports nonsense.
    """
    findings: list[Finding] = []
    names = sorted(series)

    for name in names:
        shape = distribution(name, series[name])
        if shape.is_constant:
            findings.append(
                Finding(
                    kind="constant",
                    subject=name,
                    detail=f"never left {shape.minimum:g} across {shape.count} cases, so it "
                    f"distinguishes nothing and its weight in any mean is dead",
                )
            )
        elif shape.spread < scale * NARROW_RANGE:
            findings.append(
                Finding(
                    kind="narrow_range",
                    subject=name,
                    detail=f"only spanned {shape.minimum:g}..{shape.maximum:g} of a "
                    f"0..{scale:g} scale",
                )
            )

    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            value = correlation(series[first], series[second])
            if value is not None and abs(value) >= REDUNDANT_CORRELATION:
                findings.append(
                    Finding(
                        kind="redundant_pair",
                        subject=f"{first} / {second}",
                        detail=f"correlate at {value:+.3f}; keeping both counts the same "
                        f"evidence twice",
                    )
                )

    return findings


def binding_rate(name: str, hits: Sequence[bool]) -> Finding | None:
    """Whether a threshold earns its place: one that never fires, or always fires, is not
    a threshold. Returns None when it genuinely discriminates."""
    if not hits:
        return None
    rate = sum(hits) / len(hits)
    if rate == 0.0:
        return Finding(
            kind="never_binds",
            subject=name,
            detail=f"did not fire in any of {len(hits)} cases",
        )
    if rate == 1.0:
        return Finding(
            kind="always_binds",
            subject=name,
            detail=f"fired in all {len(hits)} cases",
        )
    return None


class MetricSet(BaseModel):
    """Distributions plus whatever the checks found, ready to print or store."""

    distributions: list[Distribution] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)

    def report(self) -> str:
        lines = [shape.summary() for shape in self.distributions]
        if self.findings:
            lines.append("")
            lines.extend(finding.summary() for finding in self.findings)
        return "\n".join(lines)
