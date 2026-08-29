"""End-to-end benchmarking against a real provider.

This is the only part of the system that can answer whether compiling a prompt actually
helps. It has never done so: no provider adapter exists for either configured key until
Phase 9, so the harness is built and tested against a fake and has produced no finding.
Nothing here should be cited as evidence about quality.

**It makes real calls and costs money.** Two per prompt, the same two `compile_prompt`
always makes. It is a foreground tool a person runs deliberately - never invoked from the
CLI's normal path, never on a timer, never speculatively.

**One failure must not lose the run.** A rate limit on case nine of twelve would otherwise
discard the eight results already paid for, so an expected failure is recorded against its
case and the run continues. Unexpected exceptions still propagate: a bug in the pipeline is
not a data point.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from ..compiler import compile_prompt
from ..errors import PromptCompilerError
from ..providers.base import Provider
from .metrics import MetricSet, degeneracies, distribution


class Case(BaseModel):
    """A prompt to benchmark."""

    name: str
    prompt: str


class Outcome(BaseModel):
    """What one case produced. Deliberately holds no provider object and no credentials -
    a report is written to disk, and a key must never reach it."""

    name: str
    ok: bool
    error: str = ""

    original_tokens: int = 0
    optimized_tokens: int = 0
    sections: int = 0
    is_prose: bool = False
    safeguards: int = 0
    unverified: int = 0
    scores: dict[str, int] = Field(default_factory=dict)
    models_used: list[str] = Field(default_factory=list)

    @property
    def mixed_providers(self) -> bool:
        """More than one model served this compile, so a fallback fired mid-case."""
        return len(set(self.models_used)) > 1


class BenchmarkReport(BaseModel):
    outcomes: list[Outcome] = Field(default_factory=list)

    @property
    def succeeded(self) -> list[Outcome]:
        return [outcome for outcome in self.outcomes if outcome.ok]

    @property
    def failed(self) -> list[Outcome]:
        return [outcome for outcome in self.outcomes if not outcome.ok]

    def metrics(self) -> MetricSet:
        """Distributions over the cases that completed, plus instrument checks.

        Returns an empty set rather than raising when nothing succeeded: a run where every
        case failed is a fact worth reporting, not an error to swallow the report over.
        """
        done = self.succeeded
        if not done:
            return MetricSet()

        # Only dimensions every outcome actually reported. Taking the union and filling a
        # missing one with 0 would invent a measurement - and a fabricated zero drags a
        # mean down exactly like a real bad score, which is the worst way to be wrong in a
        # file whose whole purpose is to be trusted.
        shared = set.intersection(*(set(outcome.scores) for outcome in done))
        series: dict[str, list[float]] = {
            name: [float(outcome.scores[name]) for outcome in done]
            for name in sorted(shared)
        }
        shapes = [distribution(name, values) for name, values in series.items()]
        shapes.append(
            distribution("optimized_tokens", [float(o.optimized_tokens) for o in done])
        )
        shapes.append(
            distribution("unverified_items", [float(o.unverified) for o in done])
        )
        return MetricSet(
            distributions=shapes,
            findings=degeneracies(series) if len(done) > 1 else [],
        )

    def summary(self) -> str:
        lines = [f"{len(self.succeeded)} of {len(self.outcomes)} cases compiled"]
        if self.failed:
            lines.extend(f"  failed: {o.name}: {o.error}" for o in self.failed)
        mixed = [o.name for o in self.succeeded if o.mixed_providers]
        if mixed:
            lines.append(f"  fallback fired during: {', '.join(mixed)}")
        return "\n".join(lines)


def run(cases: Sequence[Case], provider: Provider) -> BenchmarkReport:
    """Compile every case, recording what each produced. Two live calls per case."""
    outcomes: list[Outcome] = []
    for case in cases:
        try:
            result = compile_prompt(case.prompt, provider)
        except PromptCompilerError as exc:
            # Expected failures - a rate limit, a bad key, an unusable reply. Recorded so
            # the cases already paid for survive. Anything else is a bug and propagates.
            outcomes.append(Outcome(name=case.name, ok=False, error=str(exc)))
            continue

        outcomes.append(
            Outcome(
                name=case.name,
                ok=True,
                original_tokens=result.tokens.original_tokens,
                optimized_tokens=result.tokens.optimized_tokens,
                sections=len(result.sections),
                is_prose=not result.sections,
                safeguards=len(result.safeguards),
                unverified=len(result.unverified_requirements),
                scores={d.name: d.score for d in result.quality.dimensions},
                models_used=list(result.models_used),
            )
        )
    return BenchmarkReport(outcomes=outcomes)
