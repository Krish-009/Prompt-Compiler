"""Evaluation and benchmarking.

Two entry points, separated by what they can honestly claim.

`sweep` enumerates the deterministic layer's whole input space with no model at all, and
answers questions about the machinery: can each score actually move, are two of them
secretly one, does a threshold ever fire. Runnable now, exhaustive, no sampling error.

`benchmark` runs real prompts end to end through a real provider, and is the only thing
that can answer whether compiling a prompt helps. It is built and tested against a fake,
but no live provider exists until Phase 9, so it has never produced a finding.

Importing this package must not import a provider - the same rule the rest of the core
follows.
"""

from .metrics import Distribution, Finding, MetricSet, correlation, degeneracies, distribution
from .sweep import Axes, SweepPoint, points, prose_inversions, series

__all__ = [
    "Axes",
    "Distribution",
    "Finding",
    "MetricSet",
    "SweepPoint",
    "correlation",
    "degeneracies",
    "distribution",
    "points",
    "prose_inversions",
    "series",
]

# benchmark.py is deliberately not re-exported here. It reaches the provider layer through
# compile_prompt, so importing this package would pull that in for callers who only want
# the offline sweep - and the sweep's whole point is that it needs no provider at all.
