"""Fallback between two providers, out loud.

The rule is that a switch is never silent: the user is told, on stderr, the moment it
happens. Silent degradation is how you end up unable to explain why an answer changed.

What earns a fallback is a provider that failed to deliver - rate limited, unreachable,
timed out, erroring, or answering with something unusable. What does not is a problem the
fallback would hit too, or one the user must fix: a missing key, bad configuration, or an
invalid prompt.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

from ..errors import InvalidResponseError, ProviderError
from .base import Provider, SchemaT

#: Failures that mean "this provider did not deliver" rather than "you configured it wrong".
FALLBACK_TRIGGERS = (ProviderError, InvalidResponseError)


def _warn(message: str) -> None:
    print(message, file=sys.stderr)


class FallbackProvider(Provider):
    """Tries `primary`, then `fallback`, announcing any switch."""

    def __init__(
        self,
        primary: Provider,
        fallback: Provider,
        notify: Callable[[str], None] = _warn,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self._notify = notify
        self.name = f"{primary.name}+{fallback.name}"
        self.model = primary.model
        self.used: Provider = primary
        """Whichever provider actually answered last. Recorded so a caller can report it."""
        self.models_used: list[str] = []
        """Every model that served a call, in order, without repeats.

        Each call decides primary-vs-fallback independently, which maximises use of the
        primary but means a transient failure on one call can leave a single compile built
        from two backends. Recording both keeps that observable instead of flattening it
        into whichever provider happened to answer last."""

    def structured(self, *, system: str, user: str, schema: type[SchemaT]) -> SchemaT:
        try:
            result = self.primary.structured(system=system, user=user, schema=schema)
        except FALLBACK_TRIGGERS as exc:
            self._notify(
                f"{self.primary.name} unavailable ({exc}) — using {self.fallback.name} fallback."
            )
            self.used = self.fallback
            self.model = self.fallback.model
            self._record(self.fallback.model)
            return self.fallback.structured(system=system, user=user, schema=schema)

        self.used = self.primary
        self.model = self.primary.model
        self._record(self.primary.model)
        return result

    def _record(self, model: str) -> None:
        if model not in self.models_used:
            self.models_used.append(model)
