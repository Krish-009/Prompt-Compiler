"""Provider selection: the one place that turns configuration into a provider object.

Phase 9 filled this seam. Adding Gemini and Groq was an adapter each plus an entry below;
nothing in the analysis, ambiguity, safety, optimization or scoring code changed, because
none of it can name a provider. That was the point of keeping the interface transport-only.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

from ..config import KNOWN_PROVIDERS, Settings
from ..errors import ConfigurationError, MissingCredentialsError
from .anthropic import AnthropicProvider
from .base import Provider
from .fallback import FallbackProvider
from .gemini import GeminiProvider
from .groq import GroqProvider


def _warn(message: str) -> None:
    print(message, file=sys.stderr)

#: Provider name -> constructor. Gemini is the V1 primary and Groq the fallback; Anthropic
#: stays as the reference implementation the contract suite was written against.
BUILDERS: dict[str, Callable[[Settings], Provider]] = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "anthropic": AnthropicProvider,
}


def build_one(settings: Settings) -> Provider:
    """Construct the single provider named by `settings`, with no fallback."""
    name = settings.provider
    if name not in KNOWN_PROVIDERS:
        raise ConfigurationError(
            f"Unknown provider {name!r}. Known providers: {', '.join(KNOWN_PROVIDERS)}."
        )
    builder = BUILDERS.get(name)
    if builder is None:
        raise ConfigurationError(
            f"Provider {name!r} is planned for V1 but is not implemented yet "
            f"(arriving in Phase 9). Implemented today: {', '.join(sorted(BUILDERS))}."
        )
    return builder(settings)


def build_provider(settings: Settings, notify: Callable[[str], None] | None = None) -> Provider:
    """Build the provider for this run, wrapped in a fallback when one is available.

    A fallback is only attached when it is configured, implemented, and has credentials.
    An unusable one is not worth failing the run over - the primary still works - but it is
    **said out loud**, because a fallback that cannot attach looks configured and protects
    nothing. That was not a theoretical worry: before per-provider model variables existed,
    the fallback inherited the primary's model id, failed to build on every single run, and
    was dropped here without a word.
    """
    primary = build_one(settings)

    name = settings.fallback_provider
    if not name or name == settings.provider or name not in BUILDERS:
        return primary

    warn = notify or _warn
    try:
        fallback = build_one(settings.for_provider(name))
    except (MissingCredentialsError, ConfigurationError) as exc:
        warn(f"{name} fallback not available ({exc}) — continuing with {primary.name} alone.")
        return primary

    return FallbackProvider(primary, fallback, notify=warn)
