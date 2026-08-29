"""Exception hierarchy. Everything the CLI can recover from derives from PromptCompilerError."""

from __future__ import annotations


class PromptCompilerError(Exception):
    """Base class for expected failures. The CLI reports these without a traceback."""


class InvalidInputError(PromptCompilerError):
    """The user's input cannot be compiled (empty prompt, for example)."""


class ConfigurationError(PromptCompilerError):
    """Configuration was present but unusable (a non-numeric limit, for example)."""


class MissingCredentialsError(PromptCompilerError):
    """No usable API credentials were available."""


class ProviderError(PromptCompilerError):
    """The provider was reachable in principle but the call failed."""


class RateLimitError(ProviderError):
    """The provider refused the call because of its own rate limits.

    A distinct type because it is the clearest signal to try a different provider:
    unlike a network failure it will not resolve on an immediate retry.
    """


class ProviderTimeoutError(ProviderError):
    """The provider did not answer within the configured timeout."""


class InvalidResponseError(PromptCompilerError):
    """The provider replied, but not with usable structured output."""
