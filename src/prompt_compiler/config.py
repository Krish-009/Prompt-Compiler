"""Runtime configuration, read from the environment or a .env file.

Nothing here names a provider's API. `PROVIDER_KEY_VARS` is the single place an
environment variable is tied to a provider, and `DEFAULT_MODELS` the single place a model
name appears - so adding a provider is two dictionary entries plus an adapter.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from .errors import ConfigurationError, MissingCredentialsError

#: Every provider V1 knows about. All three are implemented as of Phase 9.
KNOWN_PROVIDERS = ("gemini", "groq", "anthropic")

#: Gemini is the V1 primary, as of Phase 9. It pointed at "anthropic" until the adapter
#: existed, because a default naming an unimplemented provider breaks every run.
DEFAULT_PROVIDER = "gemini"

#: Used when the primary provider fails. None disables fallback.
DEFAULT_FALLBACK_PROVIDER = "groq"

PROVIDER_KEY_VARS: dict[str, str] = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

#: A provider with no entry requires PROMPT_COMPILER_MODEL or --model. Deliberate: a
#: guessed model id is worse than an explicit error, it could silently select a paid
#: model, and provider catalogues change.
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-5",
}


def model_variable(provider: str) -> str:
    """The per-provider model variable, e.g. PROMPT_COMPILER_GROQ_MODEL.

    Model ids are not portable - "gemini-3.6-flash" means nothing to Groq - so the generic
    `PROMPT_COMPILER_MODEL` can only ever name a model for the provider the user actually
    asked for. Without a per-provider variable there was no way to give the fallback a
    model of its own: it inherited the primary's id, failed to build, and `build_provider`
    quietly returned the bare primary. A fallback that can never attach is worse than no
    fallback, because it looks configured.
    """
    return f"PROMPT_COMPILER_{provider.upper()}_MODEL"

DEFAULT_MAX_TOKENS = 16_000
DEFAULT_TIMEOUT_SECONDS = 120.0

NumberT = TypeVar("NumberT", int, float)


def _number_from_env(name: str, default: NumberT, cast: Callable[[str], NumberT]) -> NumberT:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}.") from exc


class Settings(BaseModel):
    """Configuration for one run, against one provider.

    `api_key` is excluded from the model's repr so it cannot leak into a traceback,
    a log line, or a debug print.
    """

    provider: str = DEFAULT_PROVIDER
    api_key: str | None = Field(default=None, repr=False)
    model: str | None = None
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, gt=0)
    timeout_seconds: float = Field(default=DEFAULT_TIMEOUT_SECONDS, gt=0)
    fallback_provider: str | None = DEFAULT_FALLBACK_PROVIDER

    @classmethod
    def from_env(cls, provider: str | None = None, *, load_env_file: bool = True) -> "Settings":
        """Build settings from the environment.

        Bad configuration raises ConfigurationError, which the CLI reports as a message
        rather than a traceback.
        """
        if load_env_file:
            load_dotenv(override=False)

        name = provider or os.environ.get("PROMPT_COMPILER_PROVIDER") or DEFAULT_PROVIDER
        if name not in KNOWN_PROVIDERS:
            raise ConfigurationError(
                f"Unknown provider {name!r}. Known providers: {', '.join(KNOWN_PROVIDERS)}."
            )

        fallback = os.environ.get("PROMPT_COMPILER_FALLBACK", DEFAULT_FALLBACK_PROVIDER)
        if fallback in ("", "none", "off"):
            fallback = None
        elif fallback not in KNOWN_PROVIDERS:
            raise ConfigurationError(
                f"Unknown fallback provider {fallback!r}. "
                f"Known providers: {', '.join(KNOWN_PROVIDERS)}, or 'none'."
            )

        try:
            return cls(
                provider=name,
                api_key=os.environ.get(PROVIDER_KEY_VARS[name]) or None,
                model=(
                    os.environ.get(model_variable(name))
                    or os.environ.get("PROMPT_COMPILER_MODEL")
                    or DEFAULT_MODELS.get(name)
                ),
                max_tokens=_number_from_env("PROMPT_COMPILER_MAX_TOKENS", DEFAULT_MAX_TOKENS, int),
                timeout_seconds=_number_from_env(
                    "PROMPT_COMPILER_TIMEOUT", DEFAULT_TIMEOUT_SECONDS, float
                ),
                fallback_provider=None if fallback == name else fallback,
            )
        except ValidationError as exc:
            raise ConfigurationError(f"Invalid configuration: {exc.errors()[0]['msg']}.") from exc

    @property
    def key_variable(self) -> str:
        return PROVIDER_KEY_VARS.get(self.provider, "PROMPT_COMPILER_API_KEY")

    def for_provider(self, name: str, *, load_env_file: bool = False) -> "Settings":
        """The same run, aimed at a different provider - used to build the fallback.

        The model deliberately does *not* carry over, and neither does the generic
        `PROMPT_COMPILER_MODEL`: both name a model for the provider the user asked for, and
        handing "gemini-3.6-flash" to Groq buys a 404 instead of a fallback. Only this
        provider's own variable, or its built-in default, applies here.
        """
        return Settings.from_env(name, load_env_file=load_env_file).model_copy(
            update={
                "model": os.environ.get(model_variable(name)) or DEFAULT_MODELS.get(name),
                "max_tokens": self.max_tokens,
                "timeout_seconds": self.timeout_seconds,
                "fallback_provider": None,
            }
        )

    def require_api_key(self) -> str:
        if not self.api_key or not self.api_key.strip():
            raise MissingCredentialsError(
                f"No API key found for provider {self.provider!r}. Set {self.key_variable} "
                "in your environment or in a .env file next to the project "
                "(see .env.example)."
            )
        return self.api_key

    def require_model(self) -> str:
        if not self.model:
            raise ConfigurationError(
                f"No model configured for provider {self.provider!r}. "
                "Set PROMPT_COMPILER_MODEL or pass --model."
            )
        return self.model
