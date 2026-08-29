"""Groq provider - the V1 fallback.

Verified against the installed groq 1.7.0. `import groq` below resolves to the installed
SDK, not this module: Python 3 imports are absolute unless written as relative.

**No parse helper.** Unlike the Anthropic SDK there is no `messages.parse` and no
`beta.chat.completions.parse`, so structured output is a JSON schema on the request and a
pydantic validation on the reply, done here. That validation is not a formality: it is the
only thing standing between a malformed reply and a traceback, because every constraint
this project cares about - a non-blank goal, a bounded confidence, a closed complexity set -
travels to the API as a hint and is enforced only on the way back.

**Retries are switched off deliberately.** The project's rule is that a rate limit falls
back rather than sleeping; the SDK's default of two retries would silently reintroduce the
wait this is a fallback in order to avoid. Groq being the fallback, there is nowhere left
to fall back *to*, which makes failing fast and saying so the only honest option.
"""

from __future__ import annotations

from typing import Any

import groq
from pydantic import ValidationError

from ..config import Settings
from ..errors import (
    InvalidResponseError,
    MissingCredentialsError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from .base import Provider, SchemaT, detail

#: Groq's error code for a reply that did not satisfy the requested JSON schema. In
#: practice this means the generation ran out of room, not that the schema was wrong.
_TRUNCATED_JSON = "json_validate_failed"


class GroqProvider(Provider):
    name = "groq"

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        if client is None:
            client = groq.Groq(
                api_key=settings.require_api_key(),
                timeout=settings.timeout_seconds,
                max_retries=0,  # fall back, never sleep - see the module docstring
            )
        self.model = settings.require_model()
        self._client = client

    def structured(self, *, system: str, user: str, schema: type[SchemaT]) -> SchemaT:
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                max_completion_tokens=self._settings.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": schema.model_json_schema(),
                    },
                },
            )
        # Specific before general: the first four are subclasses of APIStatusError, which
        # is itself a subclass of APIError.
        except groq.AuthenticationError as exc:
            raise MissingCredentialsError(
                "The Groq API rejected the configured key."
            ) from exc
        except groq.PermissionDeniedError as exc:
            raise MissingCredentialsError(
                "The configured key lacks permission for this model."
            ) from exc
        except groq.RateLimitError as exc:
            raise RateLimitError(f"Rate limited by the Groq API. {detail(exc)}") from exc
        except groq.APITimeoutError as exc:
            raise ProviderTimeoutError(
                f"Groq did not respond within {self._settings.timeout_seconds:g}s."
            ) from exc
        except groq.APIConnectionError as exc:
            raise ProviderError("Could not reach the Groq API.") from exc
        except groq.APIStatusError as exc:
            explanation = detail(exc)
            if _TRUNCATED_JSON in explanation:
                # Groq validates JSON server-side, so a reply cut off mid-object comes back
                # as a 400 "Failed to validate JSON" rather than reaching us with
                # finish_reason "length". Without this the truncation check below never
                # fires for Groq, and the user is told their prompt is malformed when the
                # real problem is a token budget they can raise.
                raise InvalidResponseError(
                    "Groq could not produce valid JSON, which for this schema almost "
                    "always means the reply was cut off. Raise PROMPT_COMPILER_MAX_TOKENS "
                    "and retry."
                ) from exc
            # A 413 here is the per-minute token limit, and the body names both the limit
            # and what was requested - the only two numbers that say how to fix it.
            raise ProviderError(
                f"Groq API error (HTTP {exc.status_code}). {explanation}"
            ) from exc
        except groq.APIError as exc:
            raise ProviderError(f"The Groq API call failed. {detail(exc)}") from exc

        return self._parsed(completion, schema)

    @staticmethod
    def _parsed(completion: Any, schema: type[SchemaT]) -> SchemaT:
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise InvalidResponseError("The model did not return usable structured output.")

        choice = choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise InvalidResponseError(
                "The model's reply was cut off by max_tokens. "
                "Raise PROMPT_COMPILER_MAX_TOKENS and retry."
            )

        content = getattr(getattr(choice, "message", None), "content", None)
        if not content or not content.strip():
            raise InvalidResponseError("The model did not return usable structured output.")

        try:
            return schema.model_validate_json(content)
        except ValidationError as exc:
            # Covers both malformed JSON and JSON that satisfies the wire types but not
            # this project's constraints - neither may escape as a traceback.
            raise InvalidResponseError(
                f"The model's reply did not match the expected shape: {exc.error_count()} "
                f"field(s) invalid."
            ) from exc
