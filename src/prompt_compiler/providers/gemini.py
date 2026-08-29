"""Gemini provider - the V1 primary.

Verified against the installed google-genai 2.20.0.

**The schema risk carried since Phase 6 is closed.** The concern was that Gemini would
reject the `$defs`/`$ref` a nested pydantic model produces, and that the adapter would have
to inline the schema by hand. It does not: `_transformers.t_schema` flattens the model into
Gemini's own Schema type with every reference resolved, and it keeps the enums, the numeric
bounds and the string lengths. `AnalysisPayload` converts cleanly.

**Its error surface is unlike the other two.** There is no `RateLimitError`, no
`APITimeoutError`, no `AuthenticationError` - only `ClientError` and `ServerError`, both
carrying an HTTP `code`, so the mapping is by status rather than by exception type. Worse,
the SDK does not wrap transport failures at all: a timeout or a dropped connection arrives
as a raw `httpx` exception and would escape as a traceback if this module did not catch it.

**Retries are switched off deliberately.** The project's rule is that a rate limit falls
back to the other provider rather than sleeping - blocking an interactive CLI in a backoff
loop is worse than switching and saying so. The SDK retries by default, which would silently
reintroduce exactly that wait.
"""

from __future__ import annotations

from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
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

#: HTTP statuses that mean the key is the problem rather than the request.
_CREDENTIAL_CODES = frozenset({401, 403})

#: Finish reasons that are the model declining rather than answering.
_REFUSAL_REASONS = frozenset({"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"})


def _reason(value: Any) -> str:
    """Finish reasons arrive as an enum in production and as a plain string from a stub."""
    return getattr(value, "name", None) or str(value or "")


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        if client is None:
            # Credentials before configuration: a missing key is the commoner mistake and
            # the more useful thing to be told about first.
            client = genai.Client(
                api_key=settings.require_api_key(),
                http_options=genai_types.HttpOptions(
                    # Milliseconds here, unlike every other SDK in this project.
                    timeout=int(settings.timeout_seconds * 1000),
                    # One attempt, no retries - see the module docstring.
                    retry_options=genai_types.HttpRetryOptions(attempts=1),
                ),
            )
        self.model = settings.require_model()
        self._client = client

    def structured(self, *, system: str, user: str, schema: type[SchemaT]) -> SchemaT:
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=user,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=self._settings.max_tokens,
                    response_mime_type="application/json",
                    response_schema=schema,
                    # This application never calls tools. Left on, the SDK writes a
                    # paragraph of advice to stderr on every single call, which in a CLI
                    # lands in the middle of the user's output.
                    automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )
        # ClientError and ServerError both subclass APIError, so specific comes first.
        except genai_errors.ClientError as exc:
            raise self._from_status(exc) from exc
        except genai_errors.ServerError as exc:
            raise ProviderError(f"Gemini API error (HTTP {exc.code}). {detail(exc)}") from exc
        except genai_errors.APIError as exc:
            raise ProviderError(f"The Gemini API call failed. {detail(exc)}") from exc
        # The SDK does not wrap transport failures; without these they escape as tracebacks.
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Gemini did not respond within {self._settings.timeout_seconds:g}s."
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderError("Could not reach the Gemini API.") from exc
        except ValidationError as exc:
            raise InvalidResponseError(
                f"The model's reply did not match the expected shape: {exc.error_count()} "
                f"field(s) invalid."
            ) from exc

        return self._parsed(response, schema)

    @staticmethod
    def _from_status(exc: genai_errors.ClientError) -> Exception:
        if exc.code == 429:
            return RateLimitError(f"Rate limited by the Gemini API. {detail(exc)}")
        if exc.code in _CREDENTIAL_CODES:
            return MissingCredentialsError(
                "The Gemini API rejected the configured key, or it lacks permission "
                f"for this model. {detail(exc)}"
            )
        # A 404 here is nearly always a retired or misspelled model id, and the API says
        # which one to use instead - so the detail matters more than the status.
        return ProviderError(f"Gemini API error (HTTP {exc.code}). {detail(exc)}")

    def _parsed(self, response: Any, schema: type[SchemaT]) -> SchemaT:
        blocked = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
        if blocked:
            raise InvalidResponseError(
                f"Gemini declined to process this prompt ({_reason(blocked)})."
            )

        candidates = getattr(response, "candidates", None) or []
        finish = _reason(getattr(candidates[0], "finish_reason", None)) if candidates else ""
        if finish == "MAX_TOKENS":
            raise InvalidResponseError(
                "The model's reply was cut off by max_tokens. "
                "Raise PROMPT_COMPILER_MAX_TOKENS and retry."
            )
        if finish in _REFUSAL_REASONS:
            raise InvalidResponseError(f"The model declined to answer ({finish}).")

        parsed = getattr(response, "parsed", None)
        if parsed is None:
            raise InvalidResponseError("The model did not return usable structured output.")
        if not isinstance(parsed, schema):
            # The SDK returns a dict rather than the model when it cannot parse cleanly.
            try:
                return schema.model_validate(parsed)
            except ValidationError as exc:
                raise InvalidResponseError(
                    f"The model's reply did not match the expected shape: "
                    f"{exc.error_count()} field(s) invalid."
                ) from exc
        return parsed
