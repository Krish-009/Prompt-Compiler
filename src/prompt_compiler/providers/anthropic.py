"""Anthropic provider.

`import anthropic` below resolves to the installed SDK, not this module: Python 3
imports are absolute unless written as relative.
"""

from __future__ import annotations

from typing import Any

import anthropic
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


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        if client is None:
            # Credentials before configuration: a missing key is the commoner mistake and
            # the more useful thing to be told about first.
            client = anthropic.Anthropic(
                api_key=settings.require_api_key(),
                timeout=settings.timeout_seconds,
            )
        self.model = settings.require_model()
        self._client = client

    def structured(self, *, system: str, user: str, schema: type[SchemaT]) -> SchemaT:
        try:
            message = self._client.messages.parse(
                model=self.model,
                max_tokens=self._settings.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
            )
        # Specific before general: the first four are subclasses of APIStatusError,
        # which is itself a subclass of APIError.
        except anthropic.AuthenticationError as exc:
            raise MissingCredentialsError(
                "The Anthropic API rejected the configured key."
            ) from exc
        except anthropic.PermissionDeniedError as exc:
            raise MissingCredentialsError(
                "The configured key lacks permission for this model."
            ) from exc
        except anthropic.RateLimitError as exc:
            raise RateLimitError(f"Rate limited by the Anthropic API. {detail(exc)}") from exc
        except anthropic.APITimeoutError as exc:
            raise ProviderTimeoutError(
                f"The Anthropic API did not respond within {self._settings.timeout_seconds:g}s."
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError("Could not reach the Anthropic API.") from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(
                f"Anthropic API error (HTTP {exc.status_code}). {detail(exc)}"
            ) from exc
        except anthropic.APIError as exc:
            raise ProviderError(f"The Anthropic API call failed. {detail(exc)}") from exc
        except ValidationError as exc:
            # The SDK validates the reply with pydantic. Constraints such as min_length
            # travel to the API only as a description hint, so the model can return JSON
            # that satisfies the schema's types but not our model - that must not escape
            # as a traceback.
            raise InvalidResponseError(
                f"The model's reply did not match the expected shape: {exc.error_count()} "
                f"field(s) invalid."
            ) from exc

        stop_reason = getattr(message, "stop_reason", None)
        if stop_reason == "refusal":
            raise InvalidResponseError("The model declined to process this prompt.")
        if stop_reason == "max_tokens":
            raise InvalidResponseError(
                "The model's reply was cut off by max_tokens. "
                "Raise PROMPT_COMPILER_MAX_TOKENS and retry."
            )

        parsed = getattr(message, "parsed_output", None)
        if parsed is None:
            raise InvalidResponseError("The model did not return usable structured output.")
        return parsed
