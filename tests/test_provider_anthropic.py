"""The Anthropic provider, exercised against a stub client. No network, no key."""

from __future__ import annotations

from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from prompt_compiler.analyzer.models import AnalysisPayload
from prompt_compiler.config import Settings
from prompt_compiler.errors import (
    InvalidResponseError,
    MissingCredentialsError,
    ProviderError,
)
from prompt_compiler.providers.anthropic import AnthropicProvider

ANALYSIS = AnalysisPayload(
    task_type="explanation",
    primary_goal="Explain recursion.",
    complexity="simple",
    confidence=0.9,
)


class _StubMessages:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.kwargs: dict = {}

    def parse(self, **kwargs):
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.result


class _StubClient:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.messages = _StubMessages(result=result, error=error)


def _settings() -> Settings:
    return Settings(api_key="sk-ant-test", model="claude-opus-5", max_tokens=1234)


def _message(stop_reason: str = "end_turn", parsed_output=ANALYSIS) -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop_reason, parsed_output=parsed_output)


def _response(status: int) -> httpx2.Response:
    return httpx2.Response(
        status, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    )


def test_returns_parsed_output_and_forwards_settings():
    client = _StubClient(result=_message())
    provider = AnthropicProvider(_settings(), client=client)

    result = provider.structured(system="sys", user="usr", schema=AnalysisPayload)

    assert result is ANALYSIS
    assert client.messages.kwargs["model"] == "claude-opus-5"
    assert client.messages.kwargs["max_tokens"] == 1234
    assert client.messages.kwargs["system"] == "sys"
    assert client.messages.kwargs["output_format"] is AnalysisPayload
    assert client.messages.kwargs["messages"] == [{"role": "user", "content": "usr"}]


def test_missing_key_is_reported_before_any_call():
    """provider= is explicit: the default moved to gemini at Phase 9, and this test is
    about the Anthropic adapter naming its own variable."""
    with pytest.raises(MissingCredentialsError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider(Settings(provider="anthropic", api_key=None))


def test_refusal_is_not_treated_as_output():
    provider = AnthropicProvider(_settings(), client=_StubClient(result=_message("refusal")))

    with pytest.raises(InvalidResponseError, match="declined"):
        provider.structured(system="s", user="u", schema=AnalysisPayload)


def test_truncated_output_is_reported_with_the_remedy():
    provider = AnthropicProvider(_settings(), client=_StubClient(result=_message("max_tokens")))

    with pytest.raises(InvalidResponseError, match="PROMPT_COMPILER_MAX_TOKENS"):
        provider.structured(system="s", user="u", schema=AnalysisPayload)


def test_unparseable_output_is_reported():
    client = _StubClient(result=_message(parsed_output=None))
    provider = AnthropicProvider(_settings(), client=client)

    with pytest.raises(InvalidResponseError, match="structured output"):
        provider.structured(system="s", user="u", schema=AnalysisPayload)


@pytest.mark.parametrize(
    ("error", "expected", "match"),
    [
        (
            anthropic.AuthenticationError("bad key", response=_response(401), body=None),
            MissingCredentialsError,
            "rejected",
        ),
        (
            anthropic.RateLimitError("slow down", response=_response(429), body=None),
            ProviderError,
            "Rate limited",
        ),
        (
            anthropic.APIConnectionError(
                request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
            ),
            ProviderError,
            "Could not reach",
        ),
        (
            anthropic.APIStatusError("boom", response=_response(500), body=None),
            ProviderError,
            "HTTP 500",
        ),
    ],
)
def test_api_errors_map_to_project_errors(error, expected, match):
    provider = AnthropicProvider(_settings(), client=_StubClient(error=error))

    with pytest.raises(expected, match=match):
        provider.structured(system="s", user="u", schema=AnalysisPayload)


def test_a_reply_that_fails_validation_is_reported_not_raised():
    """Regression: min_length reaches the API only as a hint, so the model can return
    JSON the SDK's pydantic validation rejects. That must not escape as a traceback."""
    from pydantic import ValidationError

    try:
        AnalysisPayload(
            task_type="",  # blank scalar - rejected client-side, not by the API
            primary_goal="Explain recursion.",
            complexity="simple",
            confidence=0.5,
        )
    except ValidationError as exc:
        validation_error = exc

    provider = AnthropicProvider(_settings(), client=_StubClient(error=validation_error))

    with pytest.raises(InvalidResponseError, match="did not match the expected shape"):
        provider.structured(system="s", user="u", schema=AnalysisPayload)
