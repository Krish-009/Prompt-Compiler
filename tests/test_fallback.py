"""Fallback behaviour: what triggers it, what does not, and that it is never silent."""

from __future__ import annotations

import pytest

from prompt_compiler import compile_prompt
from prompt_compiler.analyzer.models import AnalysisPayload
from prompt_compiler.config import Settings
from prompt_compiler.errors import (
    ConfigurationError,
    InvalidInputError,
    InvalidResponseError,
    MissingCredentialsError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from prompt_compiler.providers.fallback import FallbackProvider

from .conftest import FakeProvider
from .corpus import BY_NAME


class Recorder:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def __call__(self, message: str) -> None:
        self.messages.append(message)


def pair(primary_error: Exception | None = None, payload: AnalysisPayload | None = None):
    payload = payload or BY_NAME["very_simple"].payload
    primary = FakeProvider(payload=payload, error=primary_error, model="primary-model")
    primary.name = "gemini"
    fallback = FakeProvider(payload=payload, optimized="from the fallback", model="fallback-model")
    fallback.name = "groq"
    notify = Recorder()
    return FallbackProvider(primary, fallback, notify=notify), primary, fallback, notify


@pytest.mark.parametrize(
    "error",
    [
        RateLimitError("rate limited"),
        ProviderTimeoutError("timed out"),
        ProviderError("could not reach the API"),
        InvalidResponseError("unusable reply"),
    ],
)
def test_a_provider_that_fails_to_deliver_triggers_the_fallback(error):
    provider, primary, fallback, notify = pair(primary_error=error)

    result = provider.structured(system="s", user="u", schema=AnalysisPayload)

    assert isinstance(result, AnalysisPayload)
    assert len(primary.calls) == 1 and len(fallback.calls) == 1
    assert provider.used is fallback


def test_the_switch_is_announced_and_names_both_providers():
    provider, _, _, notify = pair(primary_error=RateLimitError("rate limited"))

    provider.structured(system="s", user="u", schema=AnalysisPayload)

    assert len(notify.messages) == 1
    message = notify.messages[0]
    assert "gemini" in message and "groq" in message
    assert "unavailable" in message and "fallback" in message


def test_nothing_is_announced_when_the_primary_works():
    provider, primary, fallback, notify = pair()

    provider.structured(system="s", user="u", schema=AnalysisPayload)

    assert notify.messages == []
    assert fallback.calls == []
    assert provider.used is primary


@pytest.mark.parametrize(
    "error",
    [
        MissingCredentialsError("no key configured"),
        ConfigurationError("bad model"),
        InvalidInputError("empty prompt"),
    ],
)
def test_a_problem_the_user_must_fix_is_not_papered_over(error):
    """A missing key or bad config would break the fallback too - and hiding it wastes
    the user's time. Only a provider that failed to deliver earns a switch."""
    provider, _, fallback, notify = pair(primary_error=error)

    with pytest.raises(type(error)):
        provider.structured(system="s", user="u", schema=AnalysisPayload)

    assert fallback.calls == []
    assert notify.messages == []


def test_the_fallback_reports_which_model_actually_answered():
    provider, _, _, _ = pair(primary_error=RateLimitError("rate limited"))

    assert provider.model == "primary-model"
    provider.structured(system="s", user="u", schema=AnalysisPayload)

    assert provider.model == "fallback-model"


def test_a_failure_in_the_fallback_itself_surfaces():
    payload = BY_NAME["very_simple"].payload
    primary = FakeProvider(payload=payload, error=RateLimitError("rate limited"))
    primary.name = "gemini"
    fallback = FakeProvider(payload=payload, error=ProviderError("also down"))
    fallback.name = "groq"

    provider = FallbackProvider(primary, fallback, notify=Recorder())

    with pytest.raises(ProviderError, match="also down"):
        provider.structured(system="s", user="u", schema=AnalysisPayload)


def test_fallback_works_through_a_whole_compile():
    """A mid-pipeline switch must not corrupt the result: the second call goes to the
    fallback and the compile still finishes."""
    case = BY_NAME["coding"]
    payload = case.payload
    primary = FakeProvider(payload=payload, error=RateLimitError("rate limited"))
    primary.name = "gemini"
    fallback = FakeProvider(payload=payload, optimized="rewritten by the fallback")
    fallback.name = "groq"
    notify = Recorder()

    result = compile_prompt(case.prompt, FallbackProvider(primary, fallback, notify=notify))

    assert result.optimized_prompt == "rewritten by the fallback"
    assert len(notify.messages) == 2, "each failed call announces its own switch"


# ---------------------------------------------------------------------- configuration


def test_fallback_defaults_to_groq(monkeypatch):
    monkeypatch.delenv("PROMPT_COMPILER_FALLBACK", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    assert Settings.from_env("anthropic", load_env_file=False).fallback_provider == "groq"


@pytest.mark.parametrize("value", ["none", "off", ""])
def test_fallback_can_be_switched_off(monkeypatch, value):
    monkeypatch.setenv("PROMPT_COMPILER_FALLBACK", value)

    assert Settings.from_env("anthropic", load_env_file=False).fallback_provider is None


def test_a_provider_is_never_its_own_fallback(monkeypatch):
    monkeypatch.setenv("PROMPT_COMPILER_FALLBACK", "anthropic")

    assert Settings.from_env("anthropic", load_env_file=False).fallback_provider is None


def test_an_unknown_fallback_is_rejected(monkeypatch):
    monkeypatch.setenv("PROMPT_COMPILER_FALLBACK", "something-else")

    with pytest.raises(ConfigurationError, match="Unknown fallback provider"):
        Settings.from_env("anthropic", load_env_file=False)


def test_the_fallback_inherits_the_run_limits(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    settings = Settings(provider="anthropic", api_key="k", model="m", max_tokens=4321,
                        timeout_seconds=7.5)

    fallback = settings.for_provider("groq")

    assert fallback.provider == "groq"
    assert fallback.max_tokens == 4321
    assert fallback.timeout_seconds == 7.5
    assert fallback.fallback_provider is None, "a fallback must not have its own fallback"


# ------------------------------------- R1-005: a compile can legitimately span two backends


class FlakyPrimary(FakeProvider):
    """Fails the first call only, then recovers - the realistic transient case."""

    def __init__(self, payload, **kwargs):
        super().__init__(payload=payload, **kwargs)
        self.attempts = 0

    def structured(self, *, system, user, schema):
        self.attempts += 1
        if self.attempts == 1:
            self.calls.append({"system": system, "user": user, "schema": schema})
            raise RateLimitError("transient rate limit")
        return super().structured(system=system, user=user, schema=schema)


def test_a_primary_that_recovers_produces_a_two_backend_compile():
    """The previous test used a permanently-down primary, so both calls landed on the
    fallback and this case had no coverage in either direction."""
    payload = BY_NAME["coding"].payload
    primary = FlakyPrimary(payload, model="gemini-model")
    primary.name = "gemini"
    fallback = FakeProvider(payload=payload, optimized="rewritten", model="groq-model")
    fallback.name = "groq"

    result = compile_prompt(BY_NAME["coding"].prompt, FallbackProvider(primary, fallback,
                                                                      notify=Recorder()))

    assert result.models_used == ["groq-model", "gemini-model"]
    assert len(result.models_used) == 2, "a mixed compile must be visible, not flattened"


def test_a_single_backend_compile_reports_one_model():
    payload = BY_NAME["coding"].payload
    primary = FakeProvider(payload=payload, optimized="rewritten", model="gemini-model")
    primary.name = "gemini"
    fallback = FakeProvider(payload=payload, model="groq-model")
    fallback.name = "groq"

    result = compile_prompt(BY_NAME["coding"].prompt, FallbackProvider(primary, fallback,
                                                                      notify=Recorder()))

    assert result.models_used == ["gemini-model"]
    assert fallback.calls == []


def test_a_plain_provider_still_reports_its_model():
    payload = BY_NAME["coding"].payload

    result = compile_prompt(BY_NAME["coding"].prompt,
                            FakeProvider(payload=payload, optimized="x", model="solo-model"))

    assert result.models_used == ["solo-model"]
    assert result.model == "solo-model"
