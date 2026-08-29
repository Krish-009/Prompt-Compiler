"""What any provider must do, independent of whose API is behind it.

This is the specification Phase 9's Gemini and Groq adapters have to satisfy. Adding
a provider means adding it to `IMPLEMENTATIONS` below and making these pass - if a new
adapter needs the contract loosened, that is a design discussion, not a test edit.

Everything here runs against stubs. No API key, no network.
"""

from __future__ import annotations

from types import SimpleNamespace

import anthropic
import groq
import httpx
import httpx2
import pytest
from google.genai import errors as genai_errors

from prompt_compiler.analyzer.models import AnalysisPayload
from prompt_compiler.config import (
    KNOWN_PROVIDERS,
    PROVIDER_KEY_VARS,
    Settings,
    model_variable,
)
from prompt_compiler.errors import (
    ConfigurationError,
    InvalidResponseError,
    MissingCredentialsError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from prompt_compiler.providers import (
    BUILDERS,
    AnthropicProvider,
    FallbackProvider,
    GeminiProvider,
    GroqProvider,
    build_one,
    build_provider,
)
from prompt_compiler.providers.base import Provider, detail

PAYLOAD = AnalysisPayload(
    task_type="explanation", primary_goal="Explain recursion.", complexity="simple", confidence=0.9
)


def _response(status: int) -> httpx2.Response:
    return httpx2.Response(
        status, request=httpx2.Request("POST", "https://example.invalid/v1/messages")
    )


def _groq_response(status: int) -> httpx.Response:
    return httpx.Response(
        status, request=httpx.Request("POST", "https://example.invalid/openai/v1")
    )


class _StubCall:
    """One stubbed SDK entry point: records its kwargs, then raises or returns."""

    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result, self.error, self.kwargs = result, error, {}

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.result


def _anthropic(error=None, result=None, **overrides) -> Provider:
    settings = Settings(
        provider="anthropic", api_key="test-key", model="claude-opus-5", **overrides
    )
    if result is None and error is None:
        result = SimpleNamespace(stop_reason="end_turn", parsed_output=PAYLOAD)
    call = _StubCall(result=result, error=error)
    client = SimpleNamespace(messages=SimpleNamespace(parse=call))
    provider = AnthropicProvider(settings, client=client)
    provider.stub = call  # type: ignore[attr-defined]
    return provider


def _gemini(error=None, result=None, **overrides) -> Provider:
    settings = Settings(
        provider="gemini", api_key="test-key", model="gemini-test", **overrides
    )
    if result is None and error is None:
        result = SimpleNamespace(
            parsed=PAYLOAD,
            prompt_feedback=None,
            candidates=[SimpleNamespace(finish_reason="STOP")],
        )
    call = _StubCall(result=result, error=error)
    client = SimpleNamespace(models=SimpleNamespace(generate_content=call))
    provider = GeminiProvider(settings, client=client)
    provider.stub = call  # type: ignore[attr-defined]
    return provider


def _groq(error=None, result=None, **overrides) -> Provider:
    settings = Settings(provider="groq", api_key="test-key", model="groq-test", **overrides)
    if result is None and error is None:
        result = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=PAYLOAD.model_dump_json()),
                )
            ]
        )
    call = _StubCall(result=result, error=error)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=call)))
    provider = GroqProvider(settings, client=client)
    provider.stub = call  # type: ignore[attr-defined]
    return provider


#: name -> how to build a working provider, what each failure looks like for that SDK, and
#: what an unusable reply looks like. The last one is per-implementation because response
#: shapes are provider-specific: the contract is "an unusable reply is an
#: InvalidResponseError", not "every SDK returns an Anthropic-shaped object".
IMPLEMENTATIONS = {
    "anthropic": {
        "ok": _anthropic,
        "unusable": SimpleNamespace(stop_reason="end_turn", parsed_output=None),
        "truncated": SimpleNamespace(stop_reason="max_tokens", parsed_output=None),
        "rate_limit": anthropic.RateLimitError("slow down", response=_response(429), body=None),
        "timeout": anthropic.APITimeoutError(
            request=httpx2.Request("POST", "https://example.invalid")
        ),
        "network": anthropic.APIConnectionError(
            request=httpx2.Request("POST", "https://example.invalid")
        ),
        "bad_key": anthropic.AuthenticationError("bad key", response=_response(401), body=None),
        "server_error": anthropic.APIStatusError("boom", response=_response(500), body=None),
    },
    "gemini": {
        "ok": _gemini,
        "unusable": SimpleNamespace(parsed=None, prompt_feedback=None, candidates=[]),
        "truncated": SimpleNamespace(
            parsed=None,
            prompt_feedback=None,
            candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
        ),
        # Gemini has no granular error classes: everything is a status code.
        "rate_limit": genai_errors.ClientError(429, {"error": {"message": "quota"}}),
        "timeout": httpx.TimeoutException("timed out"),
        "network": httpx.ConnectError("no route"),
        "bad_key": genai_errors.ClientError(401, {"error": {"message": "bad key"}}),
        "server_error": genai_errors.ServerError(500, {"error": {"message": "boom"}}),
    },
    "groq": {
        "ok": _groq,
        "unusable": SimpleNamespace(choices=[]),
        "truncated": SimpleNamespace(
            choices=[
                SimpleNamespace(finish_reason="length", message=SimpleNamespace(content="{"))
            ]
        ),
        "rate_limit": groq.RateLimitError(
            "slow down", response=_groq_response(429), body=None
        ),
        "timeout": groq.APITimeoutError(
            request=httpx.Request("POST", "https://example.invalid")
        ),
        "network": groq.APIConnectionError(
            request=httpx.Request("POST", "https://example.invalid")
        ),
        "bad_key": groq.AuthenticationError(
            "bad key", response=_groq_response(401), body=None
        ),
        "server_error": groq.APIStatusError("boom", response=_groq_response(500), body=None),
    },
}

IMPLEMENTED = sorted(IMPLEMENTATIONS)


@pytest.mark.parametrize("name", IMPLEMENTED)
class TestProviderContract:
    """Every implemented provider must satisfy all of these."""

    def test_reports_its_name_and_model(self, name):
        provider = IMPLEMENTATIONS[name]["ok"]()

        assert provider.name == name
        assert provider.model

    def test_returns_an_instance_of_the_requested_schema(self, name):
        provider = IMPLEMENTATIONS[name]["ok"]()

        result = provider.structured(system="s", user="u", schema=AnalysisPayload)

        assert isinstance(result, AnalysisPayload)

    def test_a_rate_limit_is_its_own_error(self, name):
        provider = IMPLEMENTATIONS[name]["ok"](error=IMPLEMENTATIONS[name]["rate_limit"])

        with pytest.raises(RateLimitError):
            provider.structured(system="s", user="u", schema=AnalysisPayload)

    def test_a_timeout_is_its_own_error(self, name):
        provider = IMPLEMENTATIONS[name]["ok"](error=IMPLEMENTATIONS[name]["timeout"])

        with pytest.raises(ProviderTimeoutError):
            provider.structured(system="s", user="u", schema=AnalysisPayload)

    def test_a_network_failure_is_a_provider_error(self, name):
        provider = IMPLEMENTATIONS[name]["ok"](error=IMPLEMENTATIONS[name]["network"])

        with pytest.raises(ProviderError):
            provider.structured(system="s", user="u", schema=AnalysisPayload)

    def test_a_server_error_is_a_provider_error(self, name):
        provider = IMPLEMENTATIONS[name]["ok"](error=IMPLEMENTATIONS[name]["server_error"])

        with pytest.raises(ProviderError):
            provider.structured(system="s", user="u", schema=AnalysisPayload)

    def test_a_rejected_key_is_a_credentials_error(self, name):
        provider = IMPLEMENTATIONS[name]["ok"](error=IMPLEMENTATIONS[name]["bad_key"])

        with pytest.raises(MissingCredentialsError):
            provider.structured(system="s", user="u", schema=AnalysisPayload)

    def test_an_unusable_reply_is_an_invalid_response(self, name):
        provider = IMPLEMENTATIONS[name]["ok"](result=IMPLEMENTATIONS[name]["unusable"])

        with pytest.raises(InvalidResponseError):
            provider.structured(system="s", user="u", schema=AnalysisPayload)

    def test_a_truncated_reply_says_which_setting_to_raise(self, name):
        """Silently returning half an analysis is the worst outcome available, so the
        cut-off is detected and the message names the knob that fixes it."""
        provider = IMPLEMENTATIONS[name]["ok"](result=IMPLEMENTATIONS[name]["truncated"])

        with pytest.raises(InvalidResponseError, match="PROMPT_COMPILER_MAX_TOKENS"):
            provider.structured(system="s", user="u", schema=AnalysisPayload)

    def test_the_system_prompt_and_user_message_both_reach_the_api(self, name):
        """Both halves must arrive: the system prompt carries every rule the pipeline
        depends on, and a provider that dropped it would still return plausible output."""
        provider = IMPLEMENTATIONS[name]["ok"]()

        provider.structured(system="SYSTEM-MARKER", user="USER-MARKER", schema=AnalysisPayload)

        sent = repr(provider.stub.kwargs)
        assert "SYSTEM-MARKER" in sent
        assert "USER-MARKER" in sent

    def test_the_configured_model_and_token_budget_are_used(self, name):
        provider = IMPLEMENTATIONS[name]["ok"](max_tokens=4321)

        provider.structured(system="s", user="u", schema=AnalysisPayload)

        sent = repr(provider.stub.kwargs)
        assert provider.model in sent
        assert "4321" in sent

    def test_no_error_message_leaks_the_key(self, name):
        for failure in ("rate_limit", "timeout", "network", "bad_key", "server_error"):
            provider = IMPLEMENTATIONS[name]["ok"](error=IMPLEMENTATIONS[name][failure])
            try:
                provider.structured(system="s", user="u", schema=AnalysisPayload)
            except Exception as exc:  # noqa: BLE001 - the point is to inspect any of them
                assert "test-key" not in str(exc)


# ----------------------------------------------------------------- configuration coverage


@pytest.mark.parametrize("name", KNOWN_PROVIDERS)
def test_every_known_provider_has_a_key_variable(name):
    assert PROVIDER_KEY_VARS[name].endswith("_API_KEY")


def test_the_v1_providers_are_the_ones_the_brief_names():
    assert {"gemini", "groq"} <= set(KNOWN_PROVIDERS)
    assert "ollama" not in KNOWN_PROVIDERS


@pytest.mark.parametrize("name", ["gemini", "groq"])
def test_the_v1_providers_are_implemented(name, monkeypatch):
    """Phase 9. Until the adapters existed these reported "not implemented yet (arriving
    in Phase 9)"; the message is gone because the thing it apologised for is done."""
    monkeypatch.setenv(PROVIDER_KEY_VARS[name], "test-key")
    monkeypatch.setenv("PROMPT_COMPILER_MODEL", "some-model")

    provider = build_one(Settings.from_env(name, load_env_file=False))

    assert provider.name == name
    assert provider.model == "some-model"


def test_an_unimplemented_known_provider_would_still_name_itself(monkeypatch):
    """The guard that covered Gemini and Groq before Phase 9 must survive for whoever adds
    the next provider to KNOWN_PROVIDERS before writing its adapter."""
    monkeypatch.setitem(BUILDERS, "gemini", None)
    monkeypatch.delitem(BUILDERS, "gemini")
    monkeypatch.setenv(PROVIDER_KEY_VARS["gemini"], "test-key")
    monkeypatch.setenv("PROMPT_COMPILER_MODEL", "some-model")

    with pytest.raises(ConfigurationError, match="not implemented yet"):
        build_one(Settings.from_env("gemini", load_env_file=False))


def test_an_unknown_provider_is_rejected_by_name():
    with pytest.raises(ConfigurationError, match="Unknown provider"):
        Settings.from_env("llama-on-a-toaster", load_env_file=False)


def test_each_provider_reads_its_own_key_variable(monkeypatch):
    for name in KNOWN_PROVIDERS:
        for variable in PROVIDER_KEY_VARS.values():
            monkeypatch.delenv(variable, raising=False)
        monkeypatch.setenv(PROVIDER_KEY_VARS[name], f"key-for-{name}")

        settings = Settings.from_env(name, load_env_file=False)

        assert settings.api_key == f"key-for-{name}"


def test_a_missing_key_names_the_variable_to_set(monkeypatch):
    for variable in PROVIDER_KEY_VARS.values():
        monkeypatch.delenv(variable, raising=False)

    settings = Settings.from_env("gemini", load_env_file=False)

    with pytest.raises(MissingCredentialsError, match="GEMINI_API_KEY"):
        settings.require_api_key()


def test_a_provider_without_a_default_model_says_so(monkeypatch):
    """No default model for a V1 provider: a guess could silently select a paid model."""
    monkeypatch.delenv("PROMPT_COMPILER_MODEL", raising=False)

    settings = Settings.from_env("groq", load_env_file=False)

    assert settings.model is None
    with pytest.raises(ConfigurationError, match="--model"):
        settings.require_model()


def test_the_api_key_never_appears_in_a_settings_repr():
    settings = Settings(provider="gemini", api_key="redacted-test-value")

    assert "redacted-test-value" not in repr(settings)


def test_only_implemented_providers_are_registered():
    """Guards against a half-migration: a name in BUILDERS must actually build."""
    assert set(BUILDERS) <= set(KNOWN_PROVIDERS)
    assert set(BUILDERS) == set(IMPLEMENTED)


def test_build_provider_without_a_usable_fallback_returns_the_bare_provider(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    settings = Settings.from_env("anthropic", load_env_file=False)
    provider = build_provider(settings, notify=lambda _: None)

    assert isinstance(provider, AnthropicProvider)


# ------------------------------------------------- per-provider models (Phase 9 finding)


def test_a_provider_specific_model_variable_wins(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("PROMPT_COMPILER_MODEL", "generic-model")
    monkeypatch.setenv(model_variable("gemini"), "gemini-specific")

    assert Settings.from_env("gemini", load_env_file=False).model == "gemini-specific"


def test_the_generic_model_still_applies_to_the_provider_you_asked_for(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("PROMPT_COMPILER_MODEL", "generic-model")
    monkeypatch.delenv(model_variable("gemini"), raising=False)

    assert Settings.from_env("gemini", load_env_file=False).model == "generic-model"


def test_the_fallback_does_not_inherit_the_primarys_model(monkeypatch):
    """Phase 9, found on the first live run. Model ids are not portable, so a fallback
    that inherits PROMPT_COMPILER_MODEL is handed a Gemini id and answers 404. It then
    failed to build and was dropped in silence, which is a fallback that only looks
    configured."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("PROMPT_COMPILER_MODEL", "gemini-3.6-flash")
    monkeypatch.delenv(model_variable("groq"), raising=False)

    primary = Settings(provider="gemini", api_key="test-key", model="gemini-3.6-flash")

    assert primary.for_provider("groq").model is None


def test_the_fallback_uses_its_own_model_variable(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("PROMPT_COMPILER_MODEL", "gemini-3.6-flash")
    monkeypatch.setenv(model_variable("groq"), "openai/gpt-oss-20b")

    primary = Settings(provider="gemini", api_key="test-key", model="gemini-3.6-flash")

    assert primary.for_provider("groq").model == "openai/gpt-oss-20b"


def test_an_unusable_fallback_is_announced_rather_than_dropped(monkeypatch):
    """It cannot fail the run - the primary still works - but it must not be silent."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv(model_variable("gemini"), "gemini-test")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    said: list[str] = []

    provider = build_provider(
        Settings.from_env("gemini", load_env_file=False), notify=said.append
    )

    assert isinstance(provider, GeminiProvider)
    assert said and "groq" in said[0] and "fallback" in said[0]


def test_a_fully_configured_pair_attaches_the_fallback(monkeypatch):
    """The V1 configuration: gemini primary, groq fallback, each with its own model."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv(model_variable("gemini"), "gemini-test")
    monkeypatch.setenv(model_variable("groq"), "groq-test")
    said: list[str] = []

    provider = build_provider(
        Settings.from_env("gemini", load_env_file=False), notify=said.append
    )

    assert isinstance(provider, FallbackProvider)
    assert provider.primary.model == "gemini-test"
    assert provider.fallback.model == "groq-test"
    assert said == []


# -------------------------------------------------- provider explanations (Phase 9 finding)


def test_a_mapped_error_keeps_the_providers_own_explanation():
    """Found live: Gemini answers a retired model id with "use models/gemini-3.6-flash"
    and Groq answers a 413 with the exact token limit. Both were reduced to a bare status
    code, discarding the only part that says how to fix it."""
    provider = IMPLEMENTATIONS["gemini"]["ok"](
        error=genai_errors.ClientError(
            404, {"error": {"message": "This model is no longer available to new users."}}
        )
    )

    with pytest.raises(ProviderError, match="no longer available"):
        provider.structured(system="s", user="u", schema=AnalysisPayload)


def test_groqs_server_side_json_rejection_names_the_token_budget():
    """Found by running the CLI by hand. Groq validates JSON server-side, so a reply cut
    off mid-object returns a 400 "Failed to validate JSON" and never reaches the
    finish_reason check - telling the user their prompt is malformed when the real fix is
    a setting they can raise."""
    provider = IMPLEMENTATIONS["groq"]["ok"](
        error=groq.BadRequestError(
            "Error code: 400 - {'error': {'message': 'Failed to validate JSON.', "
            "'code': 'json_validate_failed'}}",
            response=_groq_response(400),
            body=None,
        )
    )

    with pytest.raises(InvalidResponseError, match="PROMPT_COMPILER_MAX_TOKENS"):
        provider.structured(system="s", user="u", schema=AnalysisPayload)


def test_other_bad_requests_stay_provider_errors():
    """Only the JSON-validation code is reinterpreted; a genuine bad request must not be
    reported as a token-budget problem."""
    provider = IMPLEMENTATIONS["groq"]["ok"](
        error=groq.BadRequestError(
            "Error code: 400 - unknown model", response=_groq_response(400), body=None
        )
    )

    with pytest.raises(ProviderError, match="HTTP 400"):
        provider.structured(system="s", user="u", schema=AnalysisPayload)


def test_a_very_long_provider_message_is_truncated():
    assert len(detail(RuntimeError("x" * 5000))) < 400
    assert detail(RuntimeError("x" * 5000)).endswith("...")


def test_provider_detail_is_collapsed_onto_one_line():
    """These messages are pasted into a one-line CLI error; a raw JSON body with newlines
    would break the layout."""
    assert "\n" not in detail(RuntimeError("first line\n  second line\n\tthird"))
    assert detail(RuntimeError("first line\n  second")) == "first line second"
