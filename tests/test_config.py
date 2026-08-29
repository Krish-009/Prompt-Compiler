from __future__ import annotations

import pytest

from prompt_compiler.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODELS,
    DEFAULT_PROVIDER,
    PROVIDER_KEY_VARS,
    Settings,
)
from prompt_compiler.errors import MissingCredentialsError


def test_reads_environment(monkeypatch):
    """Reads the key belonging to the selected provider, whichever that is - the default
    moved from anthropic to gemini at Phase 9, and this must not care."""
    monkeypatch.setenv(PROVIDER_KEY_VARS[DEFAULT_PROVIDER], "key-for-the-default")
    monkeypatch.delenv("PROMPT_COMPILER_PROVIDER", raising=False)
    monkeypatch.setenv("PROMPT_COMPILER_MODEL", "some-model")
    monkeypatch.setenv("PROMPT_COMPILER_MAX_TOKENS", "4096")

    settings = Settings.from_env(load_env_file=False)

    assert settings.provider == DEFAULT_PROVIDER
    assert settings.api_key == "key-for-the-default"
    assert settings.model == "some-model"
    assert settings.max_tokens == 4096


def test_defaults_when_unset(monkeypatch):
    """The autouse fixture in conftest clears every provider key, so this sees none."""
    for var in ("PROMPT_COMPILER_MODEL", "PROMPT_COMPILER_MAX_TOKENS"):
        monkeypatch.delenv(var, raising=False)

    settings = Settings.from_env(load_env_file=False)

    assert settings.api_key is None
    assert settings.model == DEFAULT_MODELS.get(DEFAULT_PROVIDER)
    assert settings.max_tokens == DEFAULT_MAX_TOKENS


def test_the_v1_primary_has_no_default_model(monkeypatch):
    """Phase 9 flipped the default provider to gemini, which deliberately has no default
    model: a guessed id could silently select a paid one."""
    monkeypatch.delenv("PROMPT_COMPILER_MODEL", raising=False)

    assert DEFAULT_PROVIDER == "gemini"
    assert DEFAULT_MODELS.get("gemini") is None
    assert Settings.from_env(load_env_file=False).model is None


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_key_raises_actionable_error(value):
    with pytest.raises(MissingCredentialsError, match="ANTHROPIC_API_KEY"):
        Settings(provider="anthropic", api_key=value).require_api_key()


def test_api_key_never_appears_in_repr():
    """Regression guard: a leaked repr in a traceback or log must not carry the key."""
    settings = Settings(api_key="sk-ant-secret-value")

    assert "sk-ant-secret-value" not in repr(settings)
    assert "sk-ant-secret-value" not in str(settings)


@pytest.mark.parametrize(
    ("value", "variable"),
    [
        ("not-a-number", "PROMPT_COMPILER_MAX_TOKENS"),
        ("12.5.1", "PROMPT_COMPILER_TIMEOUT"),
    ],
)
def test_unparseable_numeric_settings_are_reported_not_raised(monkeypatch, value, variable):
    """Regression: a typo in an env var used to escape as an uncaught ValueError."""
    from prompt_compiler.errors import ConfigurationError

    monkeypatch.setenv(variable, value)

    with pytest.raises(ConfigurationError, match=variable):
        Settings.from_env(load_env_file=False)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_out_of_range_numeric_settings_are_reported(monkeypatch, value):
    from prompt_compiler.errors import ConfigurationError

    monkeypatch.setenv("PROMPT_COMPILER_MAX_TOKENS", value)

    with pytest.raises(ConfigurationError, match="Invalid configuration"):
        Settings.from_env(load_env_file=False)


def test_blank_numeric_settings_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("PROMPT_COMPILER_MAX_TOKENS", "   ")

    assert Settings.from_env(load_env_file=False).max_tokens == DEFAULT_MAX_TOKENS
