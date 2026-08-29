"""Deterministic test doubles. Nothing here touches the network."""

from __future__ import annotations

import pytest

from prompt_compiler.analyzer.models import (
    Ambiguity,
    AnalysisPayload,
    Assumption,
    HallucinationRisk,
    PromptAnalysis,
)
from prompt_compiler.config import PROVIDER_KEY_VARS
from prompt_compiler.optimizer.generator import GeneratedPrompt
from prompt_compiler.providers.base import Provider

SAMPLE_PROMPT = "organize my downloads folder"


@pytest.fixture(autouse=True)
def _no_real_credentials(request, monkeypatch):
    """Keep every real API key out of the offline suite.

    `load_dotenv()` writes `.env` into `os.environ` for the rest of the process, so one
    test reading configuration normally leaves real keys visible to every test after it.
    That was not hypothetical: flipping DEFAULT_PROVIDER to gemini at Phase 9 made two
    unrelated config tests fail, and pytest printed the developer's actual GEMINI_API_KEY
    into the failure output - which is how a key reaches a CI log.

    Clearing them here also means a test that accidentally builds a real client cannot
    reach an API on someone's credit. Live tests opt out, since a key is the point of them.
    """
    if request.node.get_closest_marker("live"):
        return
    for variable in PROVIDER_KEY_VARS.values():
        monkeypatch.delenv(variable, raising=False)


class FakeProvider(Provider):
    """Returns canned structured objects and records every call it received."""

    name = "fake"

    def __init__(
        self,
        payload: AnalysisPayload,
        optimized: str = "Optimized prompt.",
        error: Exception | None = None,
        model: str = "fake-model",
    ) -> None:
        self.model = model
        self._payload = payload
        self._optimized = optimized
        self._error = error
        self.calls: list[dict] = []

    def structured(self, *, system: str, user: str, schema):
        self.calls.append({"system": system, "user": user, "schema": schema})
        if self._error is not None:
            raise self._error
        if schema is AnalysisPayload:
            return self._payload
        if schema is GeneratedPrompt:
            return GeneratedPrompt(optimized_prompt=self._optimized)
        raise AssertionError(f"unexpected schema requested: {schema}")


@pytest.fixture
def payload() -> AnalysisPayload:
    return AnalysisPayload(
        task_type="code generation",
        primary_goal="Write a Python script that organizes the Downloads folder.",
        context=["the user's own machine"],
        explicit_requirements=["organize files in the Downloads folder"],
        constraints=["Python"],
        expected_output="a Python script",
        assumptions=[
            Assumption(
                text="the folder is the user's own Downloads directory",
                basis="my downloads folder",
            )
        ],
        missing_information=["which categories the user wants"],
        ambiguities=[
            Ambiguity(
                kind="vague_terminology",
                text="organize could mean sorting into folders or renaming files",
                severity="medium",
            )
        ],
        hallucination_risks=[
            HallucinationRisk(
                kind="unavailable_information",
                text="the folder's actual contents were never shown",
                grounding="unknown",
                severity="medium",
            )
        ],
        unnecessary_content=[],
        complexity="moderate",
        confidence=0.72,
    )


@pytest.fixture
def analysis(payload: AnalysisPayload) -> PromptAnalysis:
    """What `analyze()` produces for SAMPLE_PROMPT given the payload above."""
    return PromptAnalysis(original_prompt=SAMPLE_PROMPT, **payload.model_dump())


@pytest.fixture
def provider(payload: AnalysisPayload) -> FakeProvider:
    return FakeProvider(payload=payload, optimized="Write a Python script that ...")
