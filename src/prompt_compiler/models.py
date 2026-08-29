"""The pipeline's output."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .analyzer.models import PromptAnalysis
from .optimizer.sections import Section
from .optimizer.token_optimizer import TokenReport
from .scoring import QualityReport


class CompiledPrompt(BaseModel):
    optimized_prompt: str
    analysis: PromptAnalysis
    sections: list[Section] = Field(default_factory=list)
    """The structure the prompt was given. Empty means plain prose was the right shape."""
    safeguards: list[str] = Field(default_factory=list)
    """Factuality rules the analysis earned. Empty means nothing was at risk."""
    tokens: TokenReport
    """Estimated token counts before and after, and what redundancy was removed."""
    quality: QualityReport
    """Heuristic scores for the prompt as written and for the rewrite. Estimates from
    countable evidence, never measured outcomes - see scoring.py."""
    unverified_requirements: list[str] = Field(default_factory=list)
    """Stated requirements and constraints that could not be found in the rewritten
    prompt. Empty is the expected case; a non-empty list is strong evidence the model
    dropped something, and is surfaced rather than silently tolerated."""
    model: str
    models_used: list[str] = Field(default_factory=list)
    """Every model that served a call in this compile. More than one means a provider
    switched mid-compile, which would otherwise be invisible in the result."""

    @property
    def original_prompt(self) -> str:
        """Convenience read-through. Stored once, on the analysis, never duplicated."""
        return self.analysis.original_prompt
