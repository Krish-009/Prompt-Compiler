"""Pipeline orchestration: input -> analysis -> section plan -> generation -> output."""

from __future__ import annotations

from .analyzer.analyzer import analyze
from .errors import InvalidInputError
from .models import CompiledPrompt
from .optimizer.generator import generate
from .optimizer.sections import plan_sections
from .optimizer.token_optimizer import measure, tighten
from .providers.base import Provider
from .safety.hallucination import safeguards as earned_safeguards
from .scoring import score
from .validation import unverified


def compile_prompt(prompt: str, provider: Provider) -> CompiledPrompt:
    """Compile `prompt` into a more precise prompt with the same intent."""
    cleaned = prompt.strip()
    if not cleaned:
        raise InvalidInputError("The prompt is empty. Pass the prompt you want compiled.")

    analysis = analyze(cleaned, provider)
    sections = plan_sections(analysis)
    safeguards = earned_safeguards(analysis)
    generated = generate(cleaned, analysis, provider, sections, safeguards)

    # Did the rewrite carry the analysis forward? Checked against generate()'s raw
    # output, before tightening, so a requirement the model never wrote is distinguished
    # from one tightening removed. Reported, not raised - see validation.py.
    stated = [*analysis.explicit_requirements, *analysis.constraints]
    unverified_requirements = unverified(generated, stated)

    # Local, deterministic, no second opinion from the model: remove only what cannot
    # change meaning, and abandon the pass entirely if a stated item would be lost.
    optimized, findings = tighten(
        generated,
        analysis.explicit_requirements,
        analysis.constraints,
        protected=safeguards,
    )
    tokens = measure(cleaned, optimized, before_tightening=generated, findings=findings)

    # Scored last, on the text that will actually be handed over, and on the findings the
    # earlier steps produced. No further model call and no new evidence - scoring only
    # counts what the pipeline already established.
    quality = score(
        analysis,
        optimized,
        safeguards=safeguards,
        unverified_requirements=unverified_requirements,
    )

    return CompiledPrompt(
        optimized_prompt=optimized,
        analysis=analysis,
        sections=sections,
        safeguards=safeguards,
        tokens=tokens,
        quality=quality,
        unverified_requirements=unverified_requirements,
        model=provider.model,
        models_used=getattr(provider, "models_used", None) or [provider.model],
    )
