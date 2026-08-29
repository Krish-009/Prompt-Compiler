"""Command-line entry point.

argparse rather than a CLI framework: one command, a handful of flags, no dependency.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .analyzer.ambiguity import by_severity, clarification_questions, worst_severity
from .compiler import compile_prompt
from .config import KNOWN_PROVIDERS, Settings
from .errors import PromptCompilerError
from .models import CompiledPrompt
from .optimizer.sections import describe_plan
from .providers.base import Provider
from .providers.registry import build_provider


def _force_utf8_output() -> None:
    """Windows consoles default to a legacy code page (cp1252), and model output is not
    ASCII - an em dash or a curly quote would otherwise raise UnicodeEncodeError."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # detached, or not a text stream
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prompt-compiler",
        description="Rewrite a basic prompt into a more precise one with the same intent.",
    )
    parser.add_argument("prompt", help="the prompt to compile")
    parser.add_argument(
        "--show-analysis", action="store_true", help="print the analysis before the prompt"
    )
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    parser.add_argument(
        "--model",
        help="override the model for the selected provider; a fallback provider keeps "
        "its own model, since model names are not portable between providers",
    )
    parser.add_argument(
        "--provider",
        choices=KNOWN_PROVIDERS,
        help="provider to use (default: from PROMPT_COMPILER_PROVIDER, else the built-in default)",
    )
    return parser


def _format_analysis(result: CompiledPrompt) -> str:
    a = result.analysis
    lines = [
        "ANALYSIS",
        "--------",
        f"Task type:   {a.task_type}",
        f"Goal:        {a.primary_goal}",
        f"Complexity:  {a.complexity}",
        f"Confidence:  {a.confidence:.2f}  (the model's own estimate)",
    ]
    lines.append(f"Structure:   {describe_plan(result.sections)}")
    lines.append(f"Tokens:      {result.tokens.summary()}")
    lines.append(f"Quality:     {result.quality.summary()}")
    worst = worst_severity(a.ambiguities)
    if worst is not None:
        lines.append(f"Ambiguity:   {worst.upper()}")
    if a.expected_output:
        lines.append(f"Output:      {a.expected_output}")

    for label, items in (
        ("Secondary goals", a.secondary_goals),
        ("Context", a.context),
        ("Explicit requirements", a.explicit_requirements),
        ("Constraints", a.constraints),
        ("Missing information", a.missing_information),
        ("Unnecessary content", a.unnecessary_content),
    ):
        if items:
            lines.append(f"{label}:")
            lines.extend(f"  - {item}" for item in items)

    if a.assumptions:
        lines.append("Assumptions (inferred, not stated):")
        lines.extend(f"  - {item.text}  [from: {item.basis}]" for item in a.assumptions)

    if a.ambiguities:
        lines.append("Ambiguities:")
        lines.extend(
            f"  - [{item.severity.upper()}] {item.kind}: {item.text}"
            for item in by_severity(a.ambiguities)
        )

    if a.hallucination_risks:
        lines.append("Hallucination risks:")
        lines.extend(
            f"  - [{risk.severity.upper()}] {risk.kind} ({risk.grounding}): {risk.text}"
            for risk in a.hallucination_risks
        )

    if result.safeguards:
        lines.append("Factuality rules added:")
        lines.extend(f"  - {rule}" for rule in result.safeguards)

    for subject, heading in (
        ("prompt", "Quality of the prompt you wrote (heuristic)"),
        ("rewrite", "Quality of the rewrite (heuristic)"),
    ):
        scored = result.quality.about(subject)
        if scored:
            lines.append(f"{heading}:")
            lines.extend(
                f"  - {item.name:<21} {item.score:>3}/100 {item.band:<6}  {item.basis}"
                for item in scored
            )

    questions = clarification_questions(a.ambiguities)
    if questions:
        lines.append("Worth asking before answering:")
        lines.extend(f"  {n}. {question}" for n, question in enumerate(questions, start=1))

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, provider: Provider | None = None) -> int:
    _force_utf8_output()
    args = build_parser().parse_args(argv)

    try:
        if provider is None:
            settings = Settings.from_env(args.provider)
            if args.model:
                settings = settings.model_copy(update={"model": args.model})
            provider = build_provider(settings)
        result = compile_prompt(args.prompt, provider)
    except PromptCompilerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if result.unverified_requirements:
        print(
            "warning: these stated items could not be found in the rewritten prompt:",
            file=sys.stderr,
        )
        for item in result.unverified_requirements:
            print(f"  - {item}", file=sys.stderr)

    if args.json:
        print(result.model_dump_json(indent=2))
        return 0

    if args.show_analysis:
        print(_format_analysis(result))
        print()
    print(result.optimized_prompt)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
