"""Golden analyses for each prompt category.

Each case pairs a prompt with the analysis a correct analyzer should produce. They are
hand-written, so they document the contract rather than prove the model meets it - the
live tests reuse the same prompts to check that.
"""

from __future__ import annotations

from dataclasses import dataclass

from prompt_compiler.analyzer.models import (
    Ambiguity,
    AnalysisPayload,
    Assumption,
    HallucinationRisk,
)


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    payload: AnalysisPayload
    optimized: str = "A rewritten prompt."


LONG_PROMPT = (
    "I have a CSV export from our billing system with about 40 columns and roughly "
    "200,000 rows. Some rows are duplicated because the export ran twice on the 3rd. "
    "I need to load it, drop the duplicates, normalise the currency column (it has a "
    "mix of comma-grouped USD amounts and dollar-prefixed ones), and produce a monthly "
    "revenue summary per customer segment. The finance team opens everything in Excel, "
    "so the output has to be an xlsx file with one sheet per segment. It needs to run "
    "on Python 3.11 on a locked-down Windows machine where we cannot install anything "
    "outside the standard library plus pandas and openpyxl."
)

CASES: list[Case] = [
    Case(
        name="very_simple",
        prompt="What is photosynthesis?",
        payload=AnalysisPayload(
            task_type="explanation",
            primary_goal="Explain what photosynthesis is.",
            complexity="simple",
            confidence=0.95,
        ),
    ),
    Case(
        name="very_short",
        prompt="recursion",
        payload=AnalysisPayload(
            task_type="explanation",
            primary_goal="Say something about recursion.",
            missing_information=[
                "what the user wants: a definition, an example, or a debugging aid",
                "the audience and depth expected",
            ],
            ambiguities=[
                Ambiguity(
                    kind="unclear_scope",
                    text="a bare noun with no verb: this could want a definition, an "
                    "example, a comparison, or help with broken code",
                    severity="high",
                    clarifying_question=(
                        "What would you like about recursion - a definition, a worked "
                        "example, or help with specific code?"
                    ),
                )
            ],
            complexity="simple",
            confidence=0.25,
        ),
    ),
    Case(
        name="coding",
        prompt="make me a python program that organizes my downloads folder",
        payload=AnalysisPayload(
            task_type="code generation",
            primary_goal="Write a Python program that organizes the Downloads folder.",
            explicit_requirements=["organize the Downloads folder"],
            constraints=["Python"],
            expected_output="a Python program",
            assumptions=[
                Assumption(
                    text="the folder is the user's own Downloads directory",
                    basis="my downloads folder",
                )
            ],
            missing_information=[
                "what organizing means here: sort by type, by date, or something else",
                "what to do when a destination file already exists",
            ],
            ambiguities=[
                # Medium, not high: sorting into folders is the overwhelmingly common
                # reading, so proceeding with it still produces something useful.
                Ambiguity(
                    kind="vague_terminology",
                    text="organizes could mean sorting into subfolders or renaming files",
                    severity="medium",
                ),
                Ambiguity(
                    kind="missing_requirement",
                    text="no rule is given for a file that already exists at the destination",
                    severity="medium",
                ),
            ],
            hallucination_risks=[
                HallucinationRisk(
                    kind="unavailable_information",
                    text="the folder's actual contents were never shown",
                    grounding="unknown",
                    severity="medium",
                )
            ],
            complexity="moderate",
            confidence=0.7,
        ),
    ),
    Case(
        name="no_invented_requirements",
        prompt="Make a Python calculator.",
        payload=AnalysisPayload(
            task_type="code generation",
            primary_goal="Write a calculator in Python.",
            explicit_requirements=["a calculator", "written in Python"],
            constraints=["Python"],
            missing_information=[
                "which operations it must support",
                "whether it runs in the terminal or has an interface",
            ],
            complexity="simple",
            confidence=0.6,
        ),
    ),
    Case(
        name="explicit_requirements",
        prompt=(
            "Write a Python function that reverses a string, includes type hints, "
            "and raises ValueError on non-string input."
        ),
        payload=AnalysisPayload(
            task_type="code generation",
            primary_goal="Write a Python function that reverses a string.",
            explicit_requirements=[
                "reverse a string",
                "include type hints",
                "raise ValueError on non-string input",
            ],
            constraints=["Python", "a single function"],
            expected_output="a Python function",
            complexity="simple",
            confidence=0.92,
        ),
    ),
    Case(
        name="constraints",
        prompt=(
            "Summarize the attached article in exactly three bullet points, "
            "no jargon, aimed at a ten-year-old."
        ),
        payload=AnalysisPayload(
            task_type="summarization",
            primary_goal="Summarize the article the user refers to.",
            constraints=["exactly three bullet points", "no jargon", "audience is a ten-year-old"],
            expected_output="three bullet points",
            missing_information=["the article itself was not provided"],
            hallucination_risks=[
                HallucinationRisk(
                    kind="unavailable_information",
                    text="no article is attached, so any summary would be invented",
                    grounding="unknown",
                    severity="high",
                )
            ],
            complexity="simple",
            confidence=0.8,
        ),
    ),
    Case(
        name="expected_output_format",
        prompt="Give me a JSON object listing five Python web frameworks and their maintainers.",
        payload=AnalysisPayload(
            task_type="research",
            primary_goal="List five Python web frameworks with their maintainers.",
            explicit_requirements=["five Python web frameworks", "include the maintainers"],
            constraints=["JSON output"],
            expected_output="a JSON object",
            hallucination_risks=[
                HallucinationRisk(
                    kind="fabrication_prone",
                    text="maintainer names change and may be recalled incorrectly",
                    grounding="assumed",
                    severity="medium",
                )
            ],
            complexity="simple",
            confidence=0.75,
        ),
    ),
    Case(
        name="missing_information",
        prompt="fix the bug in my code",
        payload=AnalysisPayload(
            task_type="debugging",
            primary_goal="Fix a bug in the code the user refers to.",
            missing_information=[
                "the code was not provided",
                "what the bug is or how it shows itself",
                "the language and runtime",
            ],
            hallucination_risks=[
                HallucinationRisk(
                    kind="unavailable_information",
                    text="no code was shown, so any diagnosis would be invented",
                    grounding="unknown",
                    severity="high",
                )
            ],
            complexity="simple",
            confidence=0.3,
        ),
    ),
    Case(
        name="ambiguous",
        prompt="build me an app",
        payload=AnalysisPayload(
            task_type="code generation",
            primary_goal="Build an application.",
            missing_information=[
                "what the app should do",
                "the platform: web, mobile or desktop",
                "the language or stack",
                "who will use it",
            ],
            ambiguities=[
                Ambiguity(
                    kind="unclear_scope",
                    text="app is unbounded: it could be a weekend script or a product",
                    severity="high",
                    clarifying_question="What should the app do, and for whom?",
                ),
                Ambiguity(
                    kind="missing_technical_constraint",
                    text="app could mean a web, mobile, desktop or command-line program",
                    severity="high",
                    clarifying_question="Which platform should it run on - web, mobile or desktop?",
                ),
                Ambiguity(
                    kind="unclear_output_format",
                    text="unclear whether working code, a design, or a plan is wanted",
                    severity="medium",
                ),
            ],
            complexity="complex",
            confidence=0.15,
        ),
    ),
    Case(
        name="research",
        prompt="Research how remote work has affected software team productivity since 2020.",
        payload=AnalysisPayload(
            task_type="research",
            primary_goal=(
                "Report on how remote work has affected software team productivity since 2020."
            ),
            explicit_requirements=["cover the period since 2020", "focus on software teams"],
            missing_information=[
                "whether published sources may be cited or only reasoning is wanted",
                "the length and format expected",
            ],
            hallucination_risks=[
                HallucinationRisk(
                    kind="fabrication_prone",
                    text="specific studies, figures and citations are easy to fabricate",
                    grounding="assumed",
                    severity="high",
                )
            ],
            complexity="moderate",
            confidence=0.55,
        ),
    ),
    Case(
        name="complex",
        prompt=(
            "Build a full-stack application where users upload CSV files, analyze the data, "
            "generate charts, authenticate, and deploy it."
        ),
        payload=AnalysisPayload(
            task_type="code generation",
            primary_goal="Build a full-stack CSV analysis application.",
            secondary_goals=["deploy the finished application"],
            explicit_requirements=[
                "upload CSV files",
                "analyze the uploaded data",
                "generate charts",
                "authenticate users",
                "deploy the application",
            ],
            missing_information=[
                "the language and framework",
                "where it should be deployed",
                "what analysis the data needs",
                "which authentication method",
            ],
            ambiguities=[
                Ambiguity(
                    kind="vague_terminology",
                    text="analyze the data covers anything from column totals to modelling",
                    severity="high",
                    clarifying_question="What should the analysis produce from an uploaded CSV?",
                ),
                Ambiguity(
                    kind="missing_technical_constraint",
                    text="no language, framework or hosting target is named",
                    severity="medium",
                ),
            ],
            complexity="complex",
            confidence=0.45,
        ),
    ),
    Case(
        name="long",
        prompt=LONG_PROMPT,
        payload=AnalysisPayload(
            task_type="code generation",
            primary_goal=(
                "Write a Python script that cleans a billing CSV and produces a monthly "
                "revenue summary per customer segment."
            ),
            context=[
                "the CSV has about 40 columns and 200,000 rows",
                "the export ran twice on the 3rd, so some rows are duplicated",
                "the finance team opens the output in Excel",
            ],
            explicit_requirements=[
                "drop duplicate rows",
                "normalise the mixed-format currency column",
                "produce a monthly revenue summary per customer segment",
                "write an xlsx file with one sheet per segment",
            ],
            constraints=[
                "Python 3.11",
                "Windows",
                "only the standard library plus pandas and openpyxl",
            ],
            expected_output="an xlsx file with one sheet per segment",
            missing_information=[
                "which column identifies the customer segment",
                "which column holds the date used for monthly grouping",
                "how to treat currencies other than USD",
            ],
            complexity="complex",
            confidence=0.8,
        ),
    ),
]

BY_NAME = {case.name: case for case in CASES}

#: Requirements a model must not invent for the "Make a Python calculator." prompt.
CALCULATOR_INVENTIONS = (
    "gui",
    "pyqt",
    "tkinter",
    "history",
    "theme",
    "keyboard shortcut",
    "unit conversion",
    "scientific",
)
