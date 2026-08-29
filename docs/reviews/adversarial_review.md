# Adversarial Code Review

This file contains persistent adversarial reviews of the Prompt Compiler.

Reviews are performed after every two completed development phases.

The reviewer is strictly read-only with respect to application code.

---

## Phase 1–6 Review — 2026-08-28

### Review Scope

Phases reviewed:

- Phases 0–6, all marked complete in `docs/development_status.md`. This is the first
  adversarial review of this repository (no prior `docs/reviews/adversarial_review.md`
  existed), so it covers everything built to date rather than only the most recent pair.
  Emphasis is on Phase 5 (hallucination-risk reduction) and Phase 6 (token optimization) as
  the most recently landed and least-scrutinized work; Phases 0–4 were re-read for
  cross-phase interaction risk but not re-derived from scratch.

Changes reviewed:

- `src/prompt_compiler/` in full: `compiler.py`, `models.py`, `errors.py`, `config.py`,
  `cli.py`, `analyzer/` (`models.py`, `analyzer.py`, `ambiguity.py`), `optimizer/`
  (`sections.py`, `generator.py`, `token_optimizer.py`), `safety/hallucination.py`,
  `providers/` (`base.py`, `anthropic.py`, `fallback.py`, `registry.py`).
- `tests/` in full (17 files, 389 tests). `pytest -q` run: **389 passed, 21 deselected**,
  matching the count claimed in `docs/development_status.md` exactly — the documented test
  state is accurate.
- `pyproject.toml`, `docs/development_status.md`, `docs/project_memory.json`,
  `docs/codebase_index.json`.
- No git history exists yet (repository has zero commits), so "since the previous review"
  and "regression vs. last commit" are not meaningful framings here; this review instead
  establishes the baseline.

---

### Overall Verdict

**NEEDS CHANGES**

Summary:

The architecture is genuinely sound: the core has no provider coupling (verified by an AST
import test, not just a grep), the deterministic/LLM split is well-reasoned and mostly
well-enforced, the fallback mechanism's non-silence guarantee is properly tested, and the
explicit/inferred/missing analysis split is a good mechanism for the project's stated goal.
Phases 0–4 hold up well under adversarial reading.

However, this review found **two CRITICAL issues** that both trace back to the same root
cause: the project's single non-negotiable rule — "Preserve intent... this outranks every
other optimization; a change that alters intent is a bug, not a tradeoff" (CLAUDE.md) — is
enforced by _prompting the model correctly_ and by _one narrow local guard_, but not by any
deterministic check that the guard's own precondition actually held. Concretely:

1. The Phase 6 "provably safe" deduplication can silently delete a real, distinct
   requirement or constraint when it differs from another only by punctuation (a decimal
   point, a minus sign, a percent sign) — and the preservation check that exists to catch
   exactly this shares the same blind spot, so it reports "nothing lost" while something
   was lost. (R1-001)
2. Nothing in the pipeline checks that `generate()`'s LLM output actually contains every
   `explicit_requirement` and `constraint` the analysis found. The existing `missing_from()`
   utility could do this and is well-tested in isolation, but it is never called at this
   boundary — only inside `tighten()`'s narrower before/after comparison, which cannot catch
   a requirement that was never generated in the first place. (R1-002)

Both are contained, well-scoped fixes, not evidence of a wrong architecture — this is
precisely the kind of gap the two-phase checkpoint process exists to catch before Phase 7
builds heuristic scoring on top of an unverified foundation. Recommend fixing R1-001 and
R1-002 before Phase 7.

---

## 1. Contrarian Review

### Critical Issues

---

```
Issue ID: R1-001
Severity: CRITICAL
Persona: Contrarian
Category: CORRECTNESS, TOKEN_OPTIMIZATION
File: src/prompt_compiler/optimizer/token_optimizer.py
Location: _normalise() (lines 119-122), used by find_redundancies(), _apply(), _contains()
          and therefore missing_from() and tighten()'s abandon-guard (lines 129-225)
Problem: _normalise() strips punctuation by deletion, with nothing put in its place:
  _WORD_CHARS = re.compile(r"[^\w\s]"); _WORD_CHARS.sub("", text). Two lines that are
  textually and semantically distinct can become byte-for-byte identical once every
  punctuation character is deleted - most concretely, a decimal point or a minus sign
  disappearing merges two different numbers into one digit string ("5.5" and "55" both
  become "55"; "-5" and "5" both become "5"). find_redundancies()/_apply() then treat the
  second line as an exact duplicate of the first and delete it. The abandon-guard in
  tighten() is supposed to catch exactly this kind of loss, but it calls the same
  _normalise()-based _contains() to check "is the stated item still present" - so it
  checks whether the *collided* text is present, which it is (that's why they collided),
  and reports nothing lost.
Why it matters: This is the exact failure mode CLAUDE.md and the module's own docstring
  say must never happen silently: "The pass is abandoned whole if it would drop a
  requirement, constraint or safeguard... Failure mode is 'no saving', never 'quietly
  weaker prompt'" (docs/development_status.md, Phase 6 section). Here the pass does not
  abandon - it saves, silently, at the cost of a real constraint. Because the safety net
  (missing_from/_contains) shares the same normalisation as the thing it is supposed to be
  checking, there is no independent verification left standing. This is squarely
  CLAUDE.md's "Preserve intent" rule, which "outranks every other optimization."
Evidence (reproduced against the installed package, no network):
  >>> from prompt_compiler.optimizer.token_optimizer import tighten, missing_from
  >>> text = "CONSTRAINTS\n- Timeout of 5.5 seconds\n- Timeout of 55 seconds"
  >>> tightened, findings = tighten(text, constraints=["Timeout of 5.5 seconds",
  ...                                                   "Timeout of 55 seconds"])
  >>> tightened
  'CONSTRAINTS\n- Timeout of 5.5 seconds'          # "55 seconds" constraint is GONE
  >>> findings
  [Redundancy(kind='repeated_constraint', text='- Timeout of 55 seconds', removable=True)]
  >>> missing_from(tightened, ["Timeout of 5.5 seconds", "Timeout of 55 seconds"])
  []                                                 # safety net reports nothing missing

  Second, independent reproduction with a sign flip instead of a decimal point:
  >>> text2 = "CONSTRAINTS\n- Operate at -5 degrees\n- Operate at 5 degrees"
  >>> tighten(text2)[0]
  'CONSTRAINTS\n- Operate at -5 degrees'            # "5 degrees" (no minus) is GONE
  Both are two-line inputs with no other redundancy-removal step involved; this is not a
  contrived multi-step interaction, it is the module's central duplicate-collapsing path.
Reproduction / scenario: Any generated prompt where two distinct explicit_requirements or
  constraints differ only by punctuation around digits - version numbers, percentages,
  temperatures, prices, timeouts, coordinates, signed deltas - will silently lose one of
  them the moment they happen to appear as two lines in the same generated prompt. This is
  realistic for the "coding"/technical corpus categories this project targets (e.g. two
  different threshold constraints, "under 2.5% overhead" and "at least 25% coverage",
  would collide).
Recommended action: Change _normalise() to not delete punctuation outright - replace it
  with a single space (then collapse whitespace runs) so digit groups on either side of a
  deleted character cannot fuse. Because find_redundancies, _apply, _contains and
  missing_from all funnel through this one function, fixing it once fixes all four call
  sites. Add a regression test pinning the two examples above (and a couple of
  percent/version-number variants) to tests/test_token_optimizer.py's "preservation
  (load-bearing part)" section, since that section is explicitly the honesty mechanism for
  this module's core claim.
```

---

```
Issue ID: R1-002
Severity: CRITICAL
Persona: Contrarian / First Principles
Category: LOGIC, LLM_BEHAVIOR
File: src/prompt_compiler/compiler.py (lines 15-43), src/prompt_compiler/optimizer/generator.py
      (lines 54-89)
Location: compile_prompt() - the call sequence between generate() and tighten()
Problem: No code anywhere checks that generate()'s LLM output actually contains every
  explicit_requirement and constraint from the analysis. compiler.py passes
  analysis.explicit_requirements/constraints into tighten() only as the "must survive
  tightening" list; tighten()'s guard compares generate()'s raw output against its own
  tightened output, so an item that generate() never wrote at all is absent from *both*
  sides of that comparison and the guard is trivially satisfied. missing_from() - the exact
  utility that would answer "which of these does this text not contain" - exists, is
  exported, and is thoroughly tested in tests/test_token_optimizer.py, but is never invoked
  at the generate() boundary itself, only inside tests and inside tighten()'s narrower
  before/after check.
Why it matters: "Preserve intent" is CLAUDE.md's highest-ranked rule, ahead of every
  optimization goal, and the analysis/generation split (Phase 1-4's central design
  decision) exists specifically so requirements captured in the analysis flow deterministically
  into the generated prompt. Right now that flow is enforced only by instructing the model
  ("REQUIREMENTS: only requirements the user stated" / rule 1, "Preserve intent exactly" in
  generator.py's SYSTEM prompt) with no code-level check that it complied. Per
  docs/development_status.md, the live acceptance tests that would exercise real model
  behaviour against this exact risk ("no invented features, gaps recorded, inferences
  grounded" - tests/test_live_smoke.py) have **never been run once** ("No live path has ever
  executed"). So today, an LLM silently dropping a requirement during rewriting is both
  unverified in reality and unguarded in code - the project's core promise rests entirely on
  an unexecuted assumption.
Evidence (reproduced with a FakeProvider whose generate() step omits one of four stated
  requirements, everything else held realistic):
  >>> payload = AnalysisPayload(..., explicit_requirements=[
  ...     "reverse the input string", "include type hints",
  ...     "raise ValueError on non-string input", "support unicode input"])
  >>> # generate() returns REQUIREMENTS with the first three only - "unicode" appears nowhere
  >>> result = compile_prompt("Write a function that reverses a string, with type hints, "
  ...                          "raises ValueError on bad input, and supports unicode.",
  ...                          DropsARequirement(payload))
  >>> "unicode" in result.optimized_prompt.lower()
  False
  >>> result.tokens.findings
  []                                    # nothing flagged
  # compile_prompt() returns a normal CompiledPrompt - no exception, no warning, no field
  # anywhere records that a stated requirement is absent.
  >>> from prompt_compiler.optimizer.token_optimizer import missing_from
  >>> missing_from(result.optimized_prompt, payload.explicit_requirements)
  ['support unicode input']             # the exact tool to have caught this, unused here
Reproduction / scenario: Any real compile where the model drops, merges, or waters down one
  of several stated requirements during rewriting - a well-documented LLM failure mode under
  multi-requirement instructions, and the more likely the more requirements a prompt states
  (exactly the "long"/"complex" corpus cases this project is built to handle well).
Recommended action: After generate() returns and before (or as part of) tighten(), call
  missing_from(generated, analysis.explicit_requirements + analysis.constraints) and treat a
  non-empty result as an InvalidResponseError (or a structured field on CompiledPrompt the
  CLI surfaces) rather than a silent pass-through. This is a small, local, deterministic
  addition consistent with "Core logic runs offline" and needs no new LLM call. Add a
  regression test using the FakeProvider pattern above.
```

### High Issues

---

```
Issue ID: R1-003
Severity: HIGH
Persona: Contrarian
Category: LOGIC
File: src/prompt_compiler/safety/hallucination.py
Location: safeguards(), lines 76-117, specifically the append order (RESOLVE_CONFLICT,
          MISSING_INFORMATION, STATE_ASSUMPTIONS, NO_FABRICATION, NAME_THE_READING,
          NO_UNPERFORMED_WORK) and the truncation `return earned[:MAX_SAFEGUARDS]` (line 117)
Problem: safeguards() collapses each HallucinationRisk's individual severity into an
  unordered `kinds: set[str]` before deciding what to append (line 84), then truncates the
  resulting list to MAX_SAFEGUARDS (4) by simply slicing it in a fixed source-code order.
  Severity is used once, up front, only to decide "actionable vs. not" (high/medium pass,
  low is dropped) - after that point nothing in the function distinguishes a high-severity
  risk from a medium one, and the RESOLVE_CONFLICT / conflicting_instructions check (line
  89-92) does not even require actionable severity, since it inspects analysis.ambiguities
  directly with no severity filter. The function's own docstring claims the result is "in
  order of importance" (line 77); it is in a fixed, severity-blind order.
Why it matters: When a prompt earns more than 4 distinct safeguard categories - realistic
  for a messy, multi-problem prompt, which is exactly when factuality safeguards matter
  most - the cap can drop a safeguard driven by a genuinely high-severity risk while keeping
  one driven by a low-severity ambiguity, simply because of source position. This
  contradicts the stated design intent and silently weakens exactly the prompts that need
  the strongest guidance.
Evidence: tests/test_hallucination.py's test_safeguards_are_capped_and_ordered_by_importance
  (lines 185-199) is the only test naming this behaviour, but its risk() helper defaults
  severity="high" (line 46) and every risk it constructs uses that default - so the test
  cannot distinguish "ordered by importance" from "ordered by source position", because
  every risk in it has the same importance. Direct reproduction with mixed severities:
  >>> a = PromptAnalysis(..., missing_information=["a trivial gap"],
  ...     ambiguities=[Ambiguity(kind="conflicting_instructions", text="trivial",
  ...                            severity="low")],
  ...     hallucination_risks=[
  ...         risk("unsupported_assumption", "high"), risk("fabrication_prone", "high"),
  ...         risk("ambiguous_reference", "high", text="a HIGH severity reference risk")])
  >>> safeguards(a)
  [RESOLVE_CONFLICT, MISSING_INFORMATION, STATE_ASSUMPTIONS, NO_FABRICATION]
  # RESOLVE_CONFLICT survives from a single LOW-severity ambiguity; NAME_THE_READING -
  # earned by a HIGH-severity ambiguous_reference risk - is cut.
Reproduction / scenario: Any analysis with 5+ distinct earned safeguard categories where the
  ones appended later in source order (NAME_THE_READING, NO_UNPERFORMED_WORK) are driven by
  the analysis's highest-severity findings.
Recommended action: Either (a) sort candidates by the worst severity of the risk(s)/
  ambiguities that earned them before truncating, or (b) if positional priority is
  intentional (e.g. "a real contradiction always outranks a naming issue"), correct the
  docstring to say so explicitly and strengthen the test to assert the priority claim with
  genuinely mixed severities rather than uniform ones.
```

### Medium Issues

---

```
Issue ID: R1-004
Severity: MEDIUM
Persona: Contrarian
Category: TOKEN_OPTIMIZATION
File: src/prompt_compiler/optimizer/token_optimizer.py
Location: count_tokens(), lines 59-70, specifically `max(1, ceil(len(piece) / 4)) if
          piece.isalnum() else 1`
Problem: _TOKEN_RE (`\w+|[^\w\s]`) already separates word-runs from single punctuation
  characters, but count_tokens() re-tests each matched piece with piece.isalnum() to decide
  which branch it "was". Python's str.isalnum() returns False for any string containing an
  underscore, so a multi-character word-run that includes an underscore (any snake_case
  identifier) is misclassified as punctuation and charged a flat 1 token, however long it
  is, instead of ceil(len/4).
Why it matters: Snake_case identifiers are the default naming convention in Python, the
  project's own implementation language and the "coding" corpus category's dominant content
  shape. This makes the systematic undercount hit code-generation prompts - one of this
  project's flagship use cases - hardest, which directly undermines the "Token counts are
  local, deterministic and labelled" engineering rule's usefulness: a labelled-but-
  systematically-biased estimate is worse than an unbiased noisier one for the before/after
  comparisons TokenReport exists to support.
Evidence:
  >>> count_tokens("max_retry_count_value")   # 21 chars, one identifier
  1                                            # ceil(21/4) = 6 if word-classified
  >>> count_tokens("maxretrycountvalue")      # same identifier, no underscores
  5
  >>> count_tokens("def __init__(self, max_retry_count=5, api_base_url=None):")
  14                                           # ~20 by the module's own stated method if
                                                # underscored identifiers were classified as
                                                # words - roughly a 30% undercount on a
                                                # representative function signature
  test_counting_is_in_the_right_ballpark (tests/test_token_optimizer.py:50-54) only
  exercises `" ".join(["word"] * 100)`, which contains no underscores, so this bias is
  invisible to the existing self-check.
Reproduction / scenario: Any compile of a coding-related prompt whose generated text
  contains identifiers, function names, or env-var-style constants - i.e., most of the
  "coding"/"long" corpus categories.
Recommended action: Classify by which regex alternative matched rather than re-testing with
  isalnum() - e.g. `piece.isalnum() or "_" in piece`, or simpler, treat any piece of length
  > 1 as a word (only single-character punctuation can come from the `[^\w\s]` alternative).
  Add an underscore-bearing case to the ballpark test.
```

---

```
Issue ID: R1-005
Severity: MEDIUM
Persona: Contrarian / First Principles
Category: RELIABILITY, ARCHITECTURE, LLM_BEHAVIOR
File: src/prompt_compiler/providers/fallback.py (lines 45-58), src/prompt_compiler/compiler.py
Location: FallbackProvider.structured() - decides primary-vs-fallback independently on
          every call
Problem: compile_prompt() makes two structured() calls (analyze, then generate) against the
  same Provider instance. When that instance is a FallbackProvider, each call independently
  tries primary first and falls back only on that call's own failure - there is no
  compile-scoped memory of "we already switched once". A primary that fails transiently on
  just the first call and recovers by the second produces one compiled prompt built from two
  different backends (e.g. analysis by Groq, generation by Gemini). CompiledPrompt.model is
  set once, from provider.model at the end of compile_prompt() - i.e. it reflects only
  whichever provider handled the *last* call, with nothing recording that the first call
  used a different one.
Why it matters: The project's reliability story ("never silently degrade") covers whether a
  switch is announced, which it is - but not whether the *result* honestly reports its
  provenance when two different models jointly produced it. If the two providers differ in
  analysis quality or rewriting style, a mixed compile is silently indistinguishable from a
  single-provider one in both the CLI output and the JSON output, which matters for a
  project whose engineering rule is "measure, don't assert" - a benchmark run (Phase 8) that
  doesn't know a given result was a hybrid would misattribute its quality to one provider.
Evidence: tests/test_fallback.py's test_fallback_works_through_a_whole_compile (lines
  125-139) is the only test of a mid-pipeline switch, but its primary FakeProvider is
  configured to fail unconditionally on every call (the error is fixed at construction and
  raised on every structured() invocation), so both calls land on the fallback - it cannot
  exercise or detect the recover-between-calls case. Direct reproduction with a primary that
  fails once then recovers:
  >>> fb = FallbackProvider(FlakyPrimary(payload), StableFallback(payload), notify=...)
  >>> result = compile_prompt("...", fb)
  # analysis served by StableFallback ("groq"), generation served by FlakyPrimary
  # ("gemini", recovered) - one compile, two backends.
  >>> result.model
  'gemini-primary-model'          # reports only the generation call's provider
Reproduction / scenario: A transient rate limit or timeout affecting only one of the two
  calls in a compile - realistic for cloud APIs under load, and more likely than a sustained
  outage.
Recommended action: Either make FallbackProvider sticky for the lifetime of one compile
  (once it falls back, stay on the fallback for subsequent calls in the same logical
  operation) for consistency, or - if independent-per-call retry is intentional, since it
  maximises primary usage - surface both providers actually used on CompiledPrompt (e.g. a
  list, not a single `model` string) so a mixed compile is observable rather than silently
  flattened.
```

### Low Issues

---

```
Issue ID: R1-006
Severity: LOW
Persona: Contrarian
Category: TESTING, MAINTAINABILITY
File: tests/test_provider_contract.py (line 15), tests/test_provider_anthropic.py,
      pyproject.toml (lines 10-14)
Location: `import httpx2` at module scope in both test files
Problem: httpx2 is not declared anywhere in pyproject.toml's dependencies or the dev extra -
  it is importable today only because it happens to be anthropic 1.2.0's own HTTP transport
  dependency (confirmed via `pip show httpx2`: Required-by: anthropic). Not a bug in the
  sense of currently-wrong behaviour - the imports work and the tests pass - but the test
  suite is relying on an undeclared transitive dependency's public API (httpx2.Response,
  httpx2.Request) to construct fake SDK errors.
Why it matters: pyproject.toml pins `anthropic>=1.2` with no upper bound. If a future
  anthropic release changes its HTTP backend (a real possibility - the library has already
  moved from a presumably httpx-based transport to httpx2), `import httpx2` breaks with a
  bare ModuleNotFoundError at test-collection time across two files, which is a confusing
  failure mode for something that is really "your pinned dependency changed its transport."
Evidence: `pip show httpx2` -> Required-by: anthropic; `pip show httpx` -> not installed;
  neither appears in pyproject.toml's dependency list.
Reproduction / scenario: Bumping anthropic to a version that drops or replaces httpx2.
Recommended action: Either add httpx2 explicitly to the dev dependency group (documenting
  that it is used directly, not just transitively), or build the fake error objects through
  a thinner project-owned helper that does not require naming anthropic's transport library
  directly.
```

---

## 2. First Principles Review

### Fundamental Findings

The core objective - transform a basic prompt into one that improves accuracy and
instruction-following while reducing token waste and hallucination risk, without changing
what the user asked for - is well understood by the implementation. The explicit/inferred/
missing split (Phase 2) and the earned-safeguards/earned-sections policies (Phases 3-5) are
a coherent, defensible mechanism for "don't invent, don't over-scaffold."

The one fundamental flaw found: **the architecture assumes generate() faithfully carries
every analysis finding forward, but nothing checks that assumption** - see R1-002. This is a
"technically works, logically incomplete" case: the two-call split (analyze, then
deterministically plan, then generate) is sound in principle specifically _because_ it
removes structural decisions from the model's discretion, but it currently leaves content
fidelity entirely in the model's discretion, unchecked. The mechanism that would close this
(missing_from) already exists in the codebase - this is a wiring gap, not a missing concept.

### Architectural Findings

The suggested canonical pipeline (Input -> Analysis -> Optimization -> Validation -> LLM
Provider -> Output) names Validation as a distinct stage. The actual pipeline has no such
stage: validation logic is scattered piecemeal across generator.py (empty-output check),
token_optimizer.py (tighten()'s abandon-guard, which validates only its own edit), and
nowhere else. There is no single place responsible for checking the _final_ output against
the _full_ set of invariants (all stated requirements present, all stated constraints
present, no forbidden boilerplate phrases, section headings match the plan exactly). This
diffusion is very likely why R1-002 was possible to build around without any test noticing -
every existing test that checks "requirements survive" hands tighten() text that already
contains them (by construction, via FakeProvider(optimized=...)), so the suite documents
that _tightening_ preserves content well, but nothing documents that _generation_ does.

Elsewhere the separation of concerns is genuinely good: `test_optimization_imports_no_provider`
(tests/test_token_optimizer.py:308-323) checks the actual AST rather than grepping text, which
is a meaningfully stronger guarantee than most projects bother with, and the Phase 9
provider seam (registry.py's BUILDERS dict) is exactly as thin as project_memory.json claims.

### Requirement Preservation Findings

Covered in depth under R1-001 and R1-002. One further observation: `tighten()` is given
`protected=safeguards` (compiler.py:32) specifically so Phase 5's earned rules survive Phase
6's tightening - a good instinct - but because `protected` items are checked with the same
`_contains()` as requirements and constraints, a safeguard phrased with any punctuation-
adjacent digit (none currently are, by inspection of hallucination.py's rule text) would
inherit the same R1-001 blind spot. Worth keeping in mind if a future safeguard's wording
changes.

### Deterministic vs LLM Findings

This split is one of the strongest parts of the codebase. Section planning, safeguard
selection, ambiguity-to-question policy, and token measurement are all local, tested without
a network connection, and importantly _documented as policy decisions_ in
project_memory.json rather than left as implicit model behaviour. The one place this review
would push further: R1-002's fix (a post-generation invariant check) belongs in this same
deterministic category and costs no additional LLM call, consistent with "One LLM call per
prompt wherever practical."

One open question worth surfacing rather than asserting as a bug: `plan_sections()`
(optimizer/sections.py:41-75) requires `MIN_CONTENT_SECTIONS` (3) distinct _categories_ to be
earned before using any structure at all, counting CONTEXT/REQUIREMENTS/CONSTRAINTS/PROCESS/
OUTPUT FORMAT/FACTUALITY RULES as categories regardless of how many items sit inside each
one. A hypothetical analysis with, say, six explicit_requirements and three constraints but
nothing in the other four categories earns only 2 categories and is written as prose - the
same "two facts are a sentence, three are a list" reasoning the module's own docstring gives
for the floor would seem to argue for a dense two-category prompt getting a list too. This
review did not find a corpus case that actually hits this today (the existing corpus's
multi-requirement cases, e.g. "complex" and "long", also carry missing_information and
therefore earn FACTUALITY RULES, clearing the floor incidentally) - so this is flagged as a
design question worth a deliberate answer, not a confirmed defect. See Expansionist,
Future Opportunities.

---

## 3. Expansionist Review

### Necessary Improvements

- A named, tested, deterministic post-generation validation step (requirement/constraint
  presence at minimum; forbidden-boilerplate absence and section-heading compliance would be
  natural to fold in) rather than validation logic implicit and split across three modules.
  This is the constructive counterpart to R1-002 and R1-001's architectural findings above.
- Fix `_normalise()` to replace stripped punctuation with a space rather than deleting it
  (R1-001's recommended action) - one change, four call sites fixed.

### High-Value Opportunities

- Once Phase 9 lands a real provider, benchmark `count_tokens()` against that provider's own
  token counting on a sample of compiled prompts, per Phase 8's own mandate to decide
  whether the deterministic-plan / two-call design earns its cost. This would have caught
  R1-004 directly and would turn "estimated" from a hedge into a measured error bound.
- Track which provider/model actually served _each_ call within a compile (not just the
  last one) and surface it on `CompiledPrompt`. Turns R1-005 from an invisible edge case
  into an observable, benchmarkable signal, and costs nothing extra since `FallbackProvider`
  already tracks `.used` per call internally.
- Extend the existing preservation-test style (tests/test_token_optimizer.py's "load-bearing"
  section) to numeric/technical-token requirements specifically - signed numbers, decimals,
  percentages, version strings - since that is precisely the gap that let R1-001 and R1-004
  both through today's otherwise-thorough test suite.

### Future Opportunities

- Before the project leans on `test_live_smoke.py`'s claims operationally, actually run it
  at least once against each V1 provider (even a single manual `pytest -m live` pass) so "no
  live path has ever executed" is no longer true for the system's central promise. Natural
  fit for early Phase 9.
- Make a deliberate, documented decision on the `MIN_CONTENT_SECTIONS` category-vs-volume
  question above (First Principles) - e.g. by printing the section distribution against a
  couple of item-dense-but-narrow synthetic analyses, the same technique the Phase 3-4
  validation pass already used successfully to catch a real bug.

---

## Testing Assessment

Test suite status: **389 passed, 21 deselected (`pytest -q`), 1.05s.** Confirmed by direct
run against `.venv`; matches `docs/development_status.md`'s claimed count exactly. All
deterministic, no network or key required for the default run, confirmed by inspection (no
`live`-marked test runs without `-m live`, and the marker mechanism in `pyproject.toml` is
correctly scoped via `addopts = "-m 'not live'"`).

Coverage concerns:

- The suite is happy-path-heavy specifically at LLM-output boundaries: nearly every test
  that checks "does X survive" constructs a FakeProvider whose canned output already
  contains X (by hand), which validates the _local_ logic thoroughly but never asks "what if
  the model's output itself was already missing X" - that question is answered nowhere
  (R1-002).
- `test_safeguards_are_capped_and_ordered_by_importance` and the ballpark token-count test
  both use inputs too uniform to exercise the behaviour their names claim to check
  (R1-003, R1-004) - a naming/intent mismatch worth the primary agent's attention beyond
  just the two specific bugs, since it suggests looking for other "test name promises more
  than the test body checks" cases during the fix pass.
- `test_fallback_works_through_a_whole_compile` tests a permanently-down primary, not a
  recovers-between-calls primary, so the mixed-provider case (R1-005) has no coverage in
  either direction.

Missing tests (concrete, addable without any application change beyond the fixes above):

- A generate()-drops-a-requirement case at the `compile_prompt()` level (would currently
  fail without R1-002's fix, which is the point).
- A punctuation-collision preservation case in the token_optimizer suite (decimal, sign,
  percent) - would currently fail without R1-001's fix.
- A mixed-severity safeguards case, replacing or supplementing the current uniform-severity
  one.
- An underscore-bearing case in the token-counting ballpark test.

---

## Previous Finding Status

| Issue                 | Previous Status | Current Status |
| --------------------- | --------------- | -------------- |
| (none - first review) | -               | -              |

---

## Final Priority List

### CRITICAL

- R1-001 - `tighten()`/`_normalise()` can silently delete a distinct requirement or
  constraint when it collides with another after punctuation is stripped, and
  `missing_from()` cannot detect the loss because it shares the same normalisation.
- R1-002 - No deterministic check that `generate()`'s output actually contains every
  `explicit_requirement` and `constraint` from the analysis; `missing_from()` exists but is
  never called at this boundary.

### HIGH

- R1-003 - `safeguards()`'s 4-item cap truncates by fixed source order, not by the
  analysis's actual risk severities, contradicting its own "in order of importance"
  contract.

### MEDIUM

- R1-004 - `count_tokens()` undercounts underscore-containing identifiers (common in code
  prompts, this project's own flagship use case) by misclassifying them via `isalnum()`.
- R1-005 - `FallbackProvider` selects primary-vs-fallback independently per call, so a
  transient failure can produce a single compiled prompt silently built from two different
  providers, unreported on `CompiledPrompt.model`.

### LOW

- R1-006 - `httpx2` used directly in two test files without being declared as a dependency;
  only present transitively via `anthropic`.
- (Design question, not a defect) `MIN_CONTENT_SECTIONS` counts earned categories rather
  than item volume - see First Principles / Expansionist for the concrete case to check.

---

## Recommendation

**FIX BEFORE CONTINUING**

Reason: R1-001 and R1-002 are both silent violations of CLAUDE.md's highest-ranked rule
("Preserve intent... outranks every other optimization; a change that alters intent is a bug,
not a tradeoff"), in the two subsystems (Phase 5-6's token optimizer, and the generate()
boundary) that most recently landed and are least battle-tested. Neither requires a
redesign - R1-001 is a one-function fix with four beneficiaries, R1-002 is wiring an
existing, already-tested utility into one new call site. Phase 7 (heuristic quality scoring)
will compute derived signals from `PromptAnalysis` and `TokenReport`; fixing R1-004 first
keeps the token half of that input trustworthy, and fixing R1-001/R1-002 first means Phase 7
and Phase 8 measure a pipeline that actually keeps its central promise rather than one that
merely claims to.

---

## Reviewer Notes

Additional observations:

- The project's documentation hygiene is unusually good for this stage: `development_status.md`'s
  test count, the "no live path has ever executed" admission, and the call-count
  open-question note were all independently verified accurate rather than optimistic. This
  made the review faster and is worth preserving as a habit.
- Several design decisions that might look like bugs on first read are deliberate and
  correctly tested: the fallback being silently _not attached_ when unconfigured (vs. a
  _switch_ always being announced) is intentional and documented in fallback.py's docstring;
  a provider-provided stub client bypassing `require_api_key()` in `AnthropicProvider.__init__`
  is intentional test seam, not a credential-check bypass in production use. Neither is
  listed above because the implementation matches its own stated intent and the intent is
  reasonable.
- `--model`'s CLI help text does not clarify that it overrides only the primary provider
  (a fallback provider's model comes from its own env var / default, which is almost
  certainly correct given model identifiers are not portable across providers) - a one-line
  documentation clarification, not filed as a numbered issue since it is not a behaviour
  problem.

---

# Resolution Pass — 2026-08-28

Performed by the Lead Code Fixer agent against the Phase 1–6 review above. Every finding was
independently reproduced against the installed package before any code changed, and the full
suite was run before the first change (389 passed), after each fix, and twice at the end
(426 passed, 21 deselected, both runs).

**Suite: 389 → 426 tests. No regressions. The original review above is unmodified.**

---

### Issue R1-001

**Original Severity:** CRITICAL
**Persona:** Contrarian
**Classification:** CONFIRMED BUG
**Status:** FIXED

**What was found:**
Reproduced exactly as reported. `_normalise()` deleted punctuation outright, so
"Timeout of 5.5 seconds" and "Timeout of 55 seconds" produced the same key; the duplicate
collapse deleted the second constraint, and `missing_from()` — sharing that normalisation —
returned `[]`, reporting nothing lost. The sign case reproduced too.

One detail in the report is inaccurate: the suggested percentage example
("under 2.5% overhead" / "at least 25% coverage") does **not** collide, because the
surrounding words differ. The bug is real; that particular pair was not an instance of it.

**What was changed — and why not as recommended:**
The recommended fix (replace punctuation with a space, then collapse whitespace) is
**insufficient**, and I verified that before adopting anything: it fixes fusion (5.5 vs 55)
but not deletion of a semantically significant character. "Operate at -5 degrees" becomes
"operate at 5 degrees" under that rule and still collides with "Operate at 5 degrees".
Confirmed by running it.

Instead, `_normalise()` now strips only what is genuinely cosmetic: bullet markers, **edge**
punctuation, case and whitespace. Internal punctuation is preserved. This fixes all six
collision shapes tested (decimal, sign, version, percentage, price, trailing precision)
while still collapsing true duplicates across bullet styles, casing and trailing
punctuation. As the reviewer noted, the single change fixes all four call sites.

**Files Modified:**

- `src/prompt_compiler/optimizer/token_optimizer.py` (`_normalise`, `_EDGE_PUNCT` replacing
  `_WORD_CHARS`)

**Tests:**

- Added `test_two_items_differing_only_by_punctuation_both_survive` (6 parametrised cases)
  and `test_the_preservation_check_is_not_blinded_by_punctuation` to the "preservation
  (load-bearing part)" section, as recommended.
- Added `test_cosmetic_differences_still_collapse` to pin that the fix did not go too far
  the other way.
- Full suite passed.

---

### Issue R1-002

**Original Severity:** CRITICAL
**Persona:** Contrarian / First Principles
**Classification:** CONFIRMED BUG
**Status:** FIXED

**What was found:**
Reproduced exactly. A fake provider whose generation step omitted one of four stated
requirements produced a normal `CompiledPrompt` with no exception, no warning and no field
recording the omission; `missing_from()` on the same output correctly returned
`['support unicode input']`.

**What was changed — and why not exactly as recommended:**
The recommendation offered two options: raise `InvalidResponseError`, or record a structured
field. **Raising was rejected**, deliberately. Requirement text is produced by the _analysis_
call; the prompt text is produced by the _generation_ call, which is instructed to rewrite
rather than transcribe. "reverse the input string" legitimately becomes "reverses the given
string". A literal check — which is what `missing_from()` is — would fire on ordinary good
output, and hard-failing a correct compile is a worse failure than the rare silent omission
it would prevent.

Added `src/prompt_compiler/validation.py` with a paraphrase-tolerant presence check: literal
match first, then content-word overlap at a 0.66 threshold with 5-character prefix matching
to absorb inflection, stopwords excluded. `compile_prompt()` runs it against `generate()`'s
raw output — **before** tightening, so a requirement the model never wrote is distinguishable
from one tightening removed — and records the result on
`CompiledPrompt.unverified_requirements`. The CLI prints a warning to stderr, so it is seen
even when stdout is piped.

`missing_from()` is retained unchanged: inside `tighten()` it compares a text against
_itself_ before and after an edit, where literal matching is exactly correct. The two checks
answer different questions and both are needed.

**Files Modified:**

- `src/prompt_compiler/validation.py` (new)
- `src/prompt_compiler/compiler.py`, `src/prompt_compiler/models.py`,
  `src/prompt_compiler/cli.py`

**Tests:**

- Added `tests/test_validation.py` (18 tests): the dropped-requirement case at
  `compile_prompt()` level as recommended, plus the inverse property — a heavily reworded but
  complete rewrite must **not** be flagged — the stopword and threshold edge cases,
  constraints as well as requirements, JSON round-trip, and every relevant corpus case.
- Full suite passed.

---

### Issue R1-003

**Original Severity:** HIGH
**Persona:** Contrarian
**Classification:** CONFIRMED BUG
**Status:** FIXED

**What was found:**
Reproduced exactly: with a low-severity `conflicting_instructions` ambiguity and three
high-severity risks, `RESOLVE_CONFLICT` survived the cap and `NAME_THE_READING` — earned by
a high-severity `ambiguous_reference` risk — was cut, purely by source position. The
docstring's "in order of importance" claim was false.

**What was changed:**
`safeguards()` now builds each candidate rule with the severities of the evidence that earned
it, sorts worst-first, and truncates. Ties keep the declared order, so a genuine contradiction
still leads a field of equals. A gap in `missing_information` carries no severity of its own
and is rated medium — worth a rule, but yielding to a risk the analysis actually rated high.
The docstring now states the real rule.

Deliberately **not** changed: what is _earnable_ is identical. A low-severity conflict still
earns `RESOLVE_CONFLICT`; it now simply sorts last and is cut first. Changing earnability
would have been a behaviour change the finding did not call for.

**Files Modified:**

- `src/prompt_compiler/safety/hallucination.py`

**Tests:**

- Added `test_a_high_severity_rule_outranks_a_low_severity_one_at_the_cap` (the reviewer's
  exact scenario), `test_ties_keep_the_declared_priority`,
  `test_a_medium_rule_yields_to_a_high_one`, and
  `test_severity_ordering_does_not_change_what_is_earned`.
- Renamed `test_safeguards_are_capped_and_ordered_by_importance` to
  `test_safeguards_are_capped`: the reviewer was right that its uniform-severity body could
  not check the ordering half of its name. The new tests cover that half properly.
- Full suite passed.

---

### Issue R1-004

**Original Severity:** MEDIUM
**Persona:** Contrarian
**Classification:** CONFIRMED BUG
**Status:** FIXED

**What was found:**
Reproduced exactly: `count_tokens("max_retry_count_value")` returned 1 against 5 for the same
identifier without underscores, because `str.isalnum()` is False for any string containing an
underscore. The representative signature returned 14.

**What was changed:**
Adopted the reviewer's simpler suggestion — the punctuation alternative can only ever match a
single character, so length alone classifies correctly and no branch is needed. The identifier
now costs 6, the signature 20.

**Files Modified:**

- `src/prompt_compiler/optimizer/token_optimizer.py` (`count_tokens`)

**Tests:**

- Added four tests including an underscore-bearing ballpark case, as recommended.
- Full suite passed.

---

### Issue R1-005

**Original Severity:** MEDIUM
**Persona:** Contrarian / First Principles
**Classification:** VALID IMPROVEMENT
**Status:** FIXED

**What was found:**
Reproduced: a primary that fails only the first call yields a compile whose analysis came from
the fallback and whose generation came from the recovered primary, with `CompiledPrompt.model`
reporting only the latter.

**What was changed — the second option, not the first:**
The reviewer offered stickiness or provenance. **Stickiness was rejected**: `FallbackProvider`
has no notion of a compile boundary, so "sticky" would mean permanently abandoning the primary
after one transient blip — worse than the problem, since it would keep a whole session on the
weaker provider.

Provenance was implemented instead. `FallbackProvider` accumulates `models_used` (in order,
deduplicated), and `compile_prompt()` surfaces it as `CompiledPrompt.models_used`, falling
back to a single-element list for a plain provider. A mixed compile is now observable, which
is what Phase 8's benchmarking needs in order to avoid misattributing quality.

**Files Modified:**

- `src/prompt_compiler/providers/fallback.py`, `src/prompt_compiler/models.py`,
  `src/prompt_compiler/compiler.py`

**Tests:**

- Added `test_a_primary_that_recovers_produces_a_two_backend_compile` with a flaky primary
  that fails once and recovers — the case the reviewer correctly identified as having no
  coverage — plus single-backend and plain-provider cases.
- Full suite passed.

---

### Issue R1-006

**Original Severity:** LOW
**Persona:** Contrarian
**Classification:** VALID IMPROVEMENT
**Status:** FIXED

**What was found:**
Confirmed: `httpx2` is imported directly at module scope in two test files and appears nowhere
in `pyproject.toml`; it is present only as the Anthropic SDK's transport dependency.

**What was changed:**
Declared `httpx2>=2` in the dev extra with a comment explaining that it is used directly, not
merely transitively. Kept the direct usage rather than building a wrapper: the tests construct
real SDK exception objects, and a project-owned shim would add indirection for a dependency
already pinned alongside the SDK it belongs to.

**Files Modified:**

- `pyproject.toml`

**Tests:**

- No new tests; this is a declaration fix. Full suite passed.

---

### Design question — `MIN_CONTENT_SECTIONS` counts categories, not volume

**Persona:** First Principles / Expansionist
**Classification:** POTENTIAL RISK
**Status:** DEFERRED

**What was found:**
The reviewer flagged this as a design question rather than a defect, and could not find a
corpus case that hits it. I constructed the case they described and **it reproduces**: an
analysis with six explicit requirements and three constraints, and nothing in the other four
categories, earns only 2 categories and is written as plain prose. Nine stated items in a
paragraph is very likely the wrong shape.

**Reason for deferral:**
The floor is one of several thresholds (`MIN_CONTENT_SECTIONS`, `MAX_SAFEGUARDS`,
`PRESENCE_THRESHOLD`, the filler-phrase list) that are currently reasoned rather than measured,
and `docs/development_status.md` already records Phase 8 as the point where they stop being
judgement calls. Changing this one by intuition now would replace one unmeasured threshold with
another and invalidate the section-distribution baseline the Phase 3–4 validation established.
The concrete reproduction above is recorded so Phase 8 can test it directly rather than
rediscovering it.

---

### Reviewer note — `--model` help text

**Classification:** VALID IMPROVEMENT
**Status:** FIXED

Not filed as a numbered issue by the reviewer. The help text now states that `--model` applies
to the selected provider and that a fallback keeps its own model, since model identifiers are
not portable between providers.

**Files Modified:** `src/prompt_compiler/cli.py`

---

## Resolution Summary

| Issue                | Severity | Classification    | Status                                     |
| -------------------- | -------- | ----------------- | ------------------------------------------ |
| R1-001               | CRITICAL | CONFIRMED BUG     | FIXED (fix differs from recommendation)    |
| R1-002               | CRITICAL | CONFIRMED BUG     | FIXED (reports rather than raises)         |
| R1-003               | HIGH     | CONFIRMED BUG     | FIXED                                      |
| R1-004               | MEDIUM   | CONFIRMED BUG     | FIXED                                      |
| R1-005               | MEDIUM   | VALID IMPROVEMENT | FIXED (provenance, not stickiness)         |
| R1-006               | LOW      | VALID IMPROVEMENT | FIXED                                      |
| MIN_CONTENT_SECTIONS | LOW      | POTENTIAL RISK    | DEFERRED to Phase 8, reproduction recorded |
| `--model` help text  | —        | VALID IMPROVEMENT | FIXED                                      |

**No findings were classified FALSE POSITIVE.** Every numbered issue reproduced. Two
recommendations were not followed as written, in both cases because verification showed a
better fix; both are documented above with the reasoning and the evidence.

**Remaining OPEN findings: none.**

The reviewer's broader observation — that test names sometimes promised more than their bodies
checked — was acted on beyond the two specific instances: the safeguards cap test was renamed to
match what it verifies, and the new tests were written to include the inverse property in each
case (paraphrase must not be flagged; cosmetic duplicates must still collapse; a single-backend
compile must still report one model), so they cannot pass vacuously.
