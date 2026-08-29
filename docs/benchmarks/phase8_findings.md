# Phase 8 — evaluation findings

Run 2026-08-29. Reproduce with the sweep in `prompt_compiler.evaluation`; 46,080 points,
about 21 seconds, no network.

## What could and could not be measured

**No live model call was possible.** `.env` holds Gemini and Groq keys and neither has an
adapter until Phase 9; the one implemented provider (Anthropic) has no key. So every
outcome question — does compiling a prompt produce better answers, does the two-call split
earn its second call — remains **unanswered**, not answered weakly. Nothing in this
document is evidence that the compiler improves anything.

What was measurable is the **deterministic policy layer**: given every combination of
evidence an analysis can carry, how do the section policy, the safeguard policy and the
Phase 7 scores behave? That is answerable exhaustively and offline.

### Why a sweep rather than the corpus

The twelve corpus analyses are hand-written fixtures by the same hand that wrote the
policies, so measuring policy against them would largely confirm its own assumptions, on a
sample far too small to show a distribution. Enumerating the input space answers a narrower
question completely instead of a broad one badly.

**The sweep gives unweighted coverage of the input space, not a frequency estimate.** Its
grid weights `many_high` risk at a third of all points, which no real prompt population
does. Rates below describe reachability, never how often something happens in practice.

## Finding 1 — the section floor measured the wrong thing (FIXED)

`plan_sections` counted section *categories*. A prompt stating **four requirements and two
constraints — six concrete items — fell in two categories and was written as unstructured
prose**, while a prompt stating a *single* requirement in a complex task cleared three
categories (REQUIREMENTS, PROCESS, FACTUALITY RULES) and earned four headings.

This is the deferred `MIN_CONTENT_SECTIONS` question from the adversarial review, and it is
the mirror image of the inversion the Phases 3–4 pass fixed: that one was scaffolding over
prompts stating *nothing*, this is prose over prompts stating *a lot*.

**Fix:** `MIN_CONTENT_ITEMS = 4` in `sections.py` — a prompt earns structure on either
enough categories *or* enough stated list items.

| threshold | prose rate | inversion configs | most content left as prose |
|---|---|---|---|
| off (before) | 20.3% | 36 | **6 items** |
| 6 | 20.2% | 34 | — |
| 5 | 20.1% | 32 | — |
| **4 (chosen)** | **17.8%** | **25** | **3 items** |
| 3 | 17.4% | 18 | — |
| 2 | 11.8% | 5 | — |

Four is the smallest value that closes the severe case without the category floor being
subsumed — three would let the volume rule swallow it and flip a large block of
modestly-specified prompts into headings at once, reintroducing over-scaffolding.
Regression tests: `test_many_items_in_few_categories_still_earn_structure` and
`test_the_volume_floor_does_not_scaffold_a_prompt_that_says_little`.

## Finding 2 — tool-added sections still count toward the floor (OPEN, deferred)

25 milder inversions survive, all from one cause: **PROCESS and FACTUALITY RULES are
sections this tool adds, and they count toward the threshold that decides whether the
user's content deserves headings.** A prompt with one requirement and one high-severity
ambiguity reaches three categories and gets four headings; a prompt stating three things in
two categories gets prose.

An alternative was measured — count only user-earned categories toward the floor:

| variant | prose rate | thinnest structured point | inversion configs |
|---|---|---|---|
| current | 17.8% | 1 stated item | 25 |
| user-earned only | **43.8%** | 3 stated items | **0** |

It eliminates inversions completely, and it **more than doubles the prose rate**, turning
off structure for complex tasks with few requirements — including the PROCESS section
Phase 4 added deliberately for multi-component work.

**Not applied.** Whether headings help an answering model is exactly the question that
needs the live A/B, and a change of that size resting on an aesthetic consistency argument
is the kind of unmeasured assertion this project forbids. Phase 9 unblocks the experiment;
the variant is specified above and takes one commit once there is evidence.

Verified safe on one axis already: `generate()` weaves safeguards into a prose prompt
without a heading (generator.py:64-70), so excluding FACTUALITY RULES from the floor would
not lose the safeguards themselves.

## Finding 3 — the Phase 7 scoring instrument is structurally sound

Across all 46,080 points:

| dimension | range | mean | distinct |
|---|---|---|---|
| clarity | 40..100 | 80.4 | 5 |
| specificity | 0..100 | 76.6 | 9 |
| completeness | 0..100 | 56.2 | 4 |
| requirement_coverage | 0..100 | 90.6 | 6 |
| risk_coverage | 0..100 | 70.7 | 4 |
| token_efficiency | 0..100 | 85.4 | 6 |

- **No dead dimension.** Every one moves; none is pinned.
- **No redundant pair.** The strongest correlation between any two is
  `risk_coverage / token_efficiency` at **0.282** — far below the 0.95 bar. The six are
  genuinely independent measurements, so none is double-counted in a group mean.
- Low distinct counts reflect the grid's coarse axes, not the scales.

This validates the *structure* of the instrument. It says nothing about calibration —
whether a higher score means a better prompt is still unmeasured.

## Finding 4 — the safeguard cap discriminates

`MAX_SAFEGUARDS = 4` is reachable and fires on some inputs but not all, so it is a real
threshold rather than dead configuration. Earned-rule counts spanned 0–6, so up to two
earned rules can be discarded. Whether the cap's *value* is right needs outcome data.

## Finding 5 — a defect in the harness's own metric (FIXED)

`prose_inversions` originally paired every prose point against every structured one — a
cartesian product returning **19,854,144 rows** where 36 distinct configurations was the
meaningful answer. An instrument that reports a number nobody can act on is worse than one
that reports nothing. Now counted by configuration, and pinned by a test.

## Still unverified — what Phase 9 must answer

1. **Does compiling help?** Answer each corpus prompt raw and compiled, compare. Needs a
   provider plus a judging protocol, which should be designed against real output rather
   than guessed at now.
2. **Does the two-call split earn its second call?** Compare the current pipeline against a
   single call returning analysis and draft together. The fallback design is recorded in
   `project_memory.json`. Costs 48–72 calls/day at expected usage.
3. **Calibrate every Phase 7 weight.** The penalties, targets and band edges are anchored
   to existing policy constants and gathered in one block in `scoring.py`; none is
   calibrated against outcomes.
4. **Finding 2's section-floor variant**, above.
5. **Exact token counts**, if a headline claim needs them — requires `count_tokens` on the
   `Provider` ABC and the contract suite.
