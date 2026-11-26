# Phase 9 results — does team deliberation beat the lone surgeon?

**Date:** 2026-11-14/15 · **Protocol:** `phase9/protocol.md` (pre-registered before any condition;
condition 7 added user-approved before any committee had completed). All conditions: blind,
identical sandbox rules, 10-pass counter-enforced budget, identical final protocol (full
MATH-500, 500 problems, 32K cap, 1 sample, `phase3c/rescore.py` extraction), stock baseline 9.4%,
random-edit control 4.0%.

> **Corrected 2026-11-24 (ISSUES #19, #25, #27).** Every final below was re-scored with the
> answer checker of that day (ISSUES #19; it was corrected twice). First-reported -> reading of
> that day: glm5.3-flash-team 10.6 -> **15.2%**; glm5.3-team 24.4 -> **30.2%**;
> glm5.3-single 23.0 -> **29.6%**; mixed-team 23.6 -> **29.6%**; interleaved-team 9.0 ->
> **11.8%**; mimo-flash-team (format-corrected) 5.8 -> **8.2%**; stock 6.0 -> 8.4; random
> control 2.2 -> 3.8. The committee count, the verdict words of conditions 6 and 7 and the
> session census were corrected as well (ISSUES #25, #27). `phase9/progress.md` is the log as
> written during the runs and, apart from one sentence, keeps the first-reported scores.
> Practice numbers are as recorded with the old checker; the generations were not saved, so
> they cannot be re-scored.
>
> **Corrected 2026-11-26 (ISSUES #30, #31).** Every final below was read again with the final
> answer checker. Reading of 2026-11-24 -> final answer checker: glm5.3-flash-team 15.2 ->
> **17.0%**; glm5.3-team 30.2 -> **35.2%**; glm5.3-single 29.6 -> **31.8%**; mixed-team 29.6 ->
> **33.2%**; interleaved-team 11.8 -> **13.0%**; mimo-flash-team (format-corrected) 8.2 ->
> **9.2%**; stock 8.4 -> 9.4; random control 3.8 -> 4.0. No verdict word changes; the reference
> line for loners below is restated.

## The complete matrix

| # | Condition | Composition / format | Passes | Practice | **Final MATH-500** | Verdict (registered rule, ±4.4 pts) |
|---|---|---|---|---|---|---|
| 1 | glm5.3-flash-team | 6-role waterfall, one model | 6 | ~40 | **17.0%** | ❌ **HURTS** (20.2 below its loner's 37.2; interval −24.6 to −16.0) |
| 2 | mimo-flash-team | 6-role waterfall | 1 | — | no edit | ❌ NULL — degenerate session |
| 3 | mimo-pro-team | 6-role waterfall | 1 | — | no edit | ❌ NULL — degenerate session |
| 4 | glm5.3-team | 6-role waterfall | 2 | 48.4 | **35.2%** | ⚪ NEUTRAL (+3.4 vs 31.8; interval −0.4 to 7.2) |
| 5 | glm5.3-single | step-driven loner | 7 | ~57 | **31.8%** | (baseline for c4) |
| 6 | mixed-team | 4 models, 6 roles (waterfall) | 3 | 54.7 | **33.2%** | descriptive only: 9.0 points below the best single editor (42.2; interval −13.0 to −5.0) |
| 7 | interleaved-team | surgeon+critic live loop (GLM-Flash) | 3 | 28.1 | **13.0%** | descriptive only: 13.0 against 37.2 (−24.2; interval −28.4 to −20.2) |
| 8 | mimo-flash-team, format-corrected | short-ask 6-role | 3 | 17.2 | **9.2%** | ❌ **HURTS** (33.0 below the mimo-flash single editor at 42.2; interval −37.4 to −28.6) |
| 9 | mimo-pro-team, format-corrected | short-ask 6-role | 2 | — | no edit | ❌ NULL — cascade contamination |

**How the verdict column is filled (corrected 2026-11-24, ISSUES #27).** The registered rule
(`phase9/protocol.md`, "Pre-registered endpoints") gives HELPS at baseline +4.4 points or more,
NEUTRAL within ±4.4, HURTS at baseline −4.4 or less, and says of conditions 6–7: "compared
descriptively against all singles and the best single." So conditions 1, 4 and 8 carry a verdict
word; conditions 6 and 7 get descriptive sentences only. The earlier table labelled condition 6
"NEUTRAL" and condition 7 "HURTS", words the rule does not define for them.

**The tally (corrected 2026-11-24, ISSUES #25 and #27).** Eight committee conditions ran (1, 2, 3,
4, 6, 7, 8, 9). Five shipped an edit, and **none of the five beat the single editor it is compared
with.** Three were far worse: the GLM-5.3-Flash team 17.0% against 37.2% (-20.2), the interleaved
pair 13.0% against 37.2% (-24.2), the mimo-flash team 9.2% against 42.2% (-33.0). The GLM-5.3
team was not distinguishable from its loner (35.2% against 31.8%). The mixed team scored 33.2%,
9.0 points below the best single editor (42.2%; interval -13.0 to -5.0). Each committee is one
session, one seed. The other three conditions (2, 3 and 9) produced no edit at all because the
session format failed; those three are harness failures, not evidence about deliberation. (The
earlier text said "0 of 7 functioning-or-attempted committees beat their loner".)

**Which single edit is the comparator for conditions 6 and 8 (added 2026-11-24).** Both are
compared with the mimo-v2.6-flash single edit (42.2%, `phase3c/p4_mimo`): the best single editor
for condition 6, and the committee's own model for condition 8.

**Format-retry addendum (c8/c9, user-directed):** the corrected short-ask format rescued the
mimo-flash committee (complete session, vs zero output before) but not mimo-pro (analyst
monologues contaminate downstream asks — a cascade the format cannot fix).

Reference line for loners (final answer checker, 2026-11-26): mimo-flash 42.2, the 50-pass
GLM-Flash edit 41.6 (phase 10) and mimo-pro 40.6 are within noise of each other (differences of
0.6 to 1.6 points, every interval including zero). The 6-pass GLM-Flash edit (37.2, phase 4) is
not distinguishable from mimo-pro (+3.4, interval −1.0 to 7.6), is below mimo-flash (+5.0,
interval 0.6 to 9.2) and is below the 50-pass edit only at the edge (+4.4, interval 0.2 to
8.6). GLM-5.3 (31.8) is the lowest, 5.4 to 10.4 points below each of the other four (−5.4,
interval −9.6 to −1.2, against the 6-pass edit): under the checker of 2026-11-24 it was within
noise of the 6-pass GLM-Flash edit (−2.0, interval −6.0 to 2.0), and it is distinguishable from
it now. That checker's other line readings (ISSUES #31) are superseded: the final answer checker
reads either line (closed 2026-11-26). Original's hinted editor 54.2 (48.6 under the checker of
2026-11-24). One sample per problem, one seed: a single
arm has a standard error of about 2 points. (Until the second checker correction of that day
this line read "the first four are within noise of each other".)

## Four points from the matrix

### 1. No committee that shipped an edit beat the single editor it is compared with

Five committees shipped an edit: three single-family teams (GLM-5.3-Flash, GLM-5.3 and
mimo-v2.6-flash), one four-model mixed team and one interleaved pair. Under the registered rule
one is NEUTRAL — the GLM-5.3 team, 35.2% against its loner's 31.8% — and two are HURTS — the
GLM-5.3-Flash team and the mimo-flash team. The mixed team is significantly below the best single
editor, and the interleaved pair scores 13.0% against 37.2%. The highest committee score is 35.2%;
the single editors reach 31.8% to 42.2%. Three more committee attempts (conditions 2, 3 and 9)
produced no edit at all because the session format failed. Each committee is one session, one
seed, so this is a statement about these sessions. A pattern noticed after the results, not one
the protocol pre-registered: committees land level with their single editor or fall far below it;
none rose above. (Corrected 2026-11-24, ISSUES #25 and #27: the heading read "Deliberation never
beat the loner — 0 for 5", the text counted "two NEUTRALs, two HURTS" and "two additional
committee attempts", and it called "committees converge-or-collapse, never exceed" a
pre-registered pattern, which is not in the protocol.)

### 2. A reading offered after the results, not a finding: small committee doses

The three committees that scored far below their single editor all shipped doses the progress log
(`phase9/progress.md`) records as conservative: additive 0.135·u (c1, final 17.0), uniform +0.02
(c7, 13.0), and the smallest dose of the phase, an additive ‖Δ‖=0.063 on two rows (c8, 9.2). The
two ×1.35 teams did not collapse (c4, 35.2; c6, 33.2). One way to read this, offered after the
results and resting on one session per committee: the red-teams were *correct* about the risks
they measured (premature-close tails, spam cliffs at ×1.56–3.5 — all real), but minimizing
measured risk meant under-dosing the dominant failure — hard-problem loops that need the full
boost to terminate — so the empiricists' recklessness was, on finals, right, and practice sets
(easy, short) systematically flatter gentle doses where MATH-500 does not. The evidence does not
test that reading: each committee ran once and chose its own dose. (Corrected 2026-11-24,
ISSUES #27: this section was headed "The mechanism: committees ship conservative doses, and
conservative doses die at scale" and presented the reading as a finding.)

### 3. Committees innovate mechanisms; none of it beat the single editor

The methodological firsts recorded in the progress log came from deliberation: the untied-head
architecture (c1's judge), dose-bracket discipline with pre-registered escalation gates (c7's
critic — which correctly *refused* a dose escalation its own sims showed burning correct
answers), the depth-gated uniform-additive mechanism (c7's surgeon — a position-adaptive boost no
other arm found), and the sharpest falsifications (c6's mimo-flash red team demolishing a ×3.5
proposal by direct simulation before a pass was spent). None of it converted into a committee
that beat the single editor it is compared with. (Heading corrected 2026-11-24, ISSUES #25 and
#27, from "Committees innovate mechanisms; loners win scores".)

### 4. Process findings (harness science)

- **Session format dominated model tier in these sessions:** mimo tiers could not hold whole
  multi-role sessions (empty reports, leaked monologues — 3 failures across c2/c3 and phase 4)
  yet excelled in single focused seats (c6's red-team: the phase's best verification work).
  GLM-family models ran every format end-to-end. (Corrected 2026-11-24 to "in these sessions"; it
  said "Session format dominates model tier".)
- **Typed structured-submit kills mimo sessions** (two deaths); plain-text replies fixed it.
- **Pass economy:** interleaved was cheapest (3/10) with gates preventing waste; waterfall
  ranged 2–7; the loners used 6–7.
- **The exit-token spam cliff (×1.56–3.5 depending on mechanism) was independently rediscovered
  and measured by four different arms** — the strongest single fact in the project about where
  this surgery's danger zone sits.

## Convergent target census

Row 248069 (the `</think>` tied-embedding row) is the row every blind session that shipped an
edit chose. Corrected count (2026-11-24, ISSUES #25): 14 blind editor sessions ran in phases 4,
4b, 4c, 9 and 10 (phase 14, 2026-11-26, adds two sessions under the original authors' rules: one
shipped no edit, the other shipped one on a different tensor). **Ten of fourteen shipped an edit; all ten used the same row** — four blind
editor models (GLM-5.3, GLM-5.3-Flash, mimo-v2.6-flash, mimo-v2.6-pro), two families, every
format, one row of 873 million. **The four that shipped none failed on session format:** the
phase-4 mimo one-ask arm and phase-9 conditions 2, 3 and 9. In this phase that makes six shipping
sessions: the condition-5 loner and the five committees that shipped an edit (1, 4, 6, 7 and 8).
(The earlier text said "nine blind sessions" and "all four functioning committees".)

## Files

Per-condition records: `phase9/progress.md` · Sessions: `phase9/session_*/` · Finals:
`phase3c/p9_*.json` (+ gzipped generations) · Workflow scripts: `.zcode/workflow-drafts/Phase-9-*`
(archived with sessions where applicable).
