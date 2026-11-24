# Phase 9 results — does team deliberation beat the lone surgeon?

**Date:** 2026-11-14/15 · **Protocol:** `phase9/protocol.md` (pre-registered before any condition;
condition 7 added user-approved before any committee had completed). All conditions: blind,
identical sandbox rules, 10-pass counter-enforced budget, identical final protocol (full
MATH-500, 500 problems, 32K cap, 1 sample, `phase3c/rescore.py` extraction), stock baseline 8.4%,
random-edit control 3.8%.

> **Corrected 2026-11-24 (ISSUES #19, #25, #27).** Every final below was re-scored with the
> final answer checker (ISSUES #19; the checker was corrected twice that day).
> First-reported -> final: glm5.3-flash-team 10.6 -> **15.2%**; glm5.3-team 24.4 -> **30.2%**;
> glm5.3-single 23.0 -> **29.6%**; mixed-team 23.6 -> **29.6%**; interleaved-team 9.0 ->
> **11.8%**; mimo-flash-team (format-corrected) 5.8 -> **8.2%**; stock 6.0 -> 8.4; random
> control 2.2 -> 3.8. The committee count, the verdict words of conditions 6 and 7 and the
> session census were corrected as well (ISSUES #25, #27). `phase9/progress.md` is the log as
> written during the runs and, apart from one sentence, keeps the first-reported scores.
> Practice numbers are as recorded with the old checker; the generations were not saved, so
> they cannot be re-scored.

## The complete matrix

| # | Condition | Composition / format | Passes | Practice | **Final MATH-500** | Verdict (registered rule, ±4.4 pts) |
|---|---|---|---|---|---|---|
| 1 | glm5.3-flash-team | 6-role waterfall, one model | 6 | ~40 | **15.2%** | ❌ **HURTS** (16.4 below its loner's 31.6; interval −20.6 to −12.2) |
| 2 | mimo-flash-team | 6-role waterfall | 1 | — | no edit | ❌ NULL — degenerate session |
| 3 | mimo-pro-team | 6-role waterfall | 1 | — | no edit | ❌ NULL — degenerate session |
| 4 | glm5.3-team | 6-role waterfall | 2 | 48.4 | **30.2%** | ⚪ NEUTRAL (+0.6 vs 29.6; interval −3.2 to 4.4) |
| 5 | glm5.3-single | step-driven loner | 7 | ~57 | **29.6%** | (baseline for c4) |
| 6 | mixed-team | 4 models, 6 roles (waterfall) | 3 | 54.7 | **29.6%** | descriptive only: 6.6 points below the best single editor (36.2; interval −10.6 to −2.8) |
| 7 | interleaved-team | surgeon+critic live loop (GLM-Flash) | 3 | 28.1 | **11.8%** | descriptive only: 11.8 against 31.6 (−19.8; interval −24.0 to −15.8) |
| 8 | mimo-flash-team, format-corrected | short-ask 6-role | 3 | 17.2 | **8.2%** | ❌ **HURTS** (28.0 below the mimo-flash single editor at 36.2; interval −32.4 to −23.8) |
| 9 | mimo-pro-team, format-corrected | short-ask 6-role | 2 | — | no edit | ❌ NULL — cascade contamination |

**How the verdict column is filled (corrected 2026-11-24, ISSUES #27).** The registered rule
(`phase9/protocol.md`, "Pre-registered endpoints") gives HELPS at baseline +4.4 points or more,
NEUTRAL within ±4.4, HURTS at baseline −4.4 or less, and says of conditions 6–7: "compared
descriptively against all singles and the best single." So conditions 1, 4 and 8 carry a verdict
word; conditions 6 and 7 get descriptive sentences only. The earlier table labelled condition 6
"NEUTRAL" and condition 7 "HURTS", words the rule does not define for them.

**The tally (corrected 2026-11-24, ISSUES #25 and #27).** Eight committee conditions ran (1, 2, 3,
4, 6, 7, 8, 9). Five shipped an edit, and **none of the five beat the single editor it is compared
with.** Three were far worse: the GLM-5.3-Flash team 15.2% against 31.6% (-16.4), the interleaved
pair 11.8% against 31.6% (-19.8), the mimo-flash team 8.2% against 36.2% (-28.0). The GLM-5.3
team was not distinguishable from its loner (30.2% against 29.6%). The mixed team scored 29.6%,
6.6 points below the best single editor (36.2%; interval -10.6 to -2.8). Each committee is one
session, one seed. The other three conditions (2, 3 and 9) produced no edit at all because the
session format failed; those three are harness failures, not evidence about deliberation. (The
earlier text said "0 of 7 functioning-or-attempted committees beat their loner".)

**Which single edit is the comparator for conditions 6 and 8 (added 2026-11-24).** Both are
compared with the mimo-v2.6-flash single edit (36.2%, `phase3c/p4_mimo`): the best single editor
for condition 6, and the committee's own model for condition 8.

**Format-retry addendum (c8/c9, user-directed):** the corrected short-ask format rescued the
mimo-flash committee (complete session, vs zero output before) but not mimo-pro (analyst
monologues contaminate downstream asks — a cascade the format cannot fix).

Reference line for loners (final answer checker, 2026-11-24): mimo-flash 36.2, the 50-pass
GLM-Flash edit 36.0 (phase 10) and mimo-pro 35.2 are within noise of each other, and above the
6-pass GLM-Flash edit (31.6, phase 4) and GLM-5.3 (29.6), which are within noise of each other.
Two pairs are less clear: mimo-pro is not distinguishable from the 6-pass GLM-Flash edit (+3.6,
interval −0.6 to 7.6), and the 50-pass edit is above the 6-pass one only at the edge (+4.4,
interval 0.2 to 8.6). Original's hinted editor 48.6. One sample per problem, one seed: a single
arm has a standard error of about 2 points. (Until the second checker correction of that day
this line read "the first four are within noise of each other".)

## Four points from the matrix

### 1. No committee that shipped an edit beat the single editor it is compared with

Five committees shipped an edit: three single-family teams (GLM-5.3-Flash, GLM-5.3 and
mimo-v2.6-flash), one four-model mixed team and one interleaved pair. Under the registered rule
one is NEUTRAL — the GLM-5.3 team, 30.2% against its loner's 29.6% — and two are HURTS — the
GLM-5.3-Flash team and the mimo-flash team. The mixed team is significantly below the best single
editor, and the interleaved pair scores 11.8% against 31.6%. The highest committee score is 30.2%;
the single editors reach 29.6% to 36.2%. Three more committee attempts (conditions 2, 3 and 9)
produced no edit at all because the session format failed. Each committee is one session, one
seed, so this is a statement about these sessions. A pattern noticed after the results, not one
the protocol pre-registered: committees land level with their single editor or fall far below it;
none rose above. (Corrected 2026-11-24, ISSUES #25 and #27: the heading read "Deliberation never
beat the loner — 0 for 5", the text counted "two NEUTRALs, two HURTS" and "two additional
committee attempts", and it called "committees converge-or-collapse, never exceed" a
pre-registered pattern, which is not in the protocol.)

### 2. A reading offered after the results, not a finding: small committee doses

The three committees that scored far below their single editor all shipped doses the progress log
(`phase9/progress.md`) records as conservative: additive 0.135·u (c1, final 15.2), uniform +0.02
(c7, 11.8), and the smallest dose of the phase, an additive ‖Δ‖=0.063 on two rows (c8, 8.2). The
two ×1.35 teams did not collapse (c4, 30.2; c6, 29.6). One way to read this, offered after the
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
4b, 4c, 9 and 10. **Ten of fourteen shipped an edit; all ten used the same row** — four blind
editor models (GLM-5.3, GLM-5.3-Flash, mimo-v2.6-flash, mimo-v2.6-pro), two families, every
format, one row of 873 million. **The four that shipped none failed on session format:** the
phase-4 mimo one-ask arm and phase-9 conditions 2, 3 and 9. In this phase that makes six shipping
sessions: the condition-5 loner and the five committees that shipped an edit (1, 4, 6, 7 and 8).
(The earlier text said "nine blind sessions" and "all four functioning committees".)

## Files

Per-condition records: `phase9/progress.md` · Sessions: `phase9/session_*/` · Finals:
`phase3c/p9_*.json` (+ gzipped generations) · Workflow scripts: `.zcode/workflow-drafts/Phase-9-*`
(archived with sessions where applicable).
