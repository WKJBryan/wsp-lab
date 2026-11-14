# Phase 9 results — does team deliberation beat the lone surgeon?

**Date:** 2026-11-14/15 · **Protocol:** `phase9/protocol.md` (pre-registered before any condition;
condition 7 added user-approved before any committee had completed). All conditions: blind,
identical sandbox rules, 10-pass counter-enforced budget, identical final protocol (full
MATH-500, 500 problems, 32K cap, 1 sample, `phase3c/rescore.py` extraction), stock baseline 6.0%,
random-edit control 2.2%.

## The complete matrix

| # | Condition | Composition / format | Passes | Practice | **Final MATH-500** | Verdict (vs same-model loner, ±4.4 pts) |
|---|---|---|---|---|---|---|
| 1 | glm5.3-flash-team | 6-role waterfall, one model | 6 | ~40 | **10.6%** | ❌ **HURTS** (vs 25.2) |
| 2 | mimo-flash-team | 6-role waterfall | 1 | — | no edit | ❌ NULL — degenerate session |
| 3 | mimo-pro-team | 6-role waterfall | 1 | — | no edit | ❌ NULL — degenerate session |
| 4 | glm5.3-team | 6-role waterfall | 2 | 48.4 | **24.4%** | ⚪ NEUTRAL (+1.4 vs 23.0) |
| 5 | glm5.3-single | step-driven loner | 7 | ~57 | **23.0%** | (baseline for c4) |
| 6 | mixed-team | 4 models, 6 roles (waterfall) | 3 | 54.7 | **23.6%** | ⚪ NEUTRAL (vs best single 27.8) |
| 7 | interleaved-team | surgeon+critic live loop (GLM-Flash) | 3 | 28.1 | **9.0%** | ❌ **HURTS** (vs 25.2) |

Reference: loners under identical rules — mimo-flash 27.8 = mimo-pro 27.8 > GLM-flash 25.2 >
GLM-5.3 23.0; original's hinted editor 39.4.

## The four findings

### 1. Deliberation never beat the loner — 0 for 5

Five functioning committee formats (same-family ×2, cross-family ×4-model, interleaved) produced
two NEUTRALs, two HURTS, and nothing above 24.4% — while the loners they were matched against
reached 25.2–27.8%. Two additional committee attempts (mimo same-family) failed to produce any
edit at all. The pre-registered pattern (committees converge-or-collapse, never exceed) held in
every cell of the matrix.

### 2. The mechanism: committees ship conservative doses, and conservative doses die at scale

Every committee's shipped edit was more conservative than its loner counterpart's: additive
0.135·u (c1, final 10.6), ×1.35 (c4, 24.4; c6 tied ×1.35, 23.6), uniform +0.02 (c7, 9.0) — while
the loners' crude empirical ×1.5 survives finals (25.2–27.8). The red-teams were *correct* about
the risks they measured (premature-close tails, spam cliffs at ×1.56–3.5 — all real), but
minimizing measured risk meant under-dosing the dominant failure: hard-problem loops that need
the full boost to terminate. The empiricists' recklessness was, on finals, right. Practice sets
(easy, short) systematically flatter gentle doses; MATH-500 does not.

### 3. Committees innovate mechanisms; loners win scores

Every methodological first of the phase came from deliberation: the untied-head architecture
(c1's judge), dose-bracket discipline with pre-registered escalation gates (c7's critic — which
correctly *refused* a dose escalation its own sims showed burning correct answers), the
depth-gated uniform-additive mechanism (c7's surgeon — a position-adaptive boost no other arm
found), and the sharpest falsifications (c6's mimo-flash red team demolishing a ×3.5 proposal by
direct simulation before a pass was spent). None of it converted into a better final score.

### 4. Process findings (harness science)

- **Session format dominates model tier:** mimo tiers cannot hold whole multi-role sessions
  (empty reports, leaked monologues — 3 failures across c2/c3 and phase 4) yet excel in single
  focused seats (c6's red-team: the phase's best verification work). GLM-family models ran every
  format end-to-end.
- **Typed structured-submit kills mimo sessions** (two deaths); plain-text replies fixed it.
- **Pass economy:** interleaved was cheapest (3/10) with gates preventing waste; waterfall
  ranged 2–7; the loners used 6–7.
- **The exit-token spam cliff (×1.56–3.5 depending on mechanism) was independently rediscovered
  and measured by four different arms** — the strongest single fact in the project about where
  this surgery's danger zone sits.

## Convergent target census

Row 248069 (the `</think>` tied-embedding row) was independently selected by: GLM-flash loner,
mimo-flash, mimo-pro, GLM-5.3 loner, and all four functioning committees — **nine blind sessions,
two families, every format, one row of 873 million.**

## Files

Per-condition records: `phase9/progress.md` · Sessions: `phase9/session_*/` · Finals:
`phase3c/p9_*.json` (+ gzipped generations) · Workflow scripts: `.zcode/workflow-drafts/Phase-9-*`
(archived with sessions where applicable).
