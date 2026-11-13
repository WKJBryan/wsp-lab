# Pre-registered protocol — Phase 9: team deliberation vs the lone surgeon

> **Provenance note:** this experiment was originally written as `phase8/protocol.md` and
> committed there (`2c6d099`/`b4e59af` on main), but a working-tree race with the parallel
> agent session put ITS phase-8 protocol ("the missing baselines") into that commit's content.
> The number was yielded; the parallel session's phase 8 keeps the number and its content
> stands (ISSUES #16). This file is the authoritative pre-registration of the deliberation
> experiment, committed before any of its conditions completed (condition 1 was in flight,
> results not yet known). Nothing else changed.

Question (user-directed 2026-11-13): does structured multi-agent deliberation — observe →
parallel analysis → red-team critique → judge decision → single executor — improve blind
weight surgery over the single-editor baselines of phases 4/4b/4c?

## Conditions (7, run serially)

| # | Condition | Team | Single-editor baseline (MATH-500) |
|---|---|---|---|
| 1 | glm5.3-flash-team | all 6 roles on GLM-5.3-Flash | 25.2% |
| 2 | mimo-flash-team | all on mimo-v2.6-flash | 27.8% |
| 3 | mimo-pro-team | all on mimo-v2.6-pro | 27.8% |
| 4 | glm5.3-team | all on GLM-5.3 | none exists → condition 5 supplies it |
| 5 | glm5.3-single | one editor, step-driven (phase-4b format) | — |
| 6 | mixed-team | roles split across all four models (assignment disclosed in results) | best single 27.8% |
| 7 | interleaved-team | analyst + red-team + executor work CONCURRENTLY — small edits proposed, attacked, applied and measured in a continuous loop (edit-as-you-think); GLM-5.3-Flash | same as condition 1 (25.2%) |

Condition 7 (added 2026-11-14, user-approved, before any team condition completed): single
editors already think-by-editing (dose bracketing); condition 7 tests whether interleaving
deliberation with live edits beats both the lone editor and the waterfall team.
**Pre-registered prediction (falsifiable): interleaved saves passes (fewer wasted cliff
discoveries) but lands in the same final-score band — the dose ceiling is a property of the
model, not the process.**

## Roles and rules (identical across team conditions 1–4, 6)

- **Observer**: spends pass 1 (baseline), reports the JSON verbatim + the 3 sample tails.
- **Analyst A / B** (parallel, independent): receive only the observer's evidence; may run any
  FREE offline probe in the sandbox; must NOT edit `model/` or spend passes; each returns ONE
  concrete edit proposal (tensor, rows, arithmetic, dose) with rationale and predicted risk.
- **Red-team** (fresh): receives both proposals; attacks them (dose cliffs, premature
  termination, scoring artifacts); may run free probes; returns ranked verdict with required
  modifications. No edits, no passes.
- **Judge** (fresh): receives proposals + critique; returns the FINAL exact edit spec.
- **Executor**: applies exactly the judge's spec (byte-exact backup first), then may bracket
  the dose ±25% within up to 4 further passes; reverts to backup freely; finalizes best.

Blindness: no hints anywhere; sandbox-only (soft boundary, transcripts audited); offline
analysis ungated — the SAME loophole as phase 4, symmetric with the single-editor baselines.

## Budget & mechanics

10 eval passes per condition, counter-enforced (GPU-busy refusals uncharged); fresh sterile
sandbox per condition; sessions archived to `phase9/session_<condition>/`. Finals = full
MATH-500 via the phase-3c protocol (500 problems, 32K, 1 sample, `phase3c/rescore.py`).

## Pre-registered endpoints

- **HELPS**: final ≥ baseline + **4.4 pts** (2 binomial SE at n=500).
- **NEUTRAL**: within ±4.4 of baseline. **HURTS**: ≤ baseline − 4.4.
- Condition 4 vs 5 (both new): same thresholds, 5 as its baseline.
- Conditions 6–7: compared descriptively against all singles and the best single.
- Secondary: passes spent, edit anatomy (site/dose vs the phase-4 convergence), and whether
  the red-team PREDICTED the dose cliff before any pass was spent on it (qualitative).
