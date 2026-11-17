# Phase 12 results — the cross-architecture atlas

**Date:** 2026-11-17 · **Protocol:** `phase12/protocol.md` (`6c87547`, pre-registered
before any probe) · Models: Qwen3.5-0.8B (dense, 24L, hidden 1024) vs Gemma-4 26B-A4B
pruned wound (MoE 8 experts, 30L, hidden 2816). Free analysis only; teacher-forced
passes over saved generations; identical probe code both sides.

## Verdict: **CORRESPONDENCE (2 of 3 tests passed)**

| Test | Question | Result | Pass |
|---|---|---|---|
| **C1** | Do loop-dynamics trajectories correspond? | **r = 0.96**, permutation p < 0.001 (0/1000 shifts); internal AUC 0.83 / 0.90 | ✅ |
| **C2** | Termination-circuit fingerprint match? | Behavioral: Qwen 71% relative termination recovery with accuracy **up** 33 pts; Gemma 57% recovery, accuracy unchanged. Statistical: both anchors in the **low tercile** of embedding-row norms (16.8th and 7.6th percentile) | ✅ |
| **C3** | Writer geometry (focal vs diffuse) distributions? | Both **focal** (no layer under 3.0 in either; ranges 6.2–18.2 vs 6.8–20.2), but KS = 0.55, p = 0.0003 — the distributions differ beyond the pre-registered p > 0.05 bar | ❌ |

## What the numbers say, plainly

1. **The descent into a loop has the same shape in both models.** Z-scored projections
   onto each model's own loop direction, binned around loop onset, correlate at **0.96**
   across architectures that share nothing but the transformer recipe — different
   vendors, 30× scale, dense vs MoE, 1024 vs 2816 hidden. The "falling into repetition"
   trajectory (flat ≈ −1σ before onset → smooth ramp through the transition → ~0 in the
   loop) is the same curve twice (`atlas_qwen.json`, `atlas_gemma.json`, trajectories).
2. **The exit door is small in both models.** The termination-critical embedding rows —
   Qwen's `</think>` (row 248069, the knob of phases 4–10) and Gemma's end_of_turn
   (token 106) — are both unusually *low-norm* rows (16.8th / 7.6th percentile), and in
   both models boosting them recovers termination (71% / 57% relative) without hurting
   accuracy. Weakly-embedded exits that need a push: a portable, checkable signature.
3. **Where they differ (the honest miss):** per-layer writer concentration. Both models
   are focal (a few columns/experts dominate the writing onto the loop direction —
   note this *refines* phase 7's "diffuse" finding, which compared across layers, not
   within), but Gemma's concentration profile is significantly steeper (KS p = 0.0003).
   The pre-registered bar demanded distribution equality; it failed as written. The
   structure agrees; the degree does not.

## Phase-13 setup (unchanged, now armed)

The transfer test's strongest asset is C2's low-norm signature: given a new model, the
candidate termination rows can be *found* statistically (low-norm eos-family rows) and
the operation class is already validated on both architectures (row boost → termination
recovery). C1's r = 0.96 says the disease's time-course is architecture-independent;
C3's miss says dose calibration will not transfer for free.

## Integrity log

First probe run: 7 usable Qwen texts (crude word-count split) → AUC 0.68; fixed to a
bimodal split over all 500 texts. Gemma run 1: 0 texts (fixed 500-token pre-window
clashed with the wound's early loops → adaptive). Gemma run 2: OOM — tensor-slice views
pinned ~20 GB of parent states (fix: clone + per-text buffer clearing). Gemma run 3:
expert-writer axis bug (fix: permute before reshape). Onset detector upgraded to
sustained-repetition (80% of a 300-token window repeated) after the naive first-triple
detector diluted sets. No pre-registered bar was touched at any point; the calculator
(`correspondence.py`) and its thresholds were committed before the first healthy atlas
existed.
