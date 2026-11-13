# Phase 5 results — optimizing beyond the original's edit: NULL on finals, with a measured trade-off

**Date:** 2026-11-13 · **Protocol:** `phase5/protocol.md` (committed `9cef8aa` before any run)
**Engine:** vLLM 0.33 / WSL2 / RTX 5090 · **Finals:** 500 MATH-500 × 4 sequential samples, cap 32,768, think35 decoding, format-aware extraction.

## Verdict (pre-registered: SUCCESS ≥ baseline+5.0, PARTIAL +2.5–5.0)

**NULL — the best practice candidate (dose 1.75×) scored 35.4% vs baseline 38.2% (−2.8, ≈1.9σ
of the arm difference: suggestive regression, decisively not an improvement).**

## What was established

1. **The original's edit anatomy (measured, ISSUES #14):** exactly 131,072 values — 64 of 3,584
   input columns × 1,024 rows × 2 layers (`down_proj` L15/L19); everything else bit-identical.
   Exactly rank-1 per layer (top singular share 1.000): one direction with per-column
   amplitudes, additive. cos(v15, v19) = +0.849 — effectively ONE residual-stream direction
   at two depths. Our earlier "56× more than claimed" was our own diff-script artifact.
2. **Dose curve on practice (64 GSM8K, cap 8192):** monotone improvement 0.75×→1.75×
   (39.1 → 50.0% accuracy; stuck 53.1 → 25.0%), cliff onset by 2.0× (45.3%, stuck 39.1).
3. **Finals reverse the practice picture:** dose 1.75× = 35.4% / 21.9% stuck vs dose 1.0× =
   38.2% / 26.6%. More termination, worse answers. **Interpretation: the loop direction also
   participates in legitimate hard-problem reasoning; suppressing it harder helps easy
   problems finish and hurts deep ones.** The original authors' dose is well-balanced for the
   actual benchmark — our "under-dosed by 43%" reading from practice was wrong, and the
   practice-set choice (GSM8K) was the confound. Lesson recorded: practice on the target
   distribution (MATH train), not a proxy.
4. **Consistency checks (harness health):** baseline n=4 = 38.2% vs phase-3c single-sample
   39.4% — same model, same engine, two sampling designs, agreeing. Also refines the
   authors-comparison: 4-sample averaging does NOT lift our number (their 48.1 vs our 38.2
   at matched averaging — the residual gap is grader/settings, not sample count).

## Cost ledger

~10 GPU-hours finals (attempt #1 void: ISSUES #13), ~1 h practice passes, free anatomy.
Practice-pass budget: 6 of 12 used.

## What Phase 6 inherits

- Dose axis: measured dead end on MATH-500 (both directions from 1.0 lose: 0.75 tied practice,
  1.75 wins practice/loses finals). Family 2 (asymmetric L15/L19) and family 1 (L23 extension
  — atlas Finding 3) remain the live hypotheses; family 4 (activation-refined direction) gains
  value given the trade-off finding: the goal is now SELECTIVE suppression (loops only), not
  more suppression.
