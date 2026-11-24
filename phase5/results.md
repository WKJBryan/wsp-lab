# Phase 5 results — optimizing beyond the original's edit: NULL on finals, with a measured trade-off

**Date:** 2026-11-13 · **Protocol:** `phase5/protocol.md` (committed `62e3de7` before any run)
**Engine:** vLLM 0.33 / WSL2 / RTX 5090 · **Finals:** 500 MATH-500 × 4 sequential samples, cap 32,768, think35 decoding, format-aware extraction.

> **Corrected 2026-11-24 (ISSUES #19).** Both finals were re-scored with the final answer
> checker (`uv run python phase5/rescore_finals.py`; the checker was corrected twice that day):
> the published edit at dose 1.0× scores 47.5% (per sample 48.2, 47.0, 48.6, 46.2; 38.2% as first
> reported) and at dose 1.75× 42.7% (per sample 43.0, 42.0, 42.4, 43.4; 35.4% as first reported).
> The verdict, NULL, stands. The generations of both finals are now committed
> (`phase5/final5v2_start*`, `phase5/final5v2_dose175*`), recovered from the Linux
> side on 2026-11-24; at run time they were scored by an extractor inside `final_vllm2.py` that
> was not the repository's. Practice numbers are as recorded with the old checker; the
> generations were not saved, so they cannot be re-scored.

## Verdict (pre-registered: SUCCESS ≥ baseline+5.0, PARTIAL +2.5–5.0)

**NULL — the best practice candidate (dose 1.75×) scored 42.7% against the baseline's 47.5% (dose
1.0×). Paired over the 2,000 problem × sample pairs, the higher dose is 4.8 points worse: 238
pairs right only at 1.0, 142 only at 1.75; 95% interval 2.8 to 6.8; p = 9.7e-7. A real
regression, decisively not an improvement.** (Corrected 2026-11-24, ISSUES #19: the scores were
35.4% and 38.2% as first reported, and this read "−2.8, ≈1.9σ of the arm difference: suggestive
regression". The NULL verdict is now statistically solid where the original text could only call
it suggestive.)

## What was established

1. **The original's edit anatomy (measured, ISSUES #14):** exactly 131,072 values — 64 of 3,584
   input columns × 1,024 rows × 2 layers (`down_proj` L15/L19); everything else bit-identical.
   Exactly rank-1 per layer (top singular share 1.000): one direction with per-column
   amplitudes, additive. cos(v15, v19) = +0.849 — effectively ONE residual-stream direction
   at two depths. Our earlier "56× more than claimed" was our own diff-script artifact.
2. **Dose curve on practice (64 GSM8K, cap 8192):** monotone improvement 0.75×→1.75×
   (39.1 → 50.0% accuracy; stuck 53.1 → 25.0%), cliff onset by 2.0× (45.3%, stuck 39.1).
3. **Finals reverse the practice picture:** dose 1.75× = 42.7% / 21.9% stuck vs dose 1.0× =
   47.5% / 26.6% (accuracies re-scored 2026-11-24, ISSUES #19; 35.4% and 38.2% as first reported;
   stuck rates unchanged). More termination, worse answers. **Interpretation: the loop direction
   also participates in legitimate hard-problem reasoning; suppressing it harder helps easy
   problems finish and hurts deep ones.** The original authors' dose is well-balanced for the
   actual benchmark — our "under-dosed by 43%" reading from practice was wrong, and the
   practice-set choice (GSM8K) was one confound. A second, which the earlier text missed:
   practice ran at an 8,192-token cap, finals at 32,768, so more termination helps more on
   practice. Lesson recorded: practice on the target distribution (MATH train), not a proxy.
4. **Consistency checks (harness health):** baseline n=4 = 47.5% vs phase-3c single-sample
   48.6% — same model, same engine, two sampling designs, agreeing. Also refines the
   authors-comparison: 4-sample averaging does NOT lift our number (their 48.1 vs our 47.5
   at matched averaging — so the residual gap is not sample count; which of the other known
   differences carries it is untested, see `phase3c/results.md`).

## Cost ledger

~10 GPU-hours finals (attempt #1 void: ISSUES #13), ~1 h practice passes, free anatomy.
Practice-pass budget: 6 of 12 used.

## What Phase 6 inherits

- Dose axis: measured dead end on MATH-500 (both directions from 1.0 lose: 0.75 tied practice,
  1.75 wins practice/loses finals). Family 2 (asymmetric L15/L19) and family 1 (L23 extension
  — atlas Finding 3) remain the live hypotheses; family 4 (activation-refined direction) gains
  value given the trade-off finding: the goal is now SELECTIVE suppression (loops only), not
  more suppression.
