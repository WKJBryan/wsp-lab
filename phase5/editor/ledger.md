# Editor ledger — Phase 5 (optimization; not blind — all prior results known, disclosed)

Budget: 12 practice passes (64 problems, cap 8192, same decoding as phase 3c, seed offset per pass).

## Free analysis findings (analyze_edit.py, 2026-11-12)

- Their edit is **exactly 131,072 values** (64 of 3,584 input columns × 1,024 rows × 2 layers);
  all other columns bit-identical — the earlier "7.34M values / 56×" claim was OUR diff-script
  counting whole-tensor sizes; correcting the record in ISSUES.
- **Exactly rank-1 per layer** (top singular share 1.000): one direction v subtracted with
  per-column amplitudes (median |a| ≈ 0.12–0.14, mixed signs). NOT column scaling
  (residual 99.5%) — additive direction subtraction, matching the blog's description.
- **cos(v15, v19) = +0.849**: effectively ONE residual-stream direction, suppressed at two
  depths; the 64-column sets are disjoint across layers (Jaccard 0.02).

## Budget ledger

| # | candidate | hypothesis | result |
|---|-----------|------------|--------|
| 0 | start (dose 1.0) | Reference point of the starting model under practice conditions | acc **39.1**, stuck 48.4, closed 56.2, mean tok 5,911 |
| 1 | dose 1.25 | Original dose may be sub-optimal; +25% more direction-removal cuts remaining loops | acc **42.2**, stuck 43.8, closed 60.9, mean tok 5,560 |
| 2 | dose 1.5 | Dose curve upper region | acc **48.4**, stuck 35.9, closed 67.2, mean tok 5,187 |
| 3 | dose 0.75 | Dose curve lower region (over-correction test: too much removal may hurt accuracy) | acc **39.1**, stuck 53.1, closed 50.0, mean tok 6,191 |

Curve is MONOTONE: accuracy ↑ and stuck ↓ with dose through 1.5×; no over-correction cliff yet;
samples remain coherent reasoning. Decision per protocol (+9.3 ≥ 8): continue upward.
Operator note: passes 0–3 were run twice because the first run's JSON summaries were lost to a
`tail -1` log filter (shutdown lines print after the result). Same seeds/questions — one set of
information; counted once against budget. Output now saved per pass (`pass_*.log` in WSL).
Budget used: 4 of 12.

| # | candidate | hypothesis | result |
|---|-----------|------------|--------|
| 4 | dose 1.75 | Curve still rising at 1.5; find the shoulder | acc **50.0**, stuck 25.0, closed 75.0 — practice peak |
| 5 | dose 2.0 | Curve upper region; watch for the cliff | acc **45.3**, stuck 39.1 — cliff onset confirmed |

## Session close (2026-11-13)

- Finals (500 MATH-500 × 4 samples each, `final_vllm2.py`): dose 1.0 = **38.2%** / 26.6% stuck;
  dose 1.75 = **35.4%** / 21.9% stuck. Pre-registered verdict: **NULL** (−2.8, not +5).
- The practice peak was a GSM8K-practice confound: harder problems need the direction for
  real reasoning. More suppression ⇒ more termination but worse answers on hard math.
- Attempt #1 of the finals was void (engine memory race + 2,000-in-flight token soup —
  ISSUES #13; probe exonerated n=4 and depth); v2 protocol (4 sequential n=1 passes,
  no prefix caching) ran clean with per-pass sanity tails.
- Budget used: 6 of 12 practice passes. Phase 6 inherits: dose axis dead on MATH-500;
  selective suppression (not more suppression) is the new target.
