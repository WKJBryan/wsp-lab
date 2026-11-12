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
| 0 | start (dose 1.0) | Reference point of the starting model under practice conditions | _pending_ |
| 1 | dose 1.25 | Original dose may be sub-optimal; +25% more direction-removal cuts remaining loops | _pending_ |
| 2 | dose 1.5 | Dose curve upper region | _pending_ |
| 3 | dose 0.75 | Dose curve lower region (over-correction test: too much removal may hurt accuracy) | _pending_ |
