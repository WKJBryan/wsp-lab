# Phase 10 results — the no-hint FULL-BUDGET run (the field's last unclaimed experiment)

**Date:** 2026-11-15 · **Protocol:** `phase10/protocol.md` (pre-registered, with a falsifiable
prediction). Editor: GLM-5.3-Flash, blind, 50 passes available, step-driven format.

## The closing table

| Editor | Hint? | Passes | MATH-500 | Notes |
|---|---|---|---|---|
| Stock | — | — | **6.0%** | baseline |
| GLM-5.3-Flash (phase 4) | no | 6 of 10 | **25.2%** | 3-row tied-embedding rescale |
| **GLM-5.3-Flash (this phase)** | **no** | **6 of 50** | **27.4%** | row 248069 ×1.5 + comma ×0.8 |
| Original's editor (published edit) | yes | 50 | **39.4%** (our stack) / 48.1% (theirs) | 2-tensor down_proj surgery |

## What the 50-pass blind run established

1. **Budget was not the binding constraint — knowledge was.** Offered 8× the budget, the
   editor spent 6 and declared the mechanism space exhausted: +2.2 points over the 6-pass run
   (25.2 → 27.4, within the ±4.4 noise bar = NEUTRAL). Tenth convergence on row 248069.
2. **Why more passes cannot help (the editor's own accounting, verified in its logs):** after
   the exit-row fix, remaining failures are ~12/64 genuine arithmetic errors ("needs training —
   forbidden"), ~4 formatting artifacts, ~3 residual loops. Weight-space levers reachable from
   behavior observation were exhausted in six measurements; the rest is the model's ability
   ceiling, not a discoverable defect.
3. **What the hint actually buys, now measurable:** every blind editor — ten sessions, every
   format — stops surgery at the OUTPUT row (the exit token's embedding). The hinted original
   performs a DEEPER operation: subtracting a measured direction from 64 down_proj columns in
   layers 15/19 (downstream of the representation, not the readout). That deeper route is what
   the hint's value consists of: not pointing at the disease (everyone finds it blind) but
   certifying that a more invasive surgical route exists and is worth the risk. Blind editors,
   given no such assurance, rationally stay at the surface and plateau at ~25–28%.
4. **Prediction scorecard (honesty section):** pre-registered prediction was 28–34% closing
   ~⅓ of the hint gap. Actual: 27.4%, closing ~15% (vs our-stack hinted 39.4). Partially
   refuted — the miss is itself the finding: I overestimated what extra measurement budget
   could buy and underestimated how fast blind editors exhaust the safe mechanism space.

## The complete project ledger (all on main, every number reproducible)

Blind loners: mimo-flash 27.8 = mimo-pro 27.8 > GLM-flash 25.2 (6-pass) / 27.4 (50-pass,
this phase) > GLM-5.3 23.0. Committees: 0 of 7 beat their loner. Random control 2.2.
Original's edit (hinted): 39.4 our stack. Ten blind sessions converged on one row of 873M.
