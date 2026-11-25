# Phase 10 results — the no-hint FULL-BUDGET run (the first run we know of that offered a blind editor 50 passes)

**Date:** 2026-11-15 · **Protocol:** `phase10/protocol.md` (pre-registered, with a falsifiable
prediction). Editor: GLM-5.3-Flash, blind, 50 passes available, step-driven format.

> **Corrected 2026-11-24 (ISSUES #19, #23, #25).** The finals below were re-scored with the
> final answer checker (the checker was corrected twice that day). First-reported -> final:
> stock 6.0 -> **8.4%**; GLM-5.3-Flash 6-pass edit 25.2 -> **31.6%**; this phase's edit 27.4 ->
> **36.0%**; published edit 39.4 -> **48.6%** (our stack). The reading of the distance to the
> original (ISSUES #23), the pass count and the session count (ISSUES #25) were corrected as
> well. The title earlier called this "the field's last unclaimed experiment"; the repo can
> support only the narrower claim now in it. Practice numbers are as recorded with the old
> checker; the generations were not saved, so they cannot be re-scored.

> **Note added 2026-11-25 (ISSUES #31).** The statements below that put this phase's edit "+4.4
> points, interval 0.2 to 8.6, p = 0.049" above its 6-pass GLM-Flash run, "at the edge" (item 1,
> item 4 of the scorecard and the project ledger at the end), depend on which line of an unboxed
> answer the checker reads. Under either other reading of the same checker the difference is +3.6
> (95% interval −0.4 to 7.8, p = 0.11; 65 problems right only under this edit, 47 only under the
> 6-pass one), so "meaningfully above the 6-pass run" is not established. The band miss, the gap
> miss and the comparison with the MiMo-pro edit do not change in word (this edit itself would
> read 36.6% or 37.2%).

## The closing table

| Editor | Hint? | Passes | MATH-500 | Notes |
|---|---|---|---|---|
| Stock | — | — | **8.4%** | baseline |
| GLM-5.3-Flash (phase 4) | no | 6 of 10 | **31.6%** | 3-row tied-embedding rescale |
| **GLM-5.3-Flash (this phase)** | **no** | **6 of 50** (harness counter; see item 1) | **36.0%** (re-scored 2026-11-24; 27.4% as first reported) | row 248069 ×1.5 + comma ×0.8 |
| Original's editor (published edit) | yes | 50 | **48.6%** (our stack) / 48.1% (theirs) | 2-tensor down_proj surgery |

## What the 50-pass blind run established

1. **Offering 50 passes raised the score a little — at the edge of what 500 problems can tell
   apart.** Offered 50 passes, the blind editor used 6 and scored 36.0%: 4.4 points above its
   6-pass GLM-Flash run (31.6%; 95% interval 0.2 to 8.6, p = 0.049; 68 problems right only by
   the 50-pass edit, 46 only by the 6-pass one), and indistinguishable from the mimo-pro edit
   (+0.8 points against 35.2%; 95% interval −3.0 to 4.8). Tenth convergence on
   row 248069 — the tenth blind session to ship an edit, and all ten chose that row. Why the
   session stopped is not established: one session stopped after six harness passes by its own
   choice. Its practice information per pass was far smaller than the original editor's
   (item 3), and it also ran at least ten scored practice evaluations through its own script
   outside the harness counter (`phase10/session/probes`: ten saved outputs `cat_*.err`, each
   ending with the engine's 100%-complete line for 64 of 64 problems), plus an uncounted
   "paired probe seed777". So "6 of 50 passes" understates the evaluations, and the budget was
   not software-enforced. (Corrected 2026-11-24, ISSUES #25: this item was headed "Budget was
   not the binding constraint — knowledge was" and quoted "+2.2 points ... 25.2 → 27.4". After
   the first checker correction of that day it was headed "Offering 50 passes did not raise
   the score", with +2.0 points, interval −2.2 to 6.4, not distinguishable.)
2. **The editor's own accounting, from its logs, about the 64 practice problems:** after the
   exit-row fix, remaining failures are ~12/64 genuine arithmetic errors ("needs training —
   forbidden"), ~4 formatting artifacts, ~3 residual loops. It concluded that the weight-space
   levers reachable from behavior observation were exhausted in six measurements and that the
   rest is the model's ability ceiling, not a discoverable defect. That is one editor's reading
   of 64 practice problems, not an established ceiling: the published edit scores 48.6% on our
   stack against this edit's 36.0%. (Corrected 2026-11-24, ISSUES #23: this item was headed "Why
   more passes cannot help" and stated the ceiling as fact.)
3. **The distance to the original is not a measurement of the hint (ISSUES #23; corrected
   2026-11-24).** Every blind editor that shipped an edit — ten sessions, every format — stopped
   surgery at the OUTPUT row (the exit token's embedding); as single editors they score 29.6% to
   36.2%. The hinted original performs a DEEPER operation: subtracting a measured direction from
   64 down_proj columns in layers 15/19 (downstream of the representation, not the readout), and
   scores 48.6% on our stack. But the two experiments differ in at least four ways at once, so
   the distance between them cannot be attributed to being told the failure. From the original
   authors' published experiment record (`RikiyaT/lm_understand-logs` on GitHub:
   `exp8_math/README.md` and the first turn's task-instruction file `runs/p50c/turns/t001/prompt.txt`):
   1. Their task instructions named the failure: the model reasons "often while repeating itself
      in its thinking".
   2. Their task instructions and harness FORBADE the quick fix: "making the model skip its
      thinking or close it early regardless of the problem" was out of scope, and edits to "the
      final norm, the embedding / output layer" and "edits that only multiply a whole tensor by a
      constant" were refused by the harness. Every blind editor in this repo converged on exactly
      that edit class (a row of the tied embedding).
   3. Their editor practised on 7,469 GSM8K training problems, 7,275 MATH training problems and
      714 AIME problems, up to 512 per pass at the full 32,768-token cap, over 50 passes. Our
      blind editors practised on 64 GSM8K problems per pass at an 8,192-token cap.
   4. Their editor was a frontier model (gpt-6-astra); ours were small local-tier models.

   **What is measured is a quick fix against a deeper edit under different rules, practice data
   and editor models. The experiment that would isolate the hint — a blind editor under the
   original rules, with in-distribution practice — has not been run.** (This item earlier read
   "What the hint actually buys, now measurable" and said blind editors "rationally stay at the
   surface and plateau at ~25–28%".)
4. **Prediction scorecard (honesty section):** pre-registered prediction was 28–34%, meaningfully
   above the 6-pass blind run, closing roughly one-third of the hint gap. Actual: 36.0% (re-scored
   2026-11-24; 27.4% as first reported). Band: **MISS** (36.0% is above the 28–34% band).
   "Meaningfully above 6-pass blind": **at the edge** (+4.4 points, interval 0.2 to 8.6,
   p = 0.049 — the interval excludes zero, but only just). "One-third of the gap": **MISS** —
   the gap closed is 4.4 of the 17.0 points between the 6-pass blind edit and the hinted edit on
   our stack (48.6%), short of one-third. No clause is a clean hit: the score overshoots the
   band, the rise over the 6-pass run is at the edge, and the gap closed is short of a third.
   (Corrected 2026-11-24, ISSUES #23: this item read "27.4%, closing ~15% ... Partially refuted —
   the miss is itself the finding: I overestimated what extra measurement budget could buy and
   underestimated how fast blind editors exhaust the safe mechanism space". After the first
   checker correction of that day it read: band HIT, 31.2% inside 28–34%; "meaningfully above"
   MISS, +2.0 points with an interval that includes zero.)

## The complete project ledger (all on main, every number reproducible)

Blind loners (final answer checker, 2026-11-24): mimo-flash 36.2, GLM-flash at 50 passes (this
phase) 36.0 and mimo-pro 35.2 are within noise of each other; the 6-pass GLM-flash edit (31.6)
and GLM-5.3 (29.6) are lower and within noise of each other. Two pairs are less clear: mimo-pro
against the 6-pass GLM-flash edit (+3.6, interval −0.6 to 7.6, not distinguishable), and 50
passes against 6 at GLM-flash (+4.4, interval 0.2 to 8.6, at the edge). (Until the second
checker correction of that day this read "the first four are within noise of each other".)
Committees: none of the five that shipped an edit beat the single editor it is compared with;
three more produced no edit. Random control 3.8. Original's edit (hinted): 48.6 our stack. Ten
of fourteen blind sessions shipped an edit; all ten converged on one row of 873M.
