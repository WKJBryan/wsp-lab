# Phase 12 results — the cross-architecture atlas

**Date:** 2026-11-17 · **Protocol:** `phase12/protocol.md` (`6c87547`, pre-registered
before any probe) · Models: Qwen3.5-0.8B (dense, 24L, hidden 1024) vs Gemma-4 26B-A4B
pruned wound (MoE 8 experts, 30L, hidden 2816). Free analysis only; teacher-forced
passes over saved generations; identical probe code both sides.

**Redone 2026-11-26 (saved data only):** the correspondence test was redone on the saved
data under a protocol committed before the redo (`phase12/protocol_redo.md`); the record is
`phase12/results_redo.md`, and it changes no number in this file. Of the question's three
parts, only writer geometry has a cross-model test that can count for or against
correspondence, and it counts against: in every layer of both models the loop direction is
written more focally than a random direction, and Qwen's about 1.35 times the wound's (95%
interval 1.19 to 1.53). The saved data cannot exclude two explanations: the wound's directions
were fitted on 5 texts against Qwen's 11, and nothing is known about how focally the wound
writes other state differences. Loop dynamics has Qwen-only evidence; termination has a
necessary-condition test, which passes. So the cross-architecture claim is **not established**,
and phase 13 is closed on that basis.

**Corrected 2026-11-24 (ISSUES #21):** the verdict first published here, "CORRESPONDENCE
(2 of 3 tests passed)", was withdrawn. The accuracies quoted for test C2 were re-scored the
same day with the answer checker of 2026-11-24 (ISSUES #19), and again on 2026-11-26 under
the final answer checker (ISSUES #30, #31).

## Verdict: **NOT ESTABLISHED** (the registered rule, applied literally, prints "2 of 3"; both passes are uninformative)

| Test | Question | Result | Pass |
|---|---|---|---|
| **C1** | Do loop-dynamics trajectories correspond? | **r = 0.96**; internal AUC 0.83 / 0.90. But the shape is fixed by construction, so the correlation is not evidence | prints a pass; uninformative |
| **C2** | Termination-circuit fingerprint match? | Behavioral pass, with the registered Qwen arm (the ×1.5 single-row edit): Qwen termination recovered 55.2% relative with accuracy **up** 32.8 points (9.4 → 42.2; up 27.8 points, 8.4 → 36.2, under the checker of 2026-11-24); Gemma 57.1%, accuracy indistinguishable (2 correct of 64 after the boost; 1 before it under the checker of 2026-11-24, none under the final answer checker). Statistical pass by luck: "low norm" cannot find the exit row | prints a pass; uninformative |
| **C3** | Writer geometry (focal vs diffuse) distributions? | Both **focal** (no layer under 3.0 in either; ranges 6.2–18.2 vs 6.8–20.2); KS = 0.55, p = 0.0003. Gemma is the *less* concentrated model | fails |

## What the numbers say, plainly

1. **C1 (r = 0.96): the same shape is built in.** The loop direction is defined as
   mean(loop states) minus mean(pre-onset states), and each trajectory is z-scored by the
   loop window's own mean and spread, so loop positions sit at 0 and pre-onset positions
   below by definition. The two 25-point curves are almost perfectly smooth (lag-1
   autocorrelation 0.982 and 0.975; effectively a quarter of an independent point each),
   and their first differences correlate at only +0.19. Two generic step curves of
   unrelated widths and positions correlate at 0.96 or more 7.8% of the time (37% when
   restricted to the range the two fits occupy), and at 0.5 or more 68% of the time. After
   each curve's own best-fitting step is removed, the two residuals correlate at −0.13 (16
   of the 24 possible circular shifts reach that: 0.667). The "1,000 circular shifts" test
   has only 24 distinct shifts, and the largest correlation among them is 0.859, so the
   "permutation p < 0.001 (0/1000 shifts)" of the first report is not a valid p-value. The
   two text sets are not comparable either: 23 Qwen stock texts (loop onset at a median
   4,489 tokens) against 10 of the Gemma wound's 24 texts, 14 of which were rejected
   because looping began under 100 tokens in (median onset 263 tokens); for 7 of the 10
   Gemma texts the window starts at position 0, so the first bins include the chat question
   itself. What is supported: in both models the state transition lags the token-level
   repetition onset by 40–80 tokens. (Post hoc, about 12 autocorrelated points, not a test:
   the small post-onset wobble agrees, demeaned r = +0.98 over bins 13–24.)
2. **C2: both exit rows are small, but that was already known, and "low norm" cannot find
   them.** Qwen's `</think>` (row 248069, the knob of phases 4–10) and Gemma's end_of_turn
   (row 106) sit at the 16.8th and 7.6th percentile of embedding-row norms, and boosting
   them recovers termination in both models: Qwen 55.2% relative (cap-hit 90.6% → 40.6%),
   Gemma 57.1% (98.4% → 42.2%), with accuracy indistinguishable there (1 correct of 64 before
   and 2 after under the checker of 2026-11-24, 0 and 2 under the final answer checker; a 95% interval for 1 of 64 is 0.3% to 8.3%). But a row boost cutting cap-hits
   in both models was already on record from phases 4–5 and 11; the test adds nothing new.
   Nor is low norm a way to find the exit row: Qwen's `</think>` row has a lower norm than
   only 41,742 of the 248,320 rows (16.8th percentile); 41,700 rows lie below it and none of the three
   termination tokens is among them; `<|im_end|>` sits at the 51st percentile and
   `<|endoftext|>` at the 99.96th. In Gemma, `<eos>` (row 1) sits at the 99.4th percentile.
   With three Gemma candidate rows, the rule "any candidate in the same norm tercile"
   passes 70% of the time by chance (only end_of_turn, row 106, matches; the other two
   candidates sit at the 99.4th and 97.8th percentile). Both anchors were chosen by name,
   before their norms were looked at: low norm is a description, not a discovery rule.
   (The first report called this "a portable, checkable signature"; withdrawn 2026-11-24.)
3. **C3: the distributions differ, in the direction opposite to the first report.** Both
   models are focal (a few columns or experts dominate the writing onto the loop direction
   within a layer — this *refines* phase 7's "diffuse" finding, which compared across
   layers, not within), and the ratio distributions differ (KS D = 0.55, p = 0.0003). But
   Gemma is *less* concentrated, not "significantly steeper": its ratio is higher in only
   145 of the 720 layer pairs. As run the test fails; with the Qwen input the protocol
   names (`phase7/atlas_geometry.json`) it passes (p = 0.40). The layers also have 3,584
   against 7,744 writer columns, and with random directions that count difference alone
   gives ratios of 4.7 against 5.0. So the degree difference the first report read into
   this test is not established either.

## Phase-13 setup: not armed; to be redesigned

Phase 13 as registered ("apply Qwen's diagnosis to the Gemma wound with zero search")
rested on this correspondence and has no premise now that the verdict is withdrawn
(ISSUES #21). It is not armed and is to be redesigned. (Closed 2026-11-26: the redo above left the correspondence not established, so phase 13 is closed on that basis.) (The first version of this section
said the candidate termination rows "can be *found* statistically (low-norm eos-family
rows)" and called phase 13 "unchanged, now armed"; both withdrawn with the verdict.)

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

Added 2026-11-24 (ISSUES #21):

- The first probe gave a held-out AUC of 0.68, below the 0.75 validity bar; the text
  selection and the onset detector were then changed (final 0.826). Pre-registration to
  verdict took 32 minutes; no intermediate result is recorded.
- The ROADMAP's original three tests (CKA similarity, rank of the true analog among all
  rows, adjusted Rand index) were replaced 8.4 hours before the run without a note.
- The calculator read `phase11/ed_pass1.json` and `ed_pass2.json`, which later editor
  attempts had overwritten; it now reads pinned copies under `phase11/editor_a2_attempt1/`,
  records its inputs in `phase12/correspondence.json` (`inputs`), and writes an error record
  if the pinned files are not attempt 1's passes. It had also read the original authors'
  two-tensor edit as the Qwen arm; it now reads the registered arm, the ×1.5 single-row
  edit.
