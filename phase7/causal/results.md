# Phase 7 causal spot-checks — results

**Date:** 2026-11-13 · **Predictions:** `phase7/protocol.md` (deliverable-3 addendum), committed
`97af18a` before any pass · **Harness:** `phase5/eval_pass5.py` unchanged, 64 GSM8K practice
problems, cap 8192, pass_index 1 · **Budget:** 6 of 6 passes, no retries, no other generation.
Raw: `passes.jsonl`, `ledger.txt`. Edits: `brake_*.json` (built by `build_brake.py`).

| pass | arm | closed | accuracy | cap-hit | mean tokens |
|---|---|---|---|---|---|
| 1 | stock | 18.8% | 12.5% | 82.8% | 7,253 |
| 2 | map brake along d, L10+L14, dose 1.0 | 23.4% | 20.3% | 76.6% | 7,198 |
| 3 | same neurons + amplitudes, RANDOM direction ⟂ d | **51.6%** | **32.8%** | 54.7% | 6,261 |
| 4 | map brake, dose 0.5 | 20.3% | 18.8% | 79.7% | 7,169 |
| 5 | map brake, dose 2.0 | 18.8% | 17.2% | 81.2% | 7,057 |
| 6 | selective map brake, dose 1.0 | 23.4% | 17.2% | 76.6% | 6,923 |

Reference on the same set and harness (phase 5): original edit 56.2% closed / 39.1% accuracy.
Stock reproduces phase 4's pass 1 exactly (12.5 / 18.8 / 82.8).

## Verdicts against the pre-registered rules

- **H1 REFUTED.** Brake along d: +4.6 points closed (needed ≥ +20; < +10 = refuted).
  Subtracting the mapped loop direction in the L10–L14 band does not stop loops.
- **H2: the "not about d" branch fired.** The random-direction control closed +32.8 points
  over stock (≈ 4 SE of a difference) and more than doubled accuracy.
- **H3 uninformative.** Doses 0.5 / 1.0 / 2.0 gave 20.3 / 23.4 / 18.8 — inside the rule, but
  all within noise of stock, so there is no dose-response to speak of.
- **H4 (exploratory) uninformative.** Same closed rate as pass 2, accuracy 2 problems lower.
  The plain map brake was already near-selective by construction (standing push ≤ 0.08).

## What the surprise is, and is not

Free follow-up checks (weights and saved activations only, no generation):

- The random directions are not secretly meaningful: |cos| with the original edit's u15 / u19
  ≤ 0.04; logit-lens z-score for `</think>` +0.4 (L10) and −0.1 (L14).
- The two arms are matched in norm but NOT in size relative to natural variation. Along d the
  residual already varies a lot (clean SD 0.13 at L10, 0.26 at L14), so the in-loop push is
  ≈ 9–11 SD. Along a random direction the residual barely moves (SD 0.04 / 0.08), so the same
  push is ≈ 27–30 SD — a large, out-of-distribution kick that fires only when the loop
  neurons fire.

Reading [inferred, single draw]: the map found WHEN to intervene (the loop-gated neurons), not
WHAT to write. A big enough loop-gated kick in an arbitrary direction breaks the loop; removing
the "this is repeating" direction does not. Not yet established: one random direction, one
seed, n = 64, practice set only.

## Consequences

- Atlas Finding 5 stands as a description (the direction marks loops) but its causal reading
  is weakened: the direction may be a readout of repetition, not the lever.
- Open question this raises about the ORIGINAL edit: no one has run "same 64 columns and
  amplitudes, random direction" at L15/L19. If that also works, the published direction is
  not what repairs the model — the loop-gated columns are.

## Not run (would need a new pre-registered budget)

1. Three more random-direction seeds at L10+L14 (is pass 3 a fluke of one draw?).
2. Original edit's columns and amplitudes with a random direction (two seeds).
3. Random neurons + random direction, norm-matched (does the gating matter?).
4. Variance-matched comparison: d-push and random-push at equal SD multiples.

---

# Replication batch (addendum 2)

**Predictions:** `phase7/protocol.md` addendum 2, committed `190cf5a` before any pass ·
**Budget:** 9 of 9 passes, no retries · same harness. "Works" = closed ≥ 38.8% (stock + 20);
"does nothing" = closed ≤ 28.8% (stock + 10). Stock: 18.8% closed / 12.5% accuracy.

| pass | arm | closed | accuracy | reading |
|---|---|---|---|---|
| 1 | `rand_s2` — L10+L14 loop neurons, new random direction | 17.2% | 12.5% | does nothing |
| 2 | `rand_s3` — same, another direction | 43.8% | 29.7% | works |
| 3 | `rand_s4` — same, another direction | 39.1% | 26.6% | works |
| 4 | `brake_rand` again, new sampling seed | 43.8% | 29.7% | works (first run 51.6%) |
| 5 | `rand_lo` — first random direction at one-third size | 28.1% | 23.4% | does nothing |
| 6 | `randneur` — same amplitudes on random non-loop neurons | 23.4% | 23.4% | does nothing |
| 7 | `orig` — the published edit | 57.8% | 43.8% | reference, as expected |
| 8 | `origcols_rand_s1` — published columns + amplitudes, random direction | 25.0% | 23.4% | does nothing |
| 9 | `origcols_rand_s2` — same, another direction | 29.7% | 28.1% | in between (not "works") |

## Verdicts

- **R1 REPLICATED** (2 of 3 new directions work). With the first draw: 3 of 4 random
  directions work at L10+L14 (51.6 / 43.8 / 39.1); one does nothing (17.2).
- **R2 confirmed.** The first result was not sampling luck (43.8% on a new seed).
- **R3 confirmed (kick size).** At one-third size the same direction does nothing.
- **R4 confirmed (gating).** The same amplitudes on random neurons do nothing — the
  loop-active neurons the map picked are needed.
- **R5: my guess was WRONG.** Neither random direction on the published columns works
  (25.0 / 29.7), while the published direction on the same columns gives 57.8.
  By the strict rule "both do nothing" is not met either (29.7 sits 0.9 above the line),
  so "the published direction matters" is supported but not formally confirmed.

## What now stands

1. Loop-gated pushes need the right neurons and enough size (R3, R4).
2. At the original edit's site the DIRECTION matters: published 57.8% vs random 25–30%.
3. At L10+L14 the mapped direction fails at three doses, yet 3 of 4 random directions work.
4. Points 2 and 3 are not reconciled. The difference between the two sites for random
   directions (3 of 4 vs 0 of 2) is far too few draws to call a site effect.

So the first-batch reading ("the map finds WHEN, not WHAT; direction is irrelevant") was too
strong. Direction is not irrelevant; the mapped direction is simply not the effective one
at L10+L14, and some arbitrary ones are.

## Not run

- The mapped direction d on the published columns (L15/L19), and the published u at L10+L14
  — the two arms that would separate "site" from "direction".
- More random directions at both sites (enough draws to estimate a hit rate).
- Anything at MATH-500 scale.
