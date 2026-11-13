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

---

# Site versus direction (addendum 3)

**Predictions:** `phase7/protocol.md` addendum 3, committed `8dc3a17` before any pass ·
**Budget:** 2 of 2 passes, no retries · same harness, pass_index 1. Both models were checked
against the arms they pair with: identical columns, amplitudes and Frobenius norm; only the
direction differs (`origcols_d` vs `origcols_rand_s1`; `u_on_loop` vs `brake_d100`).

| pass | arm | closed | accuracy | reading |
|---|---|---|---|---|
| 1 | `origcols_d` — published columns + amplitudes (L15/L19), our mapped direction d | 40.6% | 26.6% | works (1.8 above the line) |
| 2 | `u_on_loop` — L10+L14 loop neurons + brake amplitudes, published u15 / u19 | 48.4% | 32.8% | works |

The full site × direction table, closed % (stock 18.8):

| direction | L10+L14 loop neurons | L15+L19 published columns |
|---|---|---|
| published u | **48.4** (new) | 57.8 |
| our mapped d | 23.4 (also 20.3 / 18.8 at other doses) | **40.6** (new) |
| random | 51.6, 43.8, 39.1, 17.2 | 25.0, 29.7 |

## Verdicts

- **S1 confirmed, narrowly.** d on the published columns works by the rule (40.6 ≥ 38.8), but
  it is 17 points below u on the same columns (57.8) — about 2 SE of a difference. d is a
  weaker stand-in for u there (cos 0.71 / 0.74), not an equal.
- **S2: my guess was WRONG.** u works on the L10+L14 loop neurons (48.4%), where d on the same
  neurons at the same size does nothing. cos(u, d) there is 0.48 (L10) / 0.51 (L14).
- By the reading fixed in advance ("both work"): u-like directions work at both sites; d fails
  at L10+L14 for a reason specific to d there.

## What now stands

1. The published direction u is PORTABLE: it works on its own columns and on the loop
   neurons our map picked four to five floors earlier. u is the only direction that works
   at both sites.
2. Our mapped direction d is not the lever. It works only where it is mostly u (L15/L19),
   and less well than u.
3. The loop neurons from the map are a valid place to act: with u they give 48.4%, within
   noise of the published edit's 57.8% (difference SE ≈ 8).
4. Still unreconciled: random directions work at L10+L14 (3 of 4) and not at L15+L19
   (0 of 2), and they are unrelated to u (|cos| ≤ 0.04). So u is not the ONLY thing that
   works at L10+L14.

## Free check — the "size of the kick" account is dead (saved activations, no generation)

Computed after the predictions were committed. In-loop push divided by the clean-thinking SD
of the residual along the push direction:

| arm | L10 / L15 | L14 / L19 | outcome |
|---|---|---|---|
| d on loop neurons | 10.8 SD | 8.5 SD | nothing |
| u on loop neurons | 9.6 SD | 8.1 SD | works |
| random s1 on loop neurons | 30.6 SD | 27.2 SD | works |
| random s2 on loop neurons | 29.4 SD | 26.8 SD | nothing |
| published u on published columns | 5.6 SD | 4.7 SD | works |
| d on published columns | 5.7 SD | 4.5 SD | works (weakly) |
| random s1 on published columns | 22.1 SD | 19.2 SD | nothing |

Size relative to natural variation does not sort working from failing arms (5 SD works,
29 SD fails). R3 (one-third size fails) still holds as "needs enough size along a direction
that works", not as the explanation of which directions work.

## Not run

- What separates u from d: the part of u that is not along d, applied alone.
- More random draws at both sites (hit rate), and random directions at L15+L19 on the
  map's loop neurons of those floors rather than the published columns.
- Anything at MATH-500 scale.

---

# What u has that d lacks (addendum 4)

**Prediction:** `phase7/protocol.md` addendum 4, committed `898dadd` before the pass ·
**Budget:** 1 of 1 pass, no retry · same harness, pass_index 1. Model checked against
`u_on_loop`: identical columns and amplitudes; direction = u with its component along d
removed, not renormalised (Frobenius norm 0.88 / 0.86 of `u_on_loop`, cos with d = 0).

| arm (all on the L10+L14 loop neurons, same amplitudes) | closed | accuracy | cap-hit |
|---|---|---|---|
| d — the mapped loop direction | 23.4% | 20.3% | 76.6% |
| u — published direction (≈ half d, half the rest) | 48.4% | 32.8% | 57.8% |
| **u with its d-part removed** (`uperp_on_loop`, new) | **62.5%** | **43.8%** | 48.4% |
| random directions ⟂ d (four draws) | 51.6, 43.8, 39.1, 17.2 | 32.8, 29.7, 26.6, 12.5 | |
| reference: published edit at its own site (L15+L19) | 57.8% | 43.8% | |
| reference: stock | 18.8% | 12.5% | 82.8% |

## Verdict

- **T1 confirmed: works.** u does not need its d-part on these neurons.
- The caveat fixed in advance applies: most pushes ⟂ d work here, so "works" alone does not
  show u is special at this site.
- Beyond the pre-registered rule [suggestive, one draw each]: this is the highest closed rate
  of any arm in phase 7, level with the published edit (62.5 / 43.8 vs 57.8 / 43.8), from
  neurons chosen by our map at L10+L14. Removing the d-part RAISED closed by 14 points over
  full u (≈ 1.7 SE of a difference) and it is above all four random draws. That pattern —
  d: nothing, u: 48, u minus d: 62 — suggests the d-part is at best dead weight; it is not
  established at n = 64.

## What now stands (phase 7 causal, all batches)

1. WHERE: the loop-active neurons the map finds are a valid site (random neurons: nothing).
2. WHICH WAY: the mapped loop direction d is not the lever at L10+L14, and subtracting it may
   even work against the repair. What works there is off the d axis — the non-d part of u
   best, most random directions too.
3. At L15+L19 on the published columns the picture differs: u works, d works weakly, random
   directions do not. Not reconciled with point 2.
4. All of it is practice-set evidence, one run per arm.

## Not run

- A repeat of `uperp_on_loop` on a new sampling seed, and a MATH-500 final for it — the
  natural next step if this is to become a claim (it is a candidate repair found from the
  map's neurons plus the published direction, not from the map alone).
- The mirror arm at L15+L19 (the part of d that is not along u, on the published columns).
- More random draws at both sites.
