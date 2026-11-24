# Phase 7 causal spot-checks — results

**Date:** 2026-11-13 · **Predictions:** `phase7/protocol.md` (deliverable-3 addendum), committed
`97af18a` before any pass · **Harness:** `phase5/eval_pass5.py` unchanged, 64 GSM8K practice
problems, cap 8192, pass_index 1 · **Budget:** 6 of 6 passes, no retries, no other generation.
Raw: `passes.jsonl`, `ledger.txt`. Edits: `brake_*.json` (built by `build_brake.py`).

Practice numbers are as recorded with the old checker; the generations were not saved, so
they cannot be re-scored. Every 500-problem final accuracy in this file was re-scored on
2026-11-24 with the final answer checker (ISSUES #19; the checker was corrected twice that
day), and the `stack_full` final was voided (ISSUES #20); rerun clean on 2026-11-24: 46.2%.

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
- **H4 (exploratory): not met** (corrected 2026-11-24; first reported as "uninformative").
  Same closed rate as pass 2, accuracy 2 problems lower (11 of 64 against 13).
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
- **R4 confirmed (gating), as far as this control goes.** The same amplitudes on random
  neurons do nothing. (Corrected 2026-11-24: that arm's push was 2.6% and 3.8% the size of
  the loop-neuron arms' pushes, so it shows that an ungated push this small does nothing,
  not that the loop-active neurons are needed; ATLAS.md, "Corrections and limits".)
- **R5: my guess was WRONG.** Neither random direction on the published columns works
  (25.0 / 29.7), while the published direction on the same columns gives 57.8.
  By the strict rule "both do nothing" is not met either (29.7 sits 0.9 above the line),
  so "the published direction matters" is supported but not formally confirmed.

## What now stands

1. Loop-gated pushes need enough size (R3, R4); whether they need the right neurons is
   not tested by R4's control (corrected 2026-11-24).
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

---

# Confirming `uperp_on_loop` (addendum 5)

**Predictions:** `phase7/protocol.md` addendum 5, committed `8a0296b` before any run · both
runs made once, in order, no retries. Final: `phase3c/run_vllm.py` and `phase3c/rescore.py`
unchanged (500 MATH-500 problems, 1 sample, cap 32768, seed 20261105). Raw:
`phase3c/p7_uperp_on_loop*.json(.gz)`; the rescorer reads the gzipped texts directly
(`uv run python phase3c/rescore.py p7_uperp_on_loop`, no unzip needed).

## Run 1 — practice pass on a new sampling seed (pass_index 2)

| run | closed | accuracy | cap-hit |
|---|---|---|---|
| first (pass_index 1) | 62.5% | 43.8% | 48.4% |
| repeat (pass_index 2) | **76.6%** | **40.6%** | 35.9% |

**U1 confirmed: works.** The practice result is not sampling luck.

## Run 2 — MATH-500 final (re-scored 2026-11-24 with the corrected answer checker, ISSUES #19; n = 500)

| model | correct | accuracy | thinking closed | cap-hit |
|---|---|---|---|---|
| stock | 42 | 8.4% | 9.8% | 90.6% |
| random-edit control | 19 | 3.8% | 4.2% | 95.8% |
| GLM-Flash (blind LLM editor) | 158 | 31.6% | 99.0% | 24.0% |
| mimo-pro (blind LLM editor) | 176 | 35.2% | 97.6% | 42.0% |
| **`uperp_on_loop`** | **163** | **32.6%** | 75.6% | 42.8% |
| published edit | 243 | 48.6% | 83.6% | 26.4% |

**U2 confirmed: SUCCESS, below the published edit** (the 16.0–33.1 band fixed in advance; my
point guess 20–35% held). Stock is 8.4%.

Paired against the published edit on the same 500 problems: only published edit 100, only
`uperp_on_loop` 20 → −16.0 points (95% interval −20.2 to −12.0; SE 2.2). It is clearly worse
than the published edit. The two blind LLM editors score 31.6% and 35.2%; this file has no
paired comparison of this arm against them.

## What this says

1. A weight edit on neurons chosen by OUR activation map (L10+L14, four to five floors
   below the published site) repairs the model to 32.6% on the real benchmark. The map's
   neurons are a working site, on the test that counts.
2. The practice set OVERSTATED it. On 64 GSM8K problems this arm looked level with the
   published edit (62.5–76.6% closed vs 57.8%); on MATH-500 it is 16 points behind. Practice
   passes can rank "works / does nothing"; they cannot rank two working edits.
3. Caveats fixed in advance still apply: the direction comes from the published edit (this
   is not a from-the-map-alone repair), and the neurons were selected from stock's activity
   on these same MATH-500 problems (no answers used).

## Not run

- A final for full u on the same neurons (`u_on_loop`) or for a random direction that worked
  (`brake_rand`): without them the final cannot say whether removing the d-part helped at
  MATH-500 scale, or whether u is better than a random direction there.
- The published edit and this one together (both sites at once).

# Hub units (addendum 6)

Predictions committed first (`6cabd32`). Stock model. Parts A and B are forward passes
(`phase7/test_hubs.py` → `phase7/hubs.json`): the hub is held at one constant at every position
and every unit's d is recomputed; the baselines from the same script reproduce the light map
(Hard question 289 lights, Counting & Probability 14,775 vs 14,777, loop 3,682). Part C is one
practice pass.

Correction to the addendum's wording, found after it was committed and before any run: L19
unit 11 does not "get quieter" in loops. Its activation is negative and goes from −0.30 before
the loop to −2.97 inside it, i.e. it pushes about ten times harder, in the negative direction.
The predictions were left as committed.

## A and B — hold the hub steady, re-measure the lights above it

"Wire targets" are the units the hub's shipped wires point at; the wire prediction for each is
Δd = −(its contribution). "Lights above" counts units at |d| ≥ 0.8 on the floors above the hub.

| hub | job | targets that moved the predicted way | median measured / predicted | r | lights above: before → after (went dark / newly lit) | d before vs after, r | mean abs Δd: hub vs control |
|---|---|---|---|---|---|---|---|
| L14 u46 | Hard question | 255 / 460 (55%) | 0.03 | −0.02 | 137 → 125 (34 / 22) | 0.984 | 0.036 vs 0.002 |
| L14 u46 | Number Theory | 132 / 265 (50%) | 0.00 | 0.03 | 3,766 → 3,749 (293 / 276) | 0.995 | 0.038 vs 0.004 |
| L14 u46 | Geometry | 116 / 239 (49%) | −0.01 | −0.19 | 6,856 → 6,790 (401 / 335) | 0.996 | 0.040 vs 0.004 |
| L14 u46 | Precalculus | 123 / 203 (61%) | 0.06 | 0.17 | 4,803 → 4,832 (296 / 325) | 0.995 | 0.038 vs 0.004 |
| L5 u1087 | Counting & Probability | 454 / 532 (85%) | 0.19 | 0.65 | 12,097 → 11,495 (1,074 / 472) | 0.993 | 0.043 vs 0.005 |
| L19 u11 | Stuck in a loop (replay) | 160 / 173 (92%) | 0.68 | 0.84 | 496 → 420 (149 / 73) | 0.955 | 0.081 vs 0.005 |

- **H1** (≥ 65% the predicted way, median ratio 0.1–0.8): **holds** for L5 u1087 and L19 u11;
  **fails** for L14 u46 on all four jobs — its wires predict its targets no better than a coin.
- **H2** (pattern r ≥ 0.95 and lights changing by < 10%): **holds** for L14 u46 (Hard question
  −8.8%, the others within ±1%) and L5 u1087 (−5.0%); **fails** for L19 u11 (−15.3%; r 0.955).
- **H3** (hub changes the pattern more than its control): holds everywhere, 9–17×. For L14 u46
  the control is NOT matched — the hub's activation SD is 8.2 and the nearest non-source unit on
  the floor has 0.56 — so H3 says nothing there. L5 (0.152 vs 0.149) and L19 (0.974 vs 0.977)
  are matched.

## C — switch the loop hub off and generate

`hub19_zero`: down_proj column 11 of L19 set to zero (verified on disk: that column all zero,
no other value in the model changed). Zero is close to the unit's before-the-loop level (−0.30,
SD ≈ 1), so this is the generation-time counterpart of part B.

| arm | closed | accuracy | cap-hit |
|---|---|---|---|
| stock | 18.8% | 12.5% | 82.8% |
| `hub19_zero` | **26.6%** (17 / 64) | 17.2% | 73.4% |

**H4 confirmed: does not repair** (≤ 28.8%; inside the pre-set "no detectable effect" band).
+7.8 points is about one SE of a difference at n = 64. No further passes were due.

## What this says

1. **L19 unit 11 is a real, partial switch for the loop lights above it, and not a lever on
   looping.** Holding it at its before-the-loop level in replay turns off 149 of the 496 loop
   lights on floors 20–23 (30%) and its wires describe its targets well (92%, 0.68×, r 0.84).
   Switching it off during generation leaves the loop rate where it was. This is Finding 5's
   pattern again, one unit down: what marks a loop is not what causes it. It touches none of
   the loop lights at or below floor 19, which is most of them (3,186 of 3,682).
2. **L14 unit 46 was a hub by size, not by wiring — and its light is a question-length artefact.**
   Checked after the run: |activation| ≈ 75 on the first token of every benchmark question and
   ≈ 0.11 on every other token (SD 8.2; next-loudest unit on the floor 0.56, floor median 0.06).
   Its
   per-question average is ≈ 75 / question length, so it "lights up" for any job that goes with
   question length. Multiplied into the wire formula, that size makes every one of its wires
   look strong, and none predicts anything. Clamping it does move the lights above it more
   than clamping an ordinary unit, but there is no matched control for a unit that loud.
   Risk carried back to the light map: the reading jobs average over every position including
   the first, so other first-token units may be lit for the same reason. Not yet counted.
3. **L5 unit 1087's wires are real and weak**: right direction 85% of the time at a fifth of
   the predicted size; the Counting & Probability pattern survives without it (r 0.993).
4. For the wires view: a wire's sign is trustworthy for ordinary source units and its size is
   an overestimate; for an outlier-loud source it is neither. The page now says so per hub.

## Not run

- Clamping at generation time (needs hooks during sampling; the protocol allows weight edits
  only), so part C could only zero the unit.
- More than one hub per job, or any wire that is not from a hub.

# Pruning (addendum 8)

Predictions committed first (`d72262a`). "Pruning a unit" = setting its down_proj column to zero.

## What there is to prune

`scan_positions.py`, stock model reading the 500 benchmark questions (39,652 positions after
the opening):

- **Loud but empty** (|activation on token 0| >= 5 and >= 50x its usual size): **2 units of
  86,016** — L14 unit 46 (-75.0 on token 0, 0.13 elsewhere) and L3 unit 0 (-8.5, 0.16).
- **Silent: none.** The quietest unit has a per-position SD of 0.02; no unit has max
  |activation| below 0.05.

## Damage meter (forward passes, stock)

47 clean stock generations, first 1,500 generated tokens each, 67,497 tokens, teacher-forced.
Unpruned loss 0.3287.

| arm | units removed | loss | same next token as unpruned |
|---|---|---|---|
| the 2 spike units | 2 | 0.3295 | 98.91% |
| 2 ordinary units, same floors | 2 | 0.3288 | 99.52% |
| smallest 5% | 4,294 | 0.3771 | 94.31% |
| random 5% | 4,294 | 0.4541 | 90.74% |
| smallest 20% | 17,178 | 0.5910 | 86.62% |
| random 20% | 17,178 | 0.8857 | 78.89% |
| smallest 50% | 42,944 | 1.4999 | 69.15% |
| random 50% | 42,944 | 2.1285 | 58.27% |

## Generation (practice passes; base = the published-edit model)

| arm | closed | accuracy | cap-hit |
|---|---|---|---|
| published edit (reference) | 57.8% | 43.8% | 43.8% |
| `orig_prune_spike` (2 spike columns zeroed) | **64.1%** | 50.0% | 35.9% |
| `orig_prune_quiet` (smallest 5%, 4,294 columns zeroed) | **62.5%** | 34.4% | 42.2% |

The quiet arm used 5% because no dose kept >= 95% of next tokens (the pre-set rule). No control
pass was due. SE of a difference at n = 64 is about 8 points.

## Verdicts

- **S1** (spikes change >= 10x as many tokens as two ordinary units): **false** — 1.09% vs
  0.48% of positions, 2.3x.
- **S2** (the model still works without them): **confirmed** — 64.1% closed, 50.0% accuracy.
  Pre-set reading for S1 false + S2 true: the loud units are waste and can simply be removed.
- **Q1**: smallest-first beats random at every dose (**holds**); the smallest 5% keeps >= 97%
  of next tokens (**fails**, 94.3%).
- **Q2** (smallest 50% does heavy damage): **holds**, 69.2%.
- **Q3** (`orig_prune_quiet` still works): **confirmed**, 62.5% closed. Accuracy is 9 points
  under the reference, about one SE: not detectable at this size, not shown to be free either.

## What this says

1. The loud-but-empty units are real waste, and there are two of them. Removing them costs
   nothing measurable. That is 0.002% of the units: no efficiency to be had there.
2. There is no dead weight to sweep up. Every unit fires, and the simplest rule (remove the
   smallest) starts changing the model's next token immediately: 1 token in 17 at 5%, 1 in 7
   at 20%, 1 in 3 at 50%. A small model uses its units.
3. The model nonetheless still closes its thinking on practice problems with the smallest 5%
   gone. Token-level change and task-level failure are different things; the gap between them
   is where pruning could be informative.
4. Pruning is therefore more useful here as a TEST of a label than as a way to save space:
   remove the units carrying a label and see whether the matching behaviour, and only that
   one, gets worse.

## Not run

- Any pruning with a smarter rule than "smallest first" (the phase-8 design).
- A MATH-500 final for either pruned model; the practice set cannot rank working models.
- A knockout of a labelled group (e.g. the Geometry units) with a per-subject readout.

# Knockout test of the subject labels (addendum 9)

Predictions committed first (`b96374e`). Stock model, forward passes (`knockout_subjects.py` →
`knockout.json`). Even-index problems chose each subject's 1,000 units (1.2% of all units; the
ones most MORE active on that subject, question length regressed out); the test is the stock
model's own saved answers to the 250 odd-index problems, first 300 generated tokens each
(75,000 tokens). "Extra loss" = next-token loss with the units zeroed minus without.
Selectivity = extra loss on the knocked-out subject's text / mean extra loss on the other six.
Control = the same number of random units per floor. The seven sets barely overlap (at most 34
shared units). The 90% ranges resample the test problems and were added after the first look
(post hoc, not in the addendum).

| subject knocked out | extra loss on its own text | on the other subjects | selectivity (90% range) | control selectivity (90% range) | more selective than control |
|---|---|---|---|---|---|
| Number Theory | +0.028 | +0.019 | **1.53** (1.26–1.84) | 0.96 (0.83–1.09) | 100% of resamples |
| Geometry | +0.030 | +0.020 | **1.47** (1.26–1.71) | 1.00 (0.76–1.24) | 100% |
| Intermediate Algebra | +0.046 | +0.034 | **1.37** (1.19–1.55) | 1.02 (0.83–1.23) | 98% |
| Counting & Probability | +0.136 | +0.095 | 1.43 (1.29–1.59) | 1.34 (1.10–1.58) | 75% |
| Prealgebra | +0.034 | +0.033 | 1.04 (0.91–1.19) | 0.95 (0.82–1.08) | 79% |
| Precalculus | +0.023 | +0.023 | 0.99 (0.80–1.20) | 0.66 (0.51–0.83) | 98% |
| Algebra | +0.021 | +0.022 | 0.96 (0.83–1.11) | 0.96 (0.80–1.11) | 55% |

Unpruned loss is 0.33–0.50 per subject, so a knockout of 1,000 units costs 5–10% everywhere
(Counting & Probability's set: about 25%).

- **K1** (Geometry: selectivity > 1, and more than its control, control within 0.8–1.25):
  **holds** — 1.47 vs 1.00.
- **K2** (the same for at least 5 of 7 subjects): **fails** — clear for 3 (Number Theory,
  Geometry, Intermediate Algebra). Counting & Probability's text is hurt most by ANY removal
  (its control is 1.34), so its 1.43 is not about the label; Prealgebra, Precalculus and
  Algebra show no selectivity.
- **K3** (own knockout is the most damaging of the seven on its own text, for at least 4):
  **fails** — 1 of 7, and that one (Counting & Probability) only because its set is the most
  damaging on every subject.
- **K4** (median selectivity 1.1–2): **holds** — 1.37.

## What this says

1. For three subjects the label survives a knockout on held-out problems: remove the units
   tagged Geometry, Number Theory or Intermediate Algebra and that subject's text is hurt
   about 1.4–1.5 times as much as the others. Those three tags can read "used for", weakly.
2. The other four stay "goes with". Their units light up for the subject but removing them
   hurts every subject equally.
3. Even where it holds, it is a tilt, not a module: knocking out Geometry's top 1,000 units
   raises loss on geometry text by 7% and on everything else by 5%. Most of what these units
   do is shared across maths.
4. Limit stated in advance: a unit that is more active on a subject removes more when zeroed
   there, so some selectivity is built in. Three subjects showing it and four not is therefore
   informative in both directions.

## Not run

- A generation check (per-subject accuracy with a subject knocked out): 38–124 problems per
  subject on MATH-500 gives 5–8 points of noise, too coarse for a 1.4x tilt.
- Other set sizes than 1,000, or units chosen by anything other than the reading contrast.

# Stacking (addendum 10): published edit + the map's L10+L14 repair

Predictions committed first (`d146c4c`). `stack_full` = published edit plus the
`uperp_on_loop` change at dose 1.0; `stack_half` = the same at dose 0.5. Built with
`build_brake.py --base phase3/models/c3_original_edit`; both differ from the published edit
only at L10/L14 `down_proj`.

Corrected 2026-11-24: the first `stack_full` final is VOID (a corrupted run, ISSUES #20); the
same model was rerun clean the same day and scores 46.2% (last row of the finals table). The
final accuracies below are re-scored with the final answer checker (ISSUES #19; the checker
was corrected twice that day).

## Step 0 — is there anything left to fix?

`count_loops.py p3c_c3_original`: 132 of 500 answers hit the token cap; 125 of those (94.7%)
are loops by the phase-7 detector, onset median 13,250 tokens. Stop rule (<25%) not triggered.

## Practice passes (64 GSM8K, cap 8192) — gate: closed ≥ 38.8%

| arm | closed | accuracy | cap-hit |
|---|---|---|---|
| stack_full | 62.5% | 40.6% | 42.2% |
| stack_half | 56.2% | 39.1% | 46.9% |
| published edit (earlier) | 57.8% | 43.8% | — |

Both pass.

## MATH-500 finals (re-scored 2026-11-24; n = 500, paired with the published edit)

| model | accuracy | answered | cap-hit | loops among cap-hits | only stacked | only published | difference | SE | verdict |
|---|---|---|---|---|---|---|---|---|---|
| published edit | 48.6% | 83.2% | 26.4% | 125 / 132 | | | | | |
| stack_half | 44.8% | 82.8% | 26.6% | 121 / 133 | 39 | 58 | −3.8 (95% interval −7.6 to 0.2) | 2.0 | no difference |
| stack_full, first run | VOID — a corrupted run (ISSUES #20) | — | — | — | — | — | — | — | void |
| stack_full, clean rerun (2026-11-24) | 46.2% | 84.4% | 14.0% | — | 30 | 42 | −2.4 (95% interval −5.8 to 0.8) | 1.7 | no difference |

**The first `stack_full` final is VOID — a corrupted run, not a measurement of the model
(ISSUES #20, 2026-11-24).** Its score was 4.4% as first recorded and is 6.4% after re-scoring;
neither is a result. Evidence from the saved generations: 460 of the 500 (92.0%) break into
fragments of several scripts, and whether a generation was corrupted depends on the request's
position in the batch, which a property of the model cannot know. Counted with the first
corrected checker of that day (when the run read 5.0% in all), the first 287 requests (the
first wave the engine admitted) were 8.7% correct (25 of 287) and the last 213 were 0.0% (0 of
213; exact test, p = 7.6e-7); among the other 25 saved 500-problem runs the same split never
reaches p = 0.6. Under the final checker the dependence is still there: all 32 right answers
fall among the first 266 requests, and the validity check recorded in the result file
(`phase3c/runcheck.py`; `phase3c/p7_stack_full_rescored.json`) reads 12.0% correct in the
first half of the rows against 0.8% in the second half (Fisher exact test, p = 1.1e-7). The
same model's 64-problem practice pass the same day scored 40.6%. The engine log showed
nothing. The cause is not isolated (an engine-and-model interaction under late
admissions is the standing suspicion; ISSUES #13 records the same signature in a voided
phase-5 run). The first run stays void with this evidence unchanged; its clean rerun of
2026-11-24 scores 46.2%, so the "collapse at full dose" recorded on 2026-11-14 was the
corrupted run, not the model.

**Rerun 2026-11-24: the model is not broken — 46.2%.** The voided run's model (published edit
plus the map's L10+L14 repair at dose 1.0; files rebuilt and verified equal to the intended
change) was rerun under the current runner as `p7_stack_full_v2` (`phase3c/p7_stack_full_v2.json`,
`_texts.json.gz`, `_rescored.json`; log `phase3c/logs/p7_stack_full_v2.log`; 2 hours 08 minutes
of GPU). It scores 46.2% (231 of 500); an answer was read in 84.4%; 70 of 500 hit the cap
(14.0%); 426 of 500 closed their thinking; mean 18,146 tokens. The validity checks are clean:
fragment tails 0.0%; first half of the requests 46.8% right against 45.6% for the second half
(p = 0.86); closed 84.4% against 86.0% (p = 0.71); finished 85.6% against 86.4% (p = 0.90).
Paired with the published edit (48.6%): −2.4 points (95% interval −5.8 to 0.8; 30 problems
right only under the stacked model, 42 only under the published edit, 201 under both; p = 0.195;
SE 1.7) — no real difference, by the registered rule. Against the half-dose stack (44.8%):
+1.4 (interval −2.6 to 5.4; 55 only full, 48 only half; p = 0.56). Against the map repair
alone (32.6%): +13.6 (interval 9.6 to 17.6; 90 against 22). Against stock (8.4%): +37.8
(interval 33.4 to 42.4; 194 against 5).

The readings recorded on 2026-11-14 describe that corrupted run, not the model, and are
withdrawn: "`stack_full` does not loop; it falls apart"; 453 of its 500 answers ending in word
fragments ("ylim", "alem", "illis", "ython" … at least 10 in the last 3,000 characters; 0 of
500 for `stack_half`); answers that start as normal reasoning and degrade partway through;
and the pairing "19 both right, 3 only stacked, 178 only published, −35.0 points, SE 2.7,
LOSS". As recorded then, the engine log was clean (exit 0, no retry used).

## Verdicts

- **V0** (≥ 60% of the published edit's cap-hits are loops): **holds** — 94.7%.
- **V1** (both pass the practice gate): **holds**.
- **V2** (`stack_full` not a WIN, accuracy 33–42%): **not a WIN — holds** (−2.4 points, SE
  1.7: no real difference); the range **does not hold** (46.2% is above 33–42%). (Recorded on
  2026-11-14 as "first half holds, range fails — a collapse nobody predicted", on the
  corrupted run; that reading was withdrawn 2026-11-24, ISSUES #20, and the prediction is
  scored on the clean rerun.)
- **V3** (`stack_half` 38–44%, at or above `stack_full`): **does not hold on both counts.**
  The range: 44.8% is above the band's upper end of 44% (37.0% as first reported, one point
  under the band; 40.8%, inside the band, after the first checker correction of that day).
  "At or above `stack_full`": 44.8% is below the rerun's 46.2%, though the two doses are not
  distinguishable (+1.4, interval −2.6 to 5.4).
- **V4** (about 1 in 4 that either is a WIN): **neither model is a WIN** (−3.8, SE 2.0, at
  half dose; −2.4, SE 1.7, at full dose: no real difference either way).

## What this says

1. The map's repair does not add to the published edit at half strength. It changes which
   problems are solved (39 gained, 58 lost) but not how many (difference −3.8 points, 95%
   interval −7.6 to 0.2), and the loop count is untouched (121 vs 125).
2. Nor at full strength. The rerun scores 46.2% against the published edit's 48.6% (−2.4
   points, 95% interval −5.8 to 0.8) and against the half dose's 44.8% (+1.4, interval −2.6 to
   5.4). So the stacking conclusion holds for both doses: the map repair adds nothing on top of
   the published edit at either dose (−3.8 at half, −2.4 at full, neither distinguishable from
   zero) and costs nothing detectable. The "collapse at full dose" recorded on 2026-11-14 was
   the corrupted run, not the model.
3. Nothing here tests practice passes as a safety gate: both models passed the practice gate
   (`stack_full` 40.6% and `stack_half` 39.1% on 64 problems) and both clean finals scored
   46.2% and 44.8%, so this addendum has no case of a model that passes practice and then
   fails a valid final. (Until 2026-11-24 items 2 and 3 read the void final as the model
   breaking on long answers, unseen by short practice answers.)

## Not run

- Doses between 0.5 and 1.0, or below 0.5. Choosing a dose on these 500 problems would be
  tuning on the judging set.
- A second sampling seed for either final.

---

# Lever-ranked units as an edit site (addendum 11)

**Predictions:** `phase7/protocol.md` addendum 11, committed `93db98f` before the model was
built · **Budget:** 1 of 1 pass, one try, no retry · same harness, pass_index 1. (A first launch
died before the script ran — Windows line endings in `run_passes.sh`; nothing was generated and
the ledger has no entry for it. The script was then run with the carriage returns stripped.)

`u_on_lever` = `u_on_loop` with the 64 units per floor chosen by |lever score|
(`build_brake.py --u-dirs 15,19 --lever-cols`) instead of by loop light. Checked against
`u_on_loop` from the builders' own records: same floors, direction and amplitude rule; push in
loops minus before 1.272 / 2.217 in both; Frobenius norm 1.082 / 1.244 against 1.097 / 1.225;
24 of 64 units shared at L10, 27 of 64 at L14.

| arm (L10+L14, published u15 / u19, same push size) | closed | accuracy | cap-hit | mean tokens |
|---|---|---|---|---|
| `u_on_loop` — top 64 by loop light (earlier) | 48.4% (31 / 64) | 32.8% (21) | 57.8% | 6928 |
| **`u_on_lever` — top 64 by lever score (new)** | **50.0% (32 / 64)** | **26.6% (17)** | 60.9% | 6993 |
| reference, different direction: `uperp_on_loop` | 62.5% (40 / 64) | 43.8% (28) | 48.4% | 6829 |
| reference: stock | 18.8% | 12.5% | 82.8% | |

## Verdicts

- **W1** (works, closed ≥ 38.8%): **holds** — 50.0%.
- **W2** (no detectable difference from `u_on_loop`, closed strictly between 32.4% and 64.4%):
  **holds** — +1.6 points, one problem out of 64. Not BETTER (needed 42 of 64, got 32).
- **W3** (closed 40–58%, accuracy 25–41%): **holds** on both, accuracy near the bottom of the
  range (26.6%).
- **W4** (BETTER about 1 in 8, nothing about 1 in 6): neither happened.

## What this says

1. At L10+L14, picking units by lever score instead of loop light changes 40 and 37 of the 64
   units per floor and does not change the closed rate (50.0 vs 48.4). On this test the lever
   ranking is not a better edit site.
2. Accuracy is 6 points lower (17 vs 21 problems right), under one SE of a difference (≈ 8
   points). Not evidence of a worse site either.
3. It is 12.5 points under `uperp_on_loop` (≈ 1.6 SE), which uses the loop-light units with a
   different direction. Nothing here says the lever units would do better with that direction.
4. The caveats fixed in advance apply: practice does not rank two working edits, the lists
   overlap by about 40%, and this was run at floors where the lever layer shows no net push.
   The lever layer stays a description of where the published edit's units sit, not a
   selection rule that has been shown to help.

## Not run

- A second sampling seed for `u_on_lever`, or any MATH-500 final for it or for `u_on_loop`.
- The lever units with the d-part of u removed (the `uperp_on_loop` direction).
- Lever-ranked units at L15+L19, where the ranking mostly re-finds the published units
  (46 and 32 of 64).
