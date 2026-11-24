# The Matrix Atlas — Phase 7 working document

Living document; each entry carries evidence + a confidence label. Direction v = the loop
direction (unit top-singular vector of the L15 edit block; cos(v15,v19)=0.849). Corrections of
2026-11-24 are marked where they fall and listed under "Corrections and limits" at the end.

## Finding 1 — the loop direction is written DIFFUSELY, not locally [measured]

Every layer's MLP (down_proj) writes onto v with statistically similar strength:
mean |alignment| ≈ 0.006–0.008, max ≈ 0.05–0.09 across all 24 layers. No hotspot at the
edited layers (15/19 are unremarkable in this metric). The loop direction is a
network-wide residual-stream property, not the product of one layer.

Evidence: `atlas_geometry.json` → down_proj_alignment (full per-layer table).

## Finding 2 — the edited columns ARE the most weight-aligned ones [measured; reversed 2026-11-24, ISSUES #24]

The original authors' selection is recoverable from weight geometry alone, up to which of
the well-aligned columns they left out. Measured from the stock weights and the published
edit (`phase7/probe_out/edit.npz`); alignment = each column's dot product with the edit
direction u (the residual direction of the original edit: u15 at L15, u19 at L19), as stored
in `atlas_geometry.json`:

- **Layer 15:** all 5 of the 5 most weight-aligned stock columns are among the 64 edited
  columns; 31 of the 64 edited columns are in the top 64 by alignment (chance 1.1;
  probability of 31 or more by chance 1.4e-40); the median edited column ranks 69th of
  3,584 (98.1st percentile).
- **Layer 19:** 4 of the top 5; 18 of 64 in the top 64 (probability 5.2e-18); median 96.4th
  percentile.
- **Amplitude:** on every edited column the edit's amplitude is minus 6.00 times that
  column's own stock alignment with u (correlation −1.000000; opposite sign in 64 of 64
  columns at both layers). The edit multiplies each edited column's component along u by
  −5.00.
- **Not simply "the 64 most aligned":** the least-aligned edited column at layer 15 ranks
  1,113th, while more-aligned columns (322, 467, 844) are unedited.

Earlier reading, reversed on 2026-11-24: this finding said the edited columns were NOT
chosen by weight-side alignment, because it listed L15's top-aligned columns (1155, 1787,
1060…) beside the edited columns sorted by index (0, 2, 11, 25, 34…) and read "overlap ≈
none". That was a reading error. It went on to infer that the selection must have come from
activation-side evidence, that weight geometry alone could not rediscover the edit, and
that this explained our phase-3a blind failure and the field's no-hint difficulty; those
inferences are withdrawn too.

## Finding 3 — attention-side: layer 23 is the strongest writer onto v [measured]

Top layers by max |o_proj alignment| onto v: **L23 (0.134)**, L19 (0.106), L3 (0.085),
L15 (0.083), L7 (0.070). Notable: openwsp's independent novel edit (this week) targeted
layer 23's down_proj — convergence from two directions (ours: geometry; theirs: search).
L23 is a prime candidate for the phase-6 layer-extension family.

## Finding 4 — token map: the −v side is rich in "alternative-continuation" function words [suggestive]

Top −v-aligned embedding rows: " thereby", " rather", " or", " thus", " first",
" otherwise", " easily", " rather than" family; +v side: rare multilingual fragments.
Reading (SPECULATIVE until generation-time probing confirms): suppressing +v /
amplifying −v overlaps the semantics of offering-another-way-to-continue — plausibly the
"Wait, but let me reconsider…" attractor region. Alignments are weak (|a| ≤ 0.10), so this
is a hypothesis, not a claim. Tied lm_head means the same map covers output logits.

## Part 2 — activation probe (2026-11-13)

Stock model, forward passes only, over saved phase-3c generations. Lists (`sets.json`), probe
script and analysis plan were committed before probing (`b157803`; protocol addendum A1–A4).
8,544 positions from 187 generations: `loop` (in sustained repetition), `pre` (same
generations, before onset), `clean` (stock generations that finished), `deep` (depth-matched
non-looping control). Directions fit on half the generations, scored on the other half.
Numbers: `atlas_activation.json`. u15 / u19 = residual directions of the original edit.

Loop onset itself [measured, `build_sets.py` definition]: stock median 6,500 tokens
(p10 3,000 / p90 11,250); under the original edit 13,250. Stock's clean thinking has median
length 3,747 — clean generations are the ones that finish before the loop would begin.

## Finding 5 — the activation-side loop direction is the edit's direction [measured, held-out]

d = mean(loop) − mean(pre) per layer. cos(d, u15) = **0.71** after L15, cos(d, u19) = **0.74**
after L19 (chance 0.03); 0.4–0.6 at every layer from L1 on, ~0 at the embeddings. Held-out AUC
of the projection: loop-vs-pre and loop-vs-deep **0.97–0.99** from L2 onward (pre-registered
"separates" ≥ 0.75); the authors' own u15 separates equally well (0.98). 40/40 held-out
generations have higher projection in-loop than pre-loop. `fig_a1_layers.png`.

## Finding 6 — the edited columns are enriched among the top loop neurons [measured; reworded 2026-11-24, ISSUES #24]

The 64 edited columns overlap the top-64 loop-vs-pre neurons **20 of 64 at L15** and **9 of
64 at L19** (chance 1.14; pre-registered "coincide" ≥ 8). Median edited column sits at the
94.5th (L15) / 73rd (L19) percentile of effect size. 92% of edited columns have amplitude
sign opposite to their loop-vs-pre shift: neurons that fire more in loops subtract u.
`fig_a2_neurons.png`. So the edited columns are both among the most u-aligned stock columns
(Finding 2) and enriched among the top loop neurons. Until 2026-11-24 this finding was headed
"the edited columns ARE the differentially-active neurons", opened with "Resolves Finding 2"
and concluded that the authors' selection was "activation-driven, as inferred"; that
conclusion rested on the earlier, reversed Finding 2 and is withdrawn.

## Finding 7 — the original edit is a state-dependent brake [measured]

Drive s(t) = Σ a_j·act_j (the push the edit adds along u at each token): L15 **−2.32 in loops
vs −0.56 pre / −0.41 clean / −0.55 deep**; L19 **−3.53 vs −0.66 / −0.63 / −0.75**. AUC
loop-vs-anything ≤ 0.05 (i.e. ≥ 0.95 separation). Aligned to onset, the drive is nearly flat
for 2,000 tokens, then doubles within 250 tokens of onset and keeps growing. `fig_a3_onset.png`.
Not perfectly selective: outside loops the push is still ≈ 1 SD of the residual along u
(0.41 vs SD 0.43 at L15). [inferred] That standing push is a candidate for the phase-5
trade-off (more dose → worse hard-problem answers). Untested.

## Finding 8 — the strongest differential neurons sit upstream, in L10–L14 [measured]

Neurons with |Cohen's d| > 0.8 (loop vs pre): 441 in L10, 396 in L11, 340 in L12, 309 in L13,
435 in L14, vs 233 in L15 and 67 in L19; < 25 in L0–L5. Split-half reliability r ≈ 0.98.
Finding 1 (weight-side: no hotspot) does not hold activation-side: there is a mid-depth band.

## Finding 9 — to-be-stuck generations are distinguishable early [exploratory, confounded]

Within the first 1,500 tokens, held-out AUC to-be-stuck vs clean = 0.83 (positions) / 0.89
(generations, n = 40 vs 23) at L13. Problem difficulty is NOT controlled (clean stock problems
are easier), so this may read "hard problem", not "will loop". A lead only.

## Limits of part 2

- Correlational. The direction marks "this context is repeating"; that it also DRIVES the loop
  rests on the edit's measured effect (phase 3c), not on this probe.
- `deep` text was written by the edited model and read by stock.
- One model (stock 0.8B), one benchmark, 12-gram onset definition.

## Part 3 — question-type map (2026-11-13)

Stock model reading all 500 MATH-500 benchmark questions (no generation). Plan Q1–Q4 written in
the protocol addendum before probing (dated note, not a commit). Fit on half, scored on the other
half. Numbers: `atlas_questions.json`. Script: `probe_questions.py`.

## Finding 10 — subject is readable, modestly [measured, held-out]

Nearest-centroid accuracy over 7 subjects (chance 24%): **54%** at the last token of the
question (after L22; 24% at the embedding control, as designed) and **62%** averaged over the
question's tokens (after L5) vs 43% for the lexical embedding baseline — pre-registered
"decodable" and "model-added" both met. Uneven: Intermediate Algebra 80%, Geometry / Precalculus / Counting
36–37% at the last token. `fig_q1_decoding.png`.

## Finding 11 — the main axis of the question map is difficulty, partly length [measured + post-hoc control]

Hard (level 4–5) vs easy (level 1–2): held-out AUC **0.84** after L12 at the last token —
"separates". The top principal component of the question map orders questions easy → hard.
POST-HOC control (not in the plan): question length alone gives AUC 0.73; with length
regressed out the activation AUC is **0.77** (still ≥ 0.75 from L9 on). So a real difficulty
signal remains, smaller than the raw number.

## Finding 12 — subject-selective neurons live late, L16–L22 [measured]

Neurons with η² > 0.3 for subject: 169–314 per layer in L16–L22, 6–24 in L7–L11 (split-half
r 0.72–0.84). The strongest ones mostly split advanced topics (Intermediate Algebra,
Precalculus) from the rest. `fig_q2_neurons.png`.

## Finding 13 — the loop circuit and the question-type circuit are different places [measured]

- Neurons: top-64 loop neurons vs top-64 subject neurons overlap **0–3 per layer** (chance
  1.14). Loop band L10–L14 (Finding 8); subject band L16–L22 (Finding 12).
- Directions: |cos(loop direction, difficulty direction)| ≤ 0.14 at the last token (< 0.1 at
  most layers; 0.10–0.14 at L15–L19) — largely "unrelated" by the pre-registered rule.
  With the mean-token readout it is 0.1–0.2: in between, not settled.

## Finding 14 — question-reading activity predicts later looping only weakly [measured]

Which problems the original-edit model loops on (81 stuck vs 367 clean): held-out AUC from
the difficulty projection 0.76–0.78 raw, **0.72 with length regressed out** ("weak") — vs
dataset level alone 0.69 and question length alone 0.68. Little beyond "hard, long problems
loop more". Finding 9's early-warning lead therefore still needs the same-problem test.

## Limits of part 3

- All questions are maths; "subject" here is a narrow notion of question type.
- Nearest-centroid and mean-difference readouts are simple on purpose; a stronger probe would
  score higher. Correlational, like part 2.

## Part 4 — causal spot-checks on the loop band (2026-11-13)

Predictions committed first (`97af18a`); 6 of 6 practice passes; full table and verdicts in
`causal/results.md`.

## Finding 15 — the mapped direction is NOT the lever; a loop-gated kick is [measured, n = 64, one draw]

A brake built from the map alone (64 loop-active neurons each at L10 and L14, subtracting d
when they fire) moved thinking-closed from 18.8% to 23.4% — **prediction H1 REFUTED**
(needed +20). The control with the SAME neurons and amplitudes but a RANDOM direction ⟂ d
reached **51.6% closed / 32.8% accuracy** (stock 18.8 / 12.5; original edit 56.2 / 39.1 on
this set). Doses 0.5× and 2.0× of the d-brake: no effect either.

The random directions are not accidentally meaningful (|cos| with u15/u19 ≤ 0.04). They are,
relative to natural variation, a far larger kick: ≈ 27–30 SD vs ≈ 9–11 SD along d.

Reading at the time [inferred, superseded by Finding 16]: the map locates WHEN to act
(loop-gated neurons), not WHAT to write. Finding 5's direction may be a readout of repetition
rather than its cause.

## Finding 16 — replication: size matters and gating is not tested by its control; direction matters too, just not the mapped one [measured, 9 passes; gating reading corrected 2026-11-24]

Predictions committed first (`190cf5a`); table in `causal/results.md`.

- Finding 15 REPLICATES: 3 of 4 random directions on the L10+L14 loop neurons work
  (51.6 / 43.8 / 39.1% closed; one gives 17.2); a new sampling seed gives 43.8.
- The loop neurons are needed (as first written; not established, 2026-11-24): same amplitudes
  on random neurons → 23.4% (nothing). That control's push was 2.6% and 3.8% of the size of the
  loop-neuron arms' pushes, so it does not test the claim; see Corrections and limits at the end.
- Size is needed: the first random direction at one-third size → 28.1% (nothing).
- Finding 15's reading was TOO STRONG. On the published edit's own columns, the published
  direction gives 57.8% but two random directions give 25.0 / 29.7. My pre-registered guess
  ("at least one works") was wrong: there, direction matters.

Unreconciled: at L10+L14 the mapped direction fails and most random ones work; at L15+L19
the published direction works and random ones do not. With 4 and 2 draws this cannot be
called a site effect. The arms that would separate site from direction (d on the published
columns; u on the L10+L14 neurons) have not been run.

## Finding 17 — the published direction is portable; the mapped one is not the lever [measured, 2 passes]

Predictions committed first (`8dc3a17`); table in `causal/results.md`.

- u on the L10+L14 loop neurons (the map's neurons and amplitudes): **48.4% closed** — works,
  where d on the same neurons at the same size gives 23.4%. My pre-registered guess was wrong.
- d on the published columns at L15/L19: **40.6%** — works by the rule, but 17 points under
  u there (57.8%). d is 0.71–0.74 aligned with u at those floors, 0.48–0.51 at L10/L14.
- So u is the one direction that works at both sites; d works only to the extent it is u.
  Finding 5 stands as description (d marks loops) — d is a blurred copy of the lever, not it.
- The map's NEURONS are a good place to act (48.4% vs 57.8%, within noise at n = 64).
- The "large out-of-distribution kick" account (Finding 15) is dead: working pushes range
  from 5 to 30 SD of natural variation, failing ones from 9 to 29 SD.

Unreconciled: random directions unrelated to u work at L10+L14 (3 of 4) and not at L15+L19
(0 of 2). One cell each for the new arms, n = 64, practice set only.

## Finding 18 — on the map's neurons, the part of u that is NOT the loop direction works best [measured, 1 pass; comparison suggestive]

Prediction committed first (`898dadd`). Same L10+L14 loop neurons and amplitudes throughout:
d → 23.4% closed; u → 48.4%; **u with its d-part removed → 62.5% closed / 43.8% accuracy**
(published edit at its own site: 57.8 / 43.8; stock 18.8 / 12.5). Pre-registered T1 ("works")
confirmed, with the pre-stated caveat that most pushes ⟂ d work at this site (random: 3 of 4).

[suggestive] Removing the d-part raised the closed rate by 14 points (≈ 1.7 SE) and the arm
beats all four random draws. Reading: subtracting the direction that MARKS loops (Finding 5)
does not repair them and may get in the way; the effective push lies off that axis. One draw,
n = 64 — a candidate for a seed repeat and a MATH-500 final, not a claim.

## Finding 19 — the map's neurons carry a real repair: 32.6% on MATH-500 [measured, final, n = 500; re-scored 2026-11-24, ISSUES #19]

Predictions committed first (`8a0296b`). `uperp_on_loop` (Finding 18) repeated on a new
sampling seed: 76.6% closed / 40.6% accuracy on practice — not luck. MATH-500 final, same
harness and rescorer as every other final: **32.6%** (163 / 500; re-scored 2026-11-24,
ISSUES #19; 131 / 500 = 26.2% as first reported) against stock 8.4%, random control 3.8%,
blind LLM editors 31.6 / 35.2%, published edit 48.6%. Pre-registered verdict: SUCCESS, below
the published edit (paired difference −16.0 points, 95% interval −20.2 to −12.0, SE 2.2: 20
problems right only by the repair, 100 only by the published edit; −13.2 as first reported).

- The loop-active neurons found by the activation map at L10+L14 are a working repair site on
  the real benchmark, not only on practice problems.
- Practice passes overstated it (they put this arm level with the published edit). Practice
  separates "works" from "does nothing"; it does not rank working edits.
- Not a from-the-map-alone repair: the direction is derived from the published edit, and the
  neurons were selected from stock's activity on these same problems (no answers used).

## Finding 20 — direct MLP-to-MLP "wires" are real one at a time, but explain little of which units light up [measured, descriptive; no predictions made]

`build_wires.py` → `wires.json`, drawn in the light map's 3D view. A wire is the DIRECT path
only: unit i's down_proj column, read by unit j on a later floor through its RMSNorm and
gate/up weights, linearised at the stock model's operating point while reading the 500 benchmark
questions (41,152 positions). All 276 floor pairs × 3584² unit pairs computed; contribution
of a wire to a job = wire × job shift of the source / job SD of the target (Cohen's d units); the 600
strongest per job are kept (both ends lit at |d| ≥ 0.5).

- Formula checks pass on all 24 floors (this model's RMSNorm is (1 + w)·x/rms; a unit is
  silu(gate)·up).
- One wire at a time: raising the source unit by 1 SD in a real forward pass moved the target
  the predicted way for **96 of 118** top wires (81%), but the measured change was a median
  **0.30×** the direct prediction (quartiles 0.08–1.0; r = 0.48). Picking the largest
  predictions inflates them, and later layers partly cancel the push.
- All wires together: summed over every lower floor they do not reproduce which units light
  up — **r = 0.02–0.23** across the nine jobs (lit units). The same first-order read-out
  applied to EVERYTHING the unit reads (measured residual shift: embeddings + attention +
  lower MLPs) reaches r = 0.41–0.49. So the linear picture is itself only moderate, and inside
  it the MLP-to-MLP part is small (r between the two predictions 0.06–0.10).
- Why: each floor's MLP write is most of that floor's change to the stream (cos 0.4–0.95), yet
  the cumulative MLP write is nearly orthogonal to the net stream shift at mid floors
  (cos ≈ 0.01 at L10 for Algebra and Hard question) — the embedding/attention part offsets it.
- Loop job: the summed wires line up with the pattern only near the top (r 0.20–0.36 on
  L18–L21, ≈ 0 below).
- Hubs: a few source units carry most strong wires — L14 unit 46 is the source of 460 of the
  600 strongest "Hard question" wires (265 / 239 / 203 for Number Theory / Geometry /
  Precalculus); L5 unit 1087 of 532 for Counting & Probability; L19 unit 11 of 173 for the loop.
  Partly by construction: a unit with a large job shift scales all its wires.
- The map's L10/L14 brake units do not feed the published L15/L19 columns by direct wire
  (contributed d ≤ 0.02; those columns barely move in loops, mean d 0.03–0.04).

Caveats: one operating point (reading) for every job, including the loop job, which was
measured while generating; RMSNorm's projection term ignored; no path through attention is
traced; the nudge test used 8 benchmark questions and only each job's 15 strongest wires. The
reading-job numbers in this finding (the summed-wire correlations by job and the reading-job
hubs, L14 unit 46 and L5 unit 1087) describe wire lists that were later replaced and never
committed; Finding 22 and the section after it give the current ones (see Corrections and
limits, added 2026-11-24).

## Finding 21 — the hubs, tested: one real switch that is not a lever, one artefact of size, one weak [measured; forward passes + 1 practice pass]

Predictions committed first (`6cabd32`); tables in `causal/results.md`. Each hub was held at one
constant at every position and every unit's d recomputed (`test_hubs.py` → `hubs.json`).

- **L19 unit 11 (loop).** Its wires hold up: 160 of 173 targets moved the predicted way at
  0.68× the predicted size (r 0.84). Holding it at its before-the-loop level in replay turns
  off 149 of the 496 loop lights on floors 20–23 (pre-registered H2 "the pattern does not
  depend on the hub" FAILED for it). But zeroing its column and generating leaves looping
  unchanged: 26.6% closed vs stock 18.8% (H4 confirmed, within noise). A switch for the loop
  lights above it, not a lever on the loop. It reaches none of the 3,186 loop lights at or
  below floor 19.
- **L14 unit 46 (Hard question, Number Theory, Geometry, Precalculus).** Its wires are noise:
  49–61% of targets the predicted way, median size 0.00–0.06× (H1 FAILED). The unit is a first-token spike: |activation| ≈ 75 on
  the first token of every benchmark question and ≈ 0.11 elsewhere (SD 8.2; next-loudest unit
  on the floor 0.56, floor median 0.06). Its per-question average is therefore ≈ 75 / question
  length, so its "Hard question" light is a question-length artefact, and its size inflates
  every wire it sends.
  [risk for part 3] The reading jobs average over all positions including the first, so other
  first-token units may be lit for the same reason; not yet counted.
- **L5 unit 1087 (Counting & Probability).** Wires real but weak (85% the predicted way,
  0.19×); the pattern survives without it (r 0.993).
- Correction: the addendum described L19 unit 11 as "quieter" in loops; its activation is
  negative and grows from −0.30 to −2.97. Negative d means "number went down", not "less active".

## Finding 22 — the subject lights are mostly real; the "Hard question" lights are mostly question length [measured; forward passes]

Predictions committed first (`bfaf4e7`); `probe_questions.py --no-prefix`, `remeasure_reading.py`
→ `reading_remeasure.json`. Every benchmark question opens with the same 3 template tokens; the
reading jobs were re-measured with those left out of every average, and again with question
length (linear + log) regressed out of every unit. A unit that tracked the job and not length would
keep 87–100% of its d under that regression (1 − r² of job label on log length).

| job | lights at 0.8: all positions → no opening | old lights still lit | lit units keep (median) | job-only unit would keep | lights left under length control |
|---|---|---|---|---|---|
| Hard question | 289 → 200 | 62.6% | **46%** | 87% | **0** of 200 (strongest 0.76; at 0.5: 5,172 → 606) |
| Algebra | 177 → 181 | 93.2% | 83% | 96% | 32 of 181 |
| Number Theory | 6,920 → 6,924 | 92.9% | 82% | 96% | 3,258 of 6,924 |
| Geometry | 13,687 → 13,556 | 95.2% | 86% | 97% | 8,925 of 13,556 |
| Precalculus | 8,555 → 8,322 | 92.0% | 86% | 97% | 4,916 of 8,322 |
| Prealgebra | 1,857 → 1,753 | 91.3% | 94% | 100% | 1,224 of 1,753 |
| Intermediate Algebra | 8,953 → 8,752 | 94.2% | 96% | 99% | 7,216 of 8,752 |
| Counting & Probability | 14,777 → 14,990 | 98.8% | 101% | 100% | 14,025 of 14,990 |

- **R1 (L14 unit 46 drops out) FAILED on 3 of 4 jobs.** Without the opening its d does not
  vanish, it flips sign: Hard question +0.88 → −0.54, Number Theory −0.66 → +0.77, Precalculus
  +0.69 → −0.78 (Geometry +0.64 → −0.17). Its other tokens track length too, the other way.
- **R2 (≥ 90% of lights stay, r ≥ 0.98) holds for all seven subjects** (91–99%, r 0.992–0.9997)
  **and FAILS for Hard question** (62.6%, r 0.969).
- **R3**: Hard question keeps none of its lights under length control (predicted < 60%: holds);
  Geometry keeps 64.9% and Number Theory 46.4% of the old lights (predicted ≥ 70%: fails).
- Reading: the difficulty "job" at unit level is mostly a length map — hard questions are 64%
  longer (99 vs 60 tokens) and its units lose half their effect once length is accounted for.
  The subjects are real with a length component, largest for the short-question subjects
  (Number Theory, Algebra) and the long one (Geometry); Counting & Probability has none.
  This matches Finding 11's "difficulty, partly length", now at the level of single units.
- Part 3 re-run on the no-opening data: last-token decoding identical (subject 0.54, difficulty
  AUC 0.841); mean-pooled 0.62 → 0.624 and 0.775 → 0.763; subject-selective neuron counts within
  a few per floor (L21: 314 → 309). Findings 10–13 stand as far as the opening tokens go.
- The light map, wires and page now use the no-opening measurement. Rebuilt wires: the L14
  unit 46 hub is gone from all four jobs, and the nudge check improves to 105 of 122 the
  predicted way at 0.47× (was 96 of 118 at 0.30×, r 0.48 → 0.79). Finding 21's hub numbers
  were measured on the old wire lists.

## Finding 23 — there is almost nothing empty to prune; the two loud units are removable [measured; forward passes + 2 practice passes]

Predictions committed first (`d72262a`); tables in `causal/results.md`.

- Loud-but-empty units (huge on token 0, quiet elsewhere): **2 of 86,016** — L14 unit 46 and
  L3 unit 0. Silent units: **none** (quietest per-position SD 0.02).
- Zeroing the two loud units changes 1.09% of next tokens (two ordinary units: 0.48%) and the
  published-edit model still works without them: 64.1% closed / 50.0% accuracy on practice
  (reference 57.8 / 43.8). Waste, and removable — but two units.
- Smallest-first pruning (size = activity SD x column norm) beats random at every dose, yet
  is never free: 5% removed → 94.3% of next tokens unchanged, 20% → 86.6%, 50% → 69.2%.
  With the smallest 5% (4,294 units) gone the published-edit model still closes 62.5%
  (accuracy 34.4%, 9 points down, about one SE at n = 64).
- So the model is not carrying dead weight; pruning is a way to TEST labels (knock a labelled
  group out, see what breaks), not yet a way to shrink it.

## First labels for every unit [descriptive]

`label_units.py` → `labels.csv` (one row per unit), `labels_summary.json`. A tag needs
|d| >= 0.8 (subjects: with question length regressed out) or |r| >= 0.5 with log question length.

| tag | units | busiest floors |
|---|---|---|
| subject (any of the 7) | 34,349 | 20, 21, 19 |
| question length | 6,351 | 14, 23, 13 |
| loop marker | 3,682 | 10, 14, 11 |
| edited by authors / used in our repair | 128 / 128 | 15, 19 / 10, 14 |
| first-word spike | 2 | 3, 14 |
| no label yet | 45,526 (52.9%) | |

Overlaps: 1,317 loop markers are also subject units; 2,126 subject units also track length;
10,122 units carry two or more subjects. Per subject: Counting & Probability 17,064, Geometry
10,271, Intermediate Algebra 8,467, Precalculus 6,140, Number Theory 3,815, Prealgebra 1,409,
Algebra 36. No unit earns a "hard question" tag (Finding 22). These are correlational labels
inside maths questions only; none is confirmed by a knockout yet.

## Finding 24 — knockout: three subject labels are selective, but none is promoted to "used for" under the registered rule [measured, held-out half; forward passes; corrected 2026-11-24]

Predictions committed first (`b96374e`); table in `causal/results.md`. Units chosen on the
even-index problems (each subject's 1,000 most-more-active units), tested on the stock model's
answers to the odd-index problems; selectivity = extra loss on the knocked-out subject's text
over the mean on the other six; control = same count of random units per floor.

- **Number Theory 1.53, Geometry 1.47, Intermediate Algebra 1.37** (90% ranges all above 1.19;
  controls 0.96–1.02): removing the tagged units hurts that subject more. K1 holds.
- Counting & Probability 1.43 but its control is 1.34 (its text is fragile to any removal);
  Prealgebra 1.04, Precalculus 0.99, Algebra 0.96: no selectivity. K2 (5 of 7) and K3 fail.
- It is a tilt, not a module: Geometry's knockout costs geometry text +7% loss and every other
  subject +5%. Most of what subject-tagged units do is shared across maths.
- Labels: none is promoted; every tag stays "goes with". The rule fixed in advance
  (`phase7/protocol.md`, addendum 9) promotes a label only if K1–K3 all hold; K2 and K3
  failed, so three (or four) subjects show selectivity above 1 but none is promoted under the
  registered rule. Until 2026-11-24 this bullet promoted Number Theory, Geometry and
  Intermediate Algebra to "used for (weakly)", which the rule does not allow (corrected
  2026-11-24).

## Light map now uses length-controlled reading jobs [descriptive; 2026-11-14]

`jobs.job_stats()` regresses question length (linear + log) out of the seven subjects and "Hard
question" before computing d, so a subject light on the page is exactly a `subject: X` label
(asserted in `build_lightmap.py`); hover shows every label and which passed the knockout (the
page flags a subject as held when its knockout selectivity is reliably above 1 and above its
control — it does not apply the registered K1–K3 rule of Finding 24, under which no label is
promoted).
Lights at 0.8: Algebra 36, C&P 17,064, Geometry 10,271, Intermediate Algebra 8,467, Number
Theory 3,815, Prealgebra 1,409, Precalculus 6,140, Hard question 0. Wires rebuilt on the same
numbers: nudge check 113 of 127 same direction, r 0.93, median measured/predicted 0.63 (was
105 of 122, r 0.785, 0.47). The reading-job hub tests in `hubs.json` were run on the old wire
lists and are no longer shown; the loop job is unchanged.

## Finding 25 — stacking the map's repair on the published edit gives no gain at half or full dose [measured, 2 finals + 1 void run, n = 500; corrected 2026-11-24, ISSUES #19, #20]

Predictions committed first (`d146c4c`); full tables in `causal/results.md`. Reason to try:
94.7% of the published edit's capped answers (125 of 132) are still loops, and the map's
L10+L14 neurons are a working repair site on their own (Finding 19).

- `stack_half` (published edit + half the `uperp_on_loop` change): **44.8%** (224 / 500;
  re-scored 2026-11-24, ISSUES #19; 37.0% as first reported) against the published edit's
  48.6%. Paired: 39 problems right only by the stack, 58 only by the published edit,
  difference −3.8 points (95% interval −7.6 to 0.2, SE 2.0) — no difference. Loops unchanged
  (121 vs 125).
- `stack_full` (full change): the first run is **VOID — a corrupted run** (ISSUES #20,
  2026-11-24). Until that date this entry reported 4.4% with a paired difference of −35.0
  points ("LOSS") and read it as the model breaking; both are withdrawn. The score is 6.4%
  after re-scoring and is not a measurement of the model. The clean rerun of 2026-11-24
  scores **46.2%** (231 of 500).
- Both passed the practice gate (closed 62.5% and 56.2%); practice gave `stack_full` 40.6%
  accuracy.

Why the first `stack_full` run is void. 460 of the 500 generations (92.0%) break into fragments
of several scripts, and whether a generation was corrupted depends on the request's position in
the batch, which a property of the model cannot know. Counted with the first corrected checker
of that day (when the run read 5.0% in all), the first 287 requests (the first wave the engine
admitted) were 8.7% correct (25 of 287) and the last 213 were 0.0% (0 of 213; exact test,
p = 7.6e-7); among the other 25 saved 500-problem runs the same split never reaches p = 0.6.
Under the final checker the dependence is still there: all 32 right answers fall among the
first 266 requests, and the validity check recorded in the result file
(`phase3c/runcheck.py`; `phase3c/p7_stack_full_rescored.json`) reads 12.0% correct in the
first half of the rows against 0.8% in the second half (Fisher exact test, p = 1.1e-7). The
same model's 64-problem practice pass the same day scored 40.6%. The engine log showed
nothing. The cause is not isolated (an engine-and-model interaction under late admissions is
the standing suspicion; ISSUES #13 records the same signature in a voided phase-5 run). The
first run stays void with this evidence unchanged; its clean rerun of 2026-11-24 scores 46.2%,
so the "collapse at full dose" recorded on 2026-11-14 was the corrupted run, not the model.

Rerun of the full-dose model, 2026-11-24: **46.2%, and the model is not broken.** The voided
run's model (files rebuilt and verified equal to the intended change) was rerun under the
current runner (`p7_stack_full_v2`; `phase3c/p7_stack_full_v2.json`). It scores 46.2% (231 of
500), and the validity checks are clean: fragment tails 0.0%; first half of the requests 46.8%
right against 45.6% for the second half (p = 0.86). Paired with the published edit (48.6%):
−2.4 points (95% interval −5.8 to 0.8; 30 problems right only under the stacked model, 42
only under the published edit; SE 1.7) — no real difference. Against the half dose (44.8%):
+1.4 (interval −2.6 to 5.4). Full numbers in `causal/results.md`.

Conclusion of the stacking test: stacking the map's repair on the published edit gives no gain
at either dose (−3.8 points at half dose, −2.4 at full dose; neither distinguishable from
zero) and costs nothing detectable. Reading [inferred, resting on the half-dose arm's
untouched loop count]: the L10+L14 repair is not a second, independent brake. Its direction is
the published edit's, so stacking is more of the same push; the loops that remain after the
published edit are not reached by it.

## Light map gets a lever layer: which units push along the published direction [descriptive; no predictions made; 2026-11-20]

`build_lightmap.py` `lever()`; a tenth button on the page, not a job. Per unit: (activity in the
loop − before it) × (its down_proj column · u) ÷ (clean-thinking SD of the stream along u after
that floor); u = u15 for floors 0–15, u19 above. Positive = toward the loop side of u. Direct
writes only (no attention), same saved activations as the loop job, forward data only — no
generation. Self-check in the builder: the same activations and columns reproduce Finding 7's
drive (loop − pre) at L15 and L19 to 0.01.

- Net push per floor, in SDs: **L15 0.76, L19 0.62**, then L7 0.47, L3 0.29; L10 −0.17, L14 −0.02.
  The two floors the published edit uses are the two where the units' direct writes along u add up.
- The published edit's 64 units are mostly the top lever units: **46 of 64 at L15, 32 of 64 at
  L19** in the top 64 by lever, against 20 and 9 by loop light (Finding 6). 59 of 64 push toward
  the loop side on both floors; together they carry 0.70 of L15's 0.76 and 0.63 of L19's 0.62.
- The map's brake units (top loop lights at L10, L14) sum to 0.06 and −0.02: lit, but their
  writes along u cancel.
- Strongest single unit: L19 unit 11, 0.38 SD. Zeroing it did not repair (Finding 21), so a
  large push is not proof of a lever.

[inferred] Cohen's d ranks units by how much their activity changes and ignores where their
output points; the published selection looks like "change × alignment with u". Untested: whether
units chosen by lever score make a better edit site than the map's, at any floor.

## Finding 26 — lever-ranked units are not a better edit site at L10+L14 [measured, 1 practice pass, n = 64]

Predictions committed first (`93db98f`); table in `causal/results.md`. `u_on_lever` is
`u_on_loop` with the 64 units per floor chosen by |lever score| instead of loop light — same
floors, published direction and push size; 24 and 27 of the 64 units are shared.

- **50.0% closed / 26.6% accuracy**, against `u_on_loop` 48.4 / 32.8 (stock 18.8 / 12.5). It
  works, and it is not detectably different: +1.6 points closed, one problem. All four
  pre-registered predictions held, including "no detectable difference".
- Reading: at these floors the units' own writes along u cancel, the edit overwrites that part
  of each column anyway, and the gated push is the same by construction — so which loop-gated
  units carry it matters little. The lever layer describes where the published edit's units sit
  (L15, L19); it is not a selection rule shown to help.
- Limits: one draw, practice set only (cannot rank two working edits, Finding 19); the unit
  lists overlap by about 40%; not run with the `uperp_on_loop` direction or at L15+L19.

## Next

- L19 unit 11: what makes it fire in loops (its own inputs), since it sits upstream of a third
  of the top-floor loop lights.
- A length-matched difficulty contrast (hard vs easy questions of the same token count), or the
  light map rebuilt on length-controlled d: the "Hard question" job as it stands is mostly length.
- Findings 12–13 (subject-selective neurons, loop vs question circuits) under length control;
  they are unchanged by dropping the opening tokens but were never length-controlled per unit.
- What separates u from d: apply the part of u that is not along d on its own; more random
  draws at both sites to estimate a hit rate.
- Difficulty-controlled foreseeability: phase-5 finals have 4 samples per problem with mixed
  outcomes — same problem, stuck vs clean.
- Question types beyond MATH-500: benchmark questions from other task families (code, plain QA,
  instruction following), length-matched — the within-maths map (part 3) is mostly an easy/hard
  axis.
- Generation-time token probe: which tokens' logits move when v is perturbed ±.
- Where the embedding-row edits (GLM-Flash, mimo) sit relative to d.

## Corrections and limits (added 2026-11-24)

- **Every final accuracy in this document was re-scored** with the final answer checker
  (ISSUES #19; corrected twice on 2026-11-24): map repair 32.6% (26.2% as first reported),
  half-dose stack 44.8% (37.0%), full-dose stack 46.2% (first run void, ISSUES #20; clean
  rerun 2026-11-24).
- **Practice passes cannot rank working edits.** They are 23 models in 25 passes, one run per
  arm, 64 problems each (a 95% interval on 64 problems is about 12 points wide); the same
  model run with a second seed moved from 62.5% to 76.6% closed. Within-group rankings of
  arms that all "work" are not interpretable. Examples read as differences but are not: map
  repair 40 of 64 closed against the random direction 33 of 64; the lever units 32 against
  40; the published columns with the mapped direction 26 against random directions 16 and
  19.
- **"The loop neurons are needed" is not tested by its control** (Finding 16). The
  random-neuron arm's push was 2.6% and 3.8% of the size of the loop-neuron arms' pushes; an
  ungated push that small does nothing on its own. The lever arm replaced 40 and 37 of the 64
  units and still worked (50.0% closed), so which units carry it matters little.
- **Finding 20's reading-job numbers** describe wire lists that were replaced and never
  committed; later sections give the current ones.
- **Finding 24:** the rule promotes a label only if K1–K3 all hold; K2 and K3 failed, so
  three (or four) subjects show selectivity above 1 but none is promoted under the registered
  rule.
- **Prediction H4** (accuracy at least equal to pass 2, first batch) was not met: 11 of 64
  against 13 of 64; it was reported "uninformative".
- **Addendum 7** fixed in advance that a length-regressed variant would be "reported, not
  adopted"; it was adopted the next day.
- **No practice pass can be re-scored.** The practice generations were not saved. Practice
  numbers in this document are as recorded with the old checker.
