# The Matrix Atlas — Phase 7 working document

Living document; each entry carries evidence + a confidence label. Direction v = the loop
direction (unit top-singular vector of the L15 edit block; cos(v15,v19)=0.849).

## Finding 1 — the loop direction is written DIFFUSELY, not locally [measured]

Every layer's MLP (down_proj) writes onto v with statistically similar strength:
mean |alignment| ≈ 0.006–0.008, max ≈ 0.05–0.09 across all 24 layers. No hotspot at the
edited layers (15/19 are unremarkable in this metric). The loop direction is a
network-wide residual-stream property, not the product of one layer.

Evidence: `atlas_geometry.json` → down_proj_alignment (full per-layer table).

## Finding 2 — the edited columns were NOT chosen by weight-side alignment [measured]

The 64 columns the original authors edited per layer do not appear among the top-aligned
columns of either edited layer (L15's top-aligned: 1155, 1787, 1060…; their picks: 0, 2, 11,
25, 34…; overlap ≈ none). Their selection must have come from activation-side evidence
(differential activity during loops), not weight geometry. Corollary: weight-geometry alone
cannot rediscover their edit — consistent with our phase-3a blind failure and the field's
no-hint difficulty.

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

## Finding 6 — the edited columns ARE the differentially-active neurons [measured]

Resolves Finding 2: the 64 edited columns overlap the top-64 loop-vs-pre neurons **20 of 64 at
L15** and **9 of 64 at L19** (chance 1.14; pre-registered "coincide" ≥ 8). Median edited column
sits at the 94.5th (L15) / 73rd (L19) percentile of effect size. The authors' selection was
activation-driven, as inferred. 92% of edited columns have amplitude sign opposite to their
loop-vs-pre shift: neurons that fire more in loops subtract u. `fig_a2_neurons.png`.

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

Stock model reading all 500 MATH-500 prompts (no generation). Plan Q1–Q4 written in the
protocol addendum before probing (dated note, not a commit). Fit on half, scored on the other
half. Numbers: `atlas_questions.json`. Script: `probe_questions.py`.

## Finding 10 — subject is readable, modestly [measured, held-out]

Nearest-centroid accuracy over 7 subjects (chance 24%): **54%** at the last prompt token
(after L22; 24% at the embedding control, as designed) and **62%** averaged over prompt tokens
(after L5) vs 43% for the lexical embedding baseline — pre-registered "decodable" and
"model-added" both met. Uneven: Intermediate Algebra 80%, Geometry / Precalculus / Counting
36–37% at the last token. `fig_q1_decoding.png`.

## Finding 11 — the main axis of the question map is difficulty, partly length [measured + post-hoc control]

Hard (level 4–5) vs easy (level 1–2): held-out AUC **0.84** after L12 at the last token —
"separates". The top principal component of the question map orders questions easy → hard.
POST-HOC control (not in the plan): prompt length alone gives AUC 0.73; with length regressed
out the activation AUC is **0.77** (still ≥ 0.75 from L9 on). So a real difficulty signal
remains, smaller than the raw number.

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
dataset level alone 0.69 and prompt length alone 0.68. Little beyond "hard, long problems
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

## Finding 16 — replication: gating and size matter; direction matters too, just not the mapped one [measured, 9 passes]

Predictions committed first (`190cf5a`); table in `causal/results.md`.

- Finding 15 REPLICATES: 3 of 4 random directions on the L10+L14 loop neurons work
  (51.6 / 43.8 / 39.1% closed; one gives 17.2); a new sampling seed gives 43.8.
- The loop neurons are needed: same amplitudes on random neurons → 23.4% (nothing).
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

## Next

- What separates u from d: apply the part of u that is not along d on its own; more random
  draws at both sites to estimate a hit rate.
- Difficulty-controlled foreseeability: phase-5 finals have 4 samples per problem with mixed
  outcomes — same problem, stuck vs clean.
- Question types beyond MATH-500: prompts from other task families (code, plain QA, instruction
  following), length-matched — the within-maths map (part 3) is mostly an easy/hard axis.
- Generation-time token probe: which tokens' logits move when v is perturbed ±.
- Where the embedding-row edits (GLM-Flash, mimo) sit relative to d.
