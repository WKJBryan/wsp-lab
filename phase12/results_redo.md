# Phase 12 results (redo) — what the saved data can and cannot test

**Date:** 2026-11-26 · **Protocol:** `phase12/protocol_redo.md` (written before any number below was
computed; its SHA-256 `8472ae31…00527b` is recorded in `phase12/redo.json`) · **Script:**
`phase12/redo.py` (CPU only; numpy; no model loaded, no forward pass, no generation) · Models:
Qwen3.5-0.8B (dense, 24 layers) and the pruned Gemma-4 26B-A4B wound (MoE, 30 layers). **Revised
2026-11-26 after an independent review** (section 6 lists what changed). This file changes nothing
in `phase12/protocol.md`, `results.md` or `correspondence.json`; the verdict withdrawn on
2026-11-24 (ISSUES #21) stays withdrawn. Nothing was committed while the redo was run.

## Verdict

**Of the three parts, only writer geometry has a cross-model test that can count for or against
correspondence, and it counts against. In every layer of both models the loop direction is
written more focally than a random direction, and Qwen's about 1.35 times the wound's (95%
interval 1.19 to 1.53; a margin of ±0.5 layer-to-layer standard deviations is excluded). The
result holds when neighbouring layers are treated as dependent. The saved data cannot exclude
two explanations: the wound's directions were fitted on 5 texts against Qwen's 11, and nothing is
known about how focally the wound writes other state differences. Loop dynamics has Qwen-only
evidence. Termination has a necessary-condition test, which passes.**

| Part | Test (unit, count) | Result | Bar | Outcome |
|---|---|---|---|---|
| Loop dynamics | Across models | — | — | **Not testable** from saved data (section 1.1) |
| Loop dynamics, Qwen only | Q1-a: is the state's step at the onset specific to looping texts? (texts, 80 looping against 60 long non-looping) | A = 0.988, 95% interval 0.964 to 1.000; 0 of 100,000 label shuffles reached A | A ≥ 0.80 and p < 0.001 | **Passes** |
| Loop dynamics, Qwen only | Q1-b: does the state lag the token-level onset? (texts, 79) | median lag 151 tokens, 95% interval 102 to 226 | interval wholly above 0 | **Lags** |
| Termination | T2: does boosting only the exit rows cut cap-hits, in both models? (problems, 500 and 64, paired) | cap-hits down 55.2% in Qwen (interval 50.6 to 59.8) and 63.5% in the wound (50.8 to 75.0); exact p 1.1e-75 and 2.0e-11 | p < 0.05 and ≥ 25% in both | **Passes** — a necessary condition (section 2.2) |
| Writer geometry | W1: is the loop direction written more focally than a random direction? (layers, 24 and 30) | 24 of 24 and 30 of 30 layers above the null | ≥ 90% of layers | **Focal** in both |
| Writer geometry | W2: is the degree of focality the same? (layers, 24 and 30) | Qwen's excess over its own null is 1.35 times the wound's (95% interval 1.19 to 1.53) | inside ±0.5 of the layer-to-layer spread | **Differ** — counts against correspondence |

Every interval is a percentile bootstrap over 1,000,000 resamples (the protocol registered
10,000; section 6 says why the number was raised).

Two checks back the writer-geometry numbers: the weight-file reader reproduces the saved
embedding-row percentiles on both models to within 0.00003, and the alignment arithmetic,
re-derived for Qwen from saved activations, reproduces the 24 saved Qwen ratios with a correlation
of 0.98 and a median difference of 4.7%.

## What the numbers say, plainly

**Writer geometry counts against correspondence.** In every layer of both models a few writer
columns line up with the loop direction far better than any column lines up with a random
direction. Measured against each model's own baseline, Qwen's is about a third more concentrated
than the wound's. That is the one cross-model test that can count against correspondence, and it
does.

**Loop dynamics can be described, not compared.** Each model has one averaged curve. Qwen's
crosses half way up about 170 tokens after the repetition begins, the wound's about 79 tokens
after; neither number has an interval. For Qwen alone the saved per-token data show that the step
is real (non-looping texts show almost none of it) and that the state follows the words: it
crosses half way up a median of 151 tokens after the token-level onset.

**Termination: both models respond to the same kind of edit.** T2 is a cross-model test and it
passes: making the stop token's output row larger cuts cap-hits by 55% in Qwen and 63% in the
wound. What the data cannot show is specificity — that the exit row, and not just any row, is what
matters. No saved run boosts another embedding row the same way.

## 1. Loop dynamics

### 1.1 Across the two models: not testable

Each model has exactly one 25-bin curve, averaged over its texts and layers. There is no
per-text value on the Gemma side, so no Gemma number has an interval and no pairing by text is
possible. The two curves also share their shape by construction (the direction is the
loop-minus-before difference and each curve is z-scored by its own loop window). A correlation
between them is therefore not reported.

### 1.2 What each saved curve shows on its own (description, no bar)

Half way up means the middle between the level of the four bins before the onset and the level
of the last four bins. Bin 12 begins at the onset; a bin is 40 tokens.

| | Qwen | Gemma wound |
|---|---|---|
| texts in the curve | 23 | 10 |
| onset (tokens, min / median / max) | 761 / 4,489 / 9,975 | 151 / 263 / 675 |
| level before the onset (bins 8–11) | −0.936 | −1.495 |
| level of the plateau (bins 21–24) | −0.105 | −0.002 |
| size of the step | 0.831 | 1.493 |
| half-way crossing | bin 16.25, **170 tokens after the onset** | bin 13.97, **79 tokens after the onset** |
| texts covering each bin fully | all 23, in every bin | 3 in bins 0–2, 4 in bins 3–5, 5 in bin 6, 8 in bins 7–8, 10 from bin 9 |

The Gemma curve rests on thin ground before its onset: seven of its ten texts have an onset under
500 tokens, and for those the saved window begins at the first token, so their early bins contain
the chat prompt and the question, not the model's own text.

These are two descriptions. They are not compared with any null and they cannot fail.

### 1.3 Qwen only — Q1-a: the step is specific to looping texts (passes)

The saved per-token projections onto the edit direction of the published repair (which lines up
with the phase-7 loop direction to cosine 0.71–0.74), for 80 looping texts against texts that do
not loop, at matched positions. The step is the level 1,000–1,500 tokens after an anchor minus
the level 2,000–1,500 tokens before it, in the units of the projection.

| Layer | Step, looping texts (80): median | Step, long non-looping texts (60): median | A (probability that a looping text's step is the larger) | 95% interval | label shuffles that reached A |
|---|---|---|---|---|---|
| 19 (primary) | 2.41 | 0.17 | 0.988 | 0.964 to 1.000 | 0 of 100,000 |
| 15 (replication) | 1.38 | 0.09 | 0.994 | 0.980 to 1.000 | 0 of 100,000 |

The second control, the stock model's own terminating texts, agrees: only 25 of 47 are long
enough to host an anchor; their median step is 0.28 (layer 19) and 0.18 (layer 15), with
A = 0.986 (0.959 to 1.000) and 0.985 (0.9595 to 1.000); no shuffle reached A there either.

So the step is not a drift that any long text shows. With no shuffle reaching A, the one-sided p is
at its floor, 0.00001 (one more than the count, divided by 100,001).

### 1.4 Qwen only — Q1-b: the state lags the token-level onset (lags)

| Layer | texts with a lag | median lag | 95% interval | later / earlier / at the onset | sign test, two-sided |
|---|---|---|---|---|---|
| 19 (primary) | 79 of 80 | **151 tokens** | 102 to 226 | 65 / 13 / 1 | 1.8e-9 |
| 15 (replication) | 79 of 80 | **120 tokens** | 70 to 160 | 60 / 19 / 0 | 4.2e-6 |

One text (m292) has no lag: its smoothed series is already at the half-step level when the search
window opens, 1,500 tokens before the onset. The spread is wide: the middle half of the lags lies
between roughly 55 and 340 tokens at layer 19. In four texts (m119, m223, m373, m488) the state
crosses the level more than 1,000 tokens *before* the token-level onset, at both layers; the
script stores each text's id beside its lag.

The interval is the same to the token in three other seeds of the bootstrap (section 6 says why
that needed more resamples than the protocol registered).

The lag belongs to the detector as much as to the model: it is measured against the phase-7
onset (the first 250-token bin in which half of all 12-token windows have been seen before,
sustained for four bins). It is not the same measurement as the 170 tokens of section 1.2, which
uses another detector, another direction and other texts.

### 1.5 What is not established

Nothing here says whether the Gemma wound behaves the same way. The Qwen results are findings
about one model.

## 2. Termination

### 2.1 T2 — boosting only the exit rows cuts cap-hits, in both models (passes)

Paired by problem; "recovered" means capped before and not after, "lost" the reverse.

| | Qwen: `</think>` row × 1.50 | Gemma wound: the three stop rows × 1.6 |
|---|---|---|
| problems | 500 | 64 |
| capped before → after | 453 → 203 | 63 → 23 |
| recovered / lost | 250 / 0 | 41 / 1 (problem m48) |
| relative reduction, 95% interval | **55.2%** (50.6 to 59.8) | **63.5%** (50.8 to 75.0) |
| exact McNemar p | 1.1e-75 | 2.0e-11 |
| accuracy before → after, change with 95% interval, by the checker stamped `phase3b/scorer.py#d74fdebd340f` | 9.4 → 42.2, +32.8 points (28.4 to 37.2) | 0.0 → 0.0 (0 of 64 both times), 0.0 points (0.0 to 0.0) |
| boosted generations that ended within 32 tokens | 0 of 297 (0 to 1.2%) | 1 of 41 (0.1 to 12.9%) |

The accuracy row carries the checker stamp because the answer checker is being replaced and every
saved run is being re-scored as this is written; under another checker these two figures will
differ, and everything else in the table will not (section 6).

Reported, not part of the bar:

- Against Qwen's second stock run (466 capped): 56.4% (51.9 to 60.9); accuracy 8.8 → 42.2 by the
  same checker.
- The Gemma arm phase 12 registered (attempt 1, pass 2 — stop rows together with four other row
  groups): 63 → 27, 57.1% (45.2 to 69.4), accuracy 0.0 → 3.1 (0 → 2 of 64). It agrees with the
  exit-rows-only arm on cap-hits.
- The wound's two baseline passes (attempt 1 and attempt 5, same model, seed and problems) have
  the same outcome on all 64 problems.
- Unmatched controls, as counts: Qwen random noise of matched size on the same row, 453 → 479
  capped; Qwen the same row scaled on the input side only, 453 → 449; Gemma final norm × 0.4
  with no exit row, 63 → 58.

Two readings worth stating. First, "termination recovered" does not mean "pushed out at once":
almost every boosted generation that ended ran for more than 32 tokens. Second, by the checker
named, the exit-row edit raises Qwen's score by 32.8 points and leaves the wound at 0 of 64.

**What "valid" means for these inputs.** All ten generation-run files passed the script's gate:
each carries a validity record whose list of problems is empty. For the five Qwen runs both
checks in that record were judged (the fragment-tail check and the admission-order test). The
five Gemma practice passes were judged on neither. Their record reports the fragment-tail rate
(3.1% to 14.1% of generations end in characters of a non-Latin script) without passing or failing
it, and the admission-order test is not run on fewer than 100 rows where they have 64. For these
five, "valid" means only that no problem is listed. The result file records this for each input.

### 2.2 What a pass of T2 does and does not show

It shows that in both models the failure to stop is relieved by making the stop token's output
row larger. As a necessary condition, T2 could have counted against correspondence — had one
model not responded, it would have failed — but a pass cannot count for it. Raising a stop
token's output row makes any sampling model stop sooner, healthy or not. What is missing is
specificity.

### 2.3 Not testable from saved data

- **Is the exit row special?** No saved run in either model boosts a non-exit embedding row alone.
- **A geometric signature of the exit row.** The "low norm" rule was written after the four
  percentiles were looked at; no p-value is computed for it here, because re-testing it on the
  same four numbers would be circular.
- **"No accuracy loss" on 64 problems.** One problem is 1.6 points there.

## 3. Writer geometry

### 3.1 The null

For each layer, 4,000 random unit directions against the layer's real writer columns from the
weight file. A random direction gives a ratio (largest cosine over mean cosine) of 4.70 in Qwen
(3,584 columns, hidden 1024) and 4.95 in the wound (7,744 columns, hidden 2816). The null's mean
alignment equals the textbook value for evenly spread directions to within 0.1% in every layer,
which checks the arithmetic. The earlier estimate of "4.7 against 5.0" is confirmed.

**The registered "diffuse" bar could not have been met.** It asked for a ratio below 3.0; the
lowest first percentile of the null over all layers is 3.99 in Qwen and 4.27 in the wound. No
layer of either model could have been called diffuse, so "both focal" was not a finding of the
registered test.

### 3.2 W1 — within each model: focal in both

| | Qwen (24 layers) | Gemma wound (30 layers) |
|---|---|---|
| layers whose ratio is above the null at p ≤ 0.05 | **24** | **30** |
| layers whose *largest* alignment alone is above the null | 24 | 30 |
| layers whose *mean* alignment alone is above the null | 14 | 23 |
| saved ratio as a multiple of the null (geometric mean over layers) | 2.53 times | 1.87 times |
| excess E (natural log of that ratio), mean / median | 0.928 / 0.935 | 0.625 / 0.568 |
| spread of E from layer to layer (SD) | 0.256 | 0.227 |

In every layer the excess sits in the largest alignment; in 10 of Qwen's 24 layers and 7 of the
wound's 30 the average writer is no more aligned with the loop direction than with a random
direction. So "focal" means: a few columns, not the layer as a whole.

(Reference only, since neighbouring layers are not independent: a random direction would qualify
in 1 layer in 20; the probability of 24 or more of 24 is 6.0e-32 and of 30 or more of 30 is
9.3e-40.)

### 3.3 W2 — across the two models: they differ

| | Value | 95% interval |
|---|---|---|
| difference in mean E, Qwen minus wound | 0.303 in natural-log units | 0.172 to 0.428 |
| as a factor | **1.35** | 1.19 to 1.53 |
| layer labels shuffled, one layer at a time | **three of 100,000 shuffles reached the observed difference** (p = 0.00004; the smallest attainable is 0.00001) | |
| the same without the null subtraction (the registered C3 statistic) | 1.29 | 1.13 to 1.46 |
| standardised difference D (margin ± 0.5), 90% interval | 1.25 | 0.73 to 2.01 |

The equivalence test fails: the 90% interval of D lies wholly outside ±0.5. The difference test
also passes: the 95% interval of the difference excludes zero. Verdict by the registered rule:
**differ**. Taking off each model's last layer widens the gap (0.358; 0.251 to 0.466).

### 3.4 Dependence between layers (added in review; reported, not used for the registered verdict)

W2 treats layers as independent units, and neighbouring layers are not. The lag-1
autocorrelation of the excess along depth is +0.28 in Qwen (+0.37 without its last layer) and
+0.05 in the wound (−0.02 without its last layer). So the difference was recomputed with whole
blocks of consecutive layers as the unit:

| layers a block | Qwen blocks | wound blocks | 95% interval of the difference, layers resampled in blocks | 90% interval of the standardised difference | exact relabelling of whole blocks: relabellings reaching the observed difference | p |
|---|---|---|---|---|---|---|
| 3 | 8 | 10 | 0.146 to 0.459 | 0.68 to 2.12 | 91 of 43,758 | 0.0021 |
| 4 | 6 | 8 | 0.138 to 0.471 | 0.65 to 2.11 | 20 of 3,003 | 0.0067 |
| 6 | 4 | 5 | 0.216 to 0.390 | 0.91 to 1.78 | 1 of 126 | 0.0079 |

In every form the interval of the difference excludes zero and the standardised difference stays
above the 0.5 margin. Qwen's layer has the larger excess in 593 of the 720 pairs of one Qwen layer
and one wound layer. `redo.py` computes and stores these numbers (`Q3.layer_dependence`); they
agree with the review's own computation.

### 3.5 Qwen only — is the loop direction special among real state contrasts? (V2: special, with a qualification)

Five differences of mean states, scored with the same columns; median ratio over the 24 layers:

| Contrast | Median ratio | Times the null (4.70) |
|---|---|---|
| looping minus pre-onset (the loop contrast) | 12.66 | 2.69 |
| early minus late pre-onset positions of the same texts | 11.14 | 2.37 |
| terminating texts minus pre-onset | 7.22 | 1.54 |
| long non-looping minus pre-onset | 6.82 | 1.45 |
| long non-looping minus terminating | 7.75 | 1.65 |

The loop contrast has the largest ratio in **17 of 24 layers** (bar: at least 12; reference about
5; p = 1e-7). V2 passes.

The medians, which the protocol asked to be reported, add a qualification the bar does not
capture. Every real state difference is written more focally than a random direction. And a
within-text time contrast made only from positions before the loop is written almost as focally
as the loop contrast itself: of the loop contrast's excess over a random direction, about one
eighth (0.13 of 0.99 in log units) is added by the loop beyond that time contrast. This is a
reading of the reported medians, not a test.

Two things could explain it, and the saved data cannot tell them apart: state differences in
this model may be written focally whatever they mean; or the loop state may begin to build long
before the token-level onset (the four texts of section 1.4 whose state crossed more than 1,000
tokens early point that way).

### 3.6 What W2 does and does not settle

It settles the registered question: measured against random directions, the two loop directions
are not written with the same focality, also when layers are treated as dependent. Three things
it cannot settle:

- **The wound's loop directions were fitted on 5 texts, Qwen's on 11** (the probe fitted on the
  first half of the texts it accepted: 10 and 23). A noisier direction would look less focal.
  No saved wound state bounds this. In Qwen the same step (11 against 5 fitting texts) moves the
  excess by 0.02, and a direction has to be about 45° off to lose as much as the gap; the wound's
  texts would have to disagree with one another far more than Qwen's do. The held-out AUCs
  cannot detect this.

  The evidence, computed by `redo.py` on the saved Qwen states (80 texts, 32 looping and 32
  pre-onset positions each; five sets of 400 random draws of fitting texts):

  - Fitting on 11 texts gives a ratio 0.982 of the 80-text value; 5 texts give 0.961. In
    log-excess the step from 11 to 5 is 0.024 (standard error 0.0006; 0.022 to 0.026 across the
    five sets), about a thirteenth of the 0.303 gap.
  - The worst of 400 five-text draws loses 0.10 to 0.13 of excess depending on the set (0.096 to
    0.128) — between a third and 42% of the gap.
  - A direction has to be turned 44.8° from the 80-text direction before its ratio, median over
    layers, falls to the 74% the gap would need (a fall of 26%).
  - Qwen's five-text directions sit a median 17° from the 80-text direction (90th percentile
    29°); its eleven-text directions 11° (90th percentile 20°).
  - The held-out AUC is 0.972 with 3 fitting texts, 0.973 with 5, 0.974 with 11, and 0.975 for
    the 80-text direction on all 80. It does not move with the noise of the direction.

  The review's own single set of 400 draws gave 0.963 for five texts, a step of 0.022 and a worst
  draw of 0.098; the script's figures are within what one draw set to the next changes.
- **How focally the wound writes other state differences is unknown**: no wound activations were
  saved, so section 3.5 has no wound counterpart. Part of the gap could be a difference between
  the models in general, not between their loops.
- **The wound's ratios are taken as saved.** The directions themselves were not saved, so nothing
  on the wound's side can be re-derived; what is verified is the reading of the wound's weight
  file.

Two further facts about how the wound's directions were made belong here. Seven of their ten
texts have an onset under 500 tokens, so their pre-onset mean rests on windows shorter than 500
tokens (shorter still once the prompt is excluded), where every Qwen window is the full 500. And
the wound's directions come from forward passes with 8-bit quantised weights, while the columns
they are compared with come from the 16-bit weight file (`phase12/probe_atlas.py`).

## 4. Inventory: what is saved, and what it allows

| Part | Saved for Qwen | Saved for Gemma wound | Granularity | Computable without a forward pass |
|---|---|---|---|---|
| Loop dynamics | one 25-bin curve (`phase12/atlas_qwen.json`, 23 texts); per-text onsets (`phase11/logs/p12_qwen.log`); per-token projections onto the edit direction for 187 texts (`phase7/probe_out/timelines.npz`: 80 looping, 47 terminating, 60 long non-looping); residual vectors at 8,544 positions (`phase7/probe_out/resid.npz`) | one 25-bin curve (`phase12/atlas_gemma.json`, 10 texts); per-text onsets (`phase11/logs/p12_gemma.log`) | Qwen: texts and tokens. Gemma: one aggregate | Qwen: per-text tests. Both: curve description |
| Termination | per-problem rows and generations for stock, the ×1.5 row edit and unmatched controls (`phase3c/`, `phase8/`) | per-problem rows and generations for the wound and the editor passes (`phase11/`) | problems, paired | the paired test T2 |
| Writer geometry | per layer: mean, largest and ratio of alignment; stock weights in the repository | per layer: the same three numbers; the wound's weights on the Linux side (`~/wsp11/gemma4-8exp/model.safetensors`) | layers, 24 and 30 | the null from real columns (both); per-column values and other contrasts (Qwen only) |

The phase-7 activation files are on disk but git-ignored. On the Linux side there is no folder
or array file from phase 12: the Gemma loop directions, per-text trajectories and all per-text
activations were not saved. `phase7/probe_out` is the only place where per-text activations
exist, and it covers Qwen.

## 5. What a stronger test would need

Times are for one RTX 5090. Where a saved log has the figure it is marked "measured"; the rest
are estimates from arithmetic.

**Loop dynamics, across models.** Teacher-forced passes of the wound over its 490 capped saved
generations (about 8,200 tokens each) and over a non-looping control set (the full model's 500
saved generations, 499 of which terminate, mean 669 tokens, read by the wound), saving for every
token the projection onto each layer's loop direction. Estimated 20–40 minutes in all: about 72
trillion operations per text at 4.4 billion active parameters, one to three seconds each; the
phase-12 probe left no timing. The Qwen side needs under a minute for 500 texts. With per-text
lags and steps in both models the question becomes a two-sample comparison over texts with
shuffled labels. Saving each layer's mean state per class in the same passes would also give the
wound the check of section 3.5.

**Termination, specificity.** This needs generation, not forward passes. One practice pass (64
problems) per control, boosting one non-exit embedding row alone at the same factor: about 6
minutes a pass for the wound (generation took 1 min 44 s to 2 min 57 s in the saved editor logs,
plus engine start — measured) and about 2–3 minutes for Qwen (33–59 s generation in the saved
phase-8 logs — measured). Eight random rows at one dose is about an hour per model; a test that
can rank rows wants three doses and a second seed, so 3–5 hours per model. The norm-percentile
signature cannot be rescued by any run.

**Writer geometry.** The wound's other state contrasts need the same teacher-forced passes as
above (one per state class: loop, pre-onset, full-model terminating, long non-looping): the same
20–40 minutes. Any future run should also save the loop directions themselves, and run the
forward passes at the precision of the file the columns come from; the directions take no time
to keep.

## 6. Integrity log

- **Known in advance.** The aggregate inputs (both curves, all writer ratios, every cap-hit
  count, the norm percentiles, the earlier "4.7 against 5.0") were published in the repository
  and read before the protocol was written. The protocol is a registration of the tests' form,
  bars and analysis choices, not a blind prediction. What was new was every interval, every
  per-text and per-problem value, the nulls from real columns, and the Qwen contrasts.
- **No test was dropped or replaced.** Every test in the protocol was run as written.
- **One slip in the protocol's text.** Section 6.4 (b) of the protocol says the saved Qwen ratios
  were made with "12 other texts". `probe_atlas.py` fitted the Qwen direction on 11 texts, the
  first half of the 23 it accepted. The remark is descriptive and touches no test. The registered
  text is left exactly as it was registered, so its recorded hash still holds; the correct count
  is given in section 3.6.
- **Order of events before the review.** Protocol written 15:51; first result 16:04. Three bugs
  in the script were caught by its own self-tests before the first run (an exact-interval solver
  searching the wrong way, a moving average spanning 251 tokens instead of the registered 250, a
  syntax error). The first real run stopped at a consistency check that was too strict — it did
  not allow for the rounding of the saved mean and largest alignment to five decimals — after
  printing the loop-dynamics and termination sections. The check's tolerance was widened to
  exactly that rounding allowance; no definition changed.
- **Corrections after the review.** An independent reviewer reproduced the result file byte for
  byte and checked each test against the protocol, then asked for these corrections. All are
  made; none changes a registered statistic.
  1. **Input gate.** A validity record that was absent or null used to pass, and the five Gemma
     records were never judged. Every one of the ten generation-run inputs must now carry a
     validity record with an empty problems list, and the result file stores which checks in
     each record were judged (section 2.1).
  2. **Flags.** A cap flag or an accuracy flag that is not True or False stops the run
     (`np.array(..., bool)` would have turned the string "no" into True). So do a missing field,
     a `finish` that is neither stop nor length, a row that is not a problem record, and a file
     in which only some rows carry the re-scored accuracy field.
  3. **One door for every failure.** A missing input, an unparsable file or a row without a field
     gave a traceback and exit status 1. Every failure now exits with status 2, prints one line
     naming the cause, and leaves the result file saying that there is no current result. Pairs
     of arms are checked for holding the same problems before anything is computed.
  4. **Text ids beside the lags.** Section 1.4 had named m372 and m485 among the early crossings;
     the data give m373 and m488. The script now stores each text's id beside its lag and lists
     the early crossings itself.
  5. **The shuffle count.** Section 3.3 called 0.00004 "the smallest 100,000 shuffles can give".
     It is three of 100,000 shuffles reaching the observed difference; the smallest is 0.00001.
     The script prints and stores the count.
  6. **Resamples: 10,000 → 1,000,000 for every interval.** This departs from the registered
     settings; the percentile method and the confidence levels are as registered. With 10,000
     the lag interval of layer 19 moved from one seed to the next (lower limit 102 or 120
     tokens, upper 203 or 226), because its 2.5% and 97.5% levels lie within one standard error
     of the cumulative mass at lag values 102 and 203. Every interval now stores how far its
     limits move over three other seeds: the lag intervals not at all, the others by at most
     0.02 points for the cap-hit reduction, 0.002 for the standardised difference and 0.00021 for
     A. Limits that moved with the change: Qwen's cap-hit reduction 50.5–59.7 → 50.6–59.8; the
     standardised difference 0.74–2.02 → 0.73–2.01; the difference 0.172–0.426 → 0.172–0.428;
     two A intervals by 0.001.
  7. **Windows and log lines.** The looping-text windows are checked to fit their text; an onset
     line that is malformed stops the run instead of being skipped; so does a trajectory value
     that is not a finite number.
  8. **Two sections added in review**, marked as such in the script and the result file, not
     pre-registered and not used for any verdict: dependence between layers (section 3.4) and
     the number of fitting texts (section 3.6). The script's numbers are its own; where the
     review's differ slightly, both are given.
- **Self-tests.** They run at the start of every run. The exact helpers are checked on known
  values; the AUC and resampling helpers against their definitions; block resampling against
  its definition. The input gate is exercised on scratch copies of the inputs, in a temporary
  folder outside the repository: eleven faults (a validity record that lists a problem, is
  absent, is null or has no problems list; a cap flag or an accuracy flag that is a string; a
  missing field; a missing re-score stamp; a missing file; a file that is not JSON; an input
  needed after the gate) are each sent through the whole entry point, which must exit with
  status 2 and mark the earlier result as superseded. When an input is itself missing or
  invalid, this part stands aside and says so, and the run's own gate reports the input.
- **Failure suite.** A separate suite of 44 cases, modelled on the reviewer's, was run on a
  scratch copy of the inputs and of the script outside the repository. Every fault case exited
  with status 2, printed no traceback, named its cause and left no result; the one case with
  nothing wrong ran to a complete result with the same verdicts. The suite found four faults in
  the first version of these corrections, since fixed: the in-script self-test misfired when an
  input was already faulty; a missing input made it stumble before the gate could speak;
  mismatched pairs were noticed only after the loop tests had run; and a value that is not a
  number passed silently into the result.
- **Registered statistics unchanged.** Compared value by value with the result before the
  review: of 1,257 values, 1,219 are identical, and every registered statistic is among them — A,
  the median lags and sign tests, the cap-hit reductions and McNemar tests, the layer counts, the
  difference and the standardised difference, the two Qwen-only checks; so are the two
  permutation probabilities. What changed is the hashes and accuracy figures of the ten
  generation-run inputs, and interval limits in their last digit or two (item 6).
- **Second code path.** Before the review, an independent script (another tensor reader,
  brute-force loops instead of vectorised code) had reproduced the cap-hit counts and exact
  tests, the half-way crossing, all 79 per-text lags at both layers, the nulls for one layer of
  each model, the embedding-row percentiles, the contrast ratios at layer 19 and the arithmetic
  of W2.
- **Concurrent work in the repository.** Another session replaced the answer checker and
  re-scored every saved run while this was revised (the checker stamp in the files read changed
  twice). None of those files is touched here. Every verdict rests on cap-hit and finish fields,
  which re-scoring does not change. The accuracy figures are those in the files at the last run,
  labelled with the checker stamp; `redo.json` records a hash of each input.
- **Numbers in this report that do not come from `redo.py`**, although the protocol says every
  number does:
  - the saved-generation counts 490, 500, 499 and the mean length 669 tokens (section 5): the
    summaries of `phase11/a1_wound_final.json` (500 problems, 98.0% capped, so 490) and
    `phase11/a0_full_final.json` (500 problems, 0.2% capped, so 499 end on their own; mean 669
    tokens), as listed in `phase11/results.md`;
  - the practice-pass timings (section 5): the progress lines of `phase11/editor/ed_pass1.log`
    and `ed_pass8.log` (2 min 57 s and 1 min 44 s) and of the phase-8 practice logs
    (`c1_pass_b*.log`, 33–59 s); "about 6 minutes with engine start" is the editor's own note
    in `phase11/editor/journal.md`; the 20–40 minutes for teacher-forced passes is an estimate
    from arithmetic;
  - the cosine 0.71–0.74 (section 1.3): `phase7/ATLAS.md`, Finding 5;
  - the 8,544 saved positions (section 4) and "about 8,200 tokens each" (section 5): the shape of
    `phase7/probe_out/resid.npz`, and the 8,192-token cap under which the wound's final was
    generated (`phase11/results.md`);
  - 8-bit forward passes against 16-bit columns (section 3.6): `phase12/probe_atlas.py`;
  - the smallest row count for the admission-order test, 100 (section 2.1):
    `phase3c/runcheck.py`;
  - how far the lag interval moved across seeds at 10,000 resamples (item 6): measured during
    the revision with a scratch script outside the repository;
  - the review's own figures where they are quoted beside the script's (section 3.6), and the
    count of 1,257 and 1,219 values (item "Registered statistics unchanged");
  - the running time of the script.

  The problem id "m48" is printed by the script. Every other number in sections 1 to 3 comes from
  `redo.py` and is stored in `redo.json`.
- **Reproduce:** `<repo>/.venv/Scripts/python.exe -B phase12/redo.py <repo> --gemma-weights
  <the wound's model.safetensors>` — about 130 seconds; prints every number and rewrites
  `phase12/redo.json`. (An optional `--qwen-weights` names another copy of the stock weight file.)
  A run that fails anywhere exits with status 2 and leaves that file saying that there is no
  current result.
