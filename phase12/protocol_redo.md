# Pre-registered protocol (redo) — Phase 12: what the saved data can and cannot test

**Date:** 2026-11-26. Written before any statistic in this file was computed. The script that
computes them is `phase12/redo.py`; it records the SHA-256 of this file in `phase12/redo.json`,
so the text that was registered can be checked against the text that was applied. Results go to
`phase12/results_redo.md`. This file replaces nothing: `phase12/protocol.md` and
`phase12/results.md` stay as the registered text and the withdrawn verdict (ISSUES #21), and
`probe_atlas.py` and `correspondence.py` are neither re-run nor modified. Nothing is committed.

## 1. The question, and the limits of this attempt

The phase-12 question, unchanged: do Qwen3.5-0.8B (dense, 24 layers, hidden 1024) and the pruned
Gemma-4 26B-A4B "wound" (MoE, 30 layers, hidden 2816) write the same functions with corresponding
number signatures? Both repeat themselves and both fail to stop. Three questions: loop dynamics
(C1), termination (C2), writer geometry (C3).

Limits set by the user: no GPU work — no model load that needs the GPU, no generation, no new
forward pass. Allowed: data already saved, plus arithmetic on weight files on the CPU, using only
files in the repository (Qwen stock weights, `phase3/models/stock/`) and the Gemma wound weight
file on the Linux side (`~/wsp11/gemma4-8exp/model.safetensors`, passed to the script as an
argument and never written into any file).

**Disclosed before computing.** This is a pre-registration of the tests' form, bars and analysis
choices, not a blind prediction of their outcome. The aggregate numbers of every input below are
already published in this repository and were read while designing the tests: the two 25-bin
trajectories, the per-layer writer ratios, the cap-hit rates of every generation run used, the
embedding-row norm percentiles, and the earlier estimate that random directions give ratios of
about 4.7 (Qwen) and 5.0 (Gemma). Nothing below was chosen to produce a particular verdict, but a
reader should treat a verdict that agrees with what those aggregates suggest as expected, and a
verdict that disagrees as informative. What is genuinely not yet known is every interval, every
per-text or per-problem quantity, the null distributions computed from the real columns, and the
Qwen-only contrasts in section 6.4.

**Why the registered tests failed** (so that the redo does not repeat them):

- C1 (loop dynamics, r = 0.96). Each curve is the projection onto a direction defined as
  mean(loop states) minus mean(pre-onset states), z-scored by the loop window's own mean and
  spread. "Low before the onset, zero in the loop" is therefore built into both curves, and two
  smooth step curves correlate whatever the models do. Each model has exactly one curve (an
  average over texts and layers), so there is no per-text value to pair, no interval, and the
  "1,000 circular shifts" have only 24 distinct values.
- C2 (termination). The behavioural half ("a row boost cuts cap-hits") was already on record from
  phases 4-5 and 11 and had no null: no saved run boosts a non-exit row. The statistical half
  ("any of three Gemma candidate rows falls in the Qwen row's norm tercile") passes 70% of the
  time by chance, and two of the three candidates sit at the 97.8th and 99.4th percentile.
- C3 (writer geometry). "Diffuse" was defined as a ratio below 3.0 — an absolute threshold that
  ignores layer width and writer count. The earlier analysis reports that a random direction
  already gives about 4.7 and 5.0, which would put the bar below the null and make "both focal" an
  outcome no model could avoid; the null computed here checks that directly. The direction of the
  difference between the two models was also misstated.

## 2. Inventory in brief

The full inventory is printed by the analysis session and repeated in `results_redo.md`.

| Question | Saved for Qwen | Saved for Gemma | Granularity |
|---|---|---|---|
| Loop dynamics | one 25-bin trajectory (`phase12/atlas_qwen.json`, 23 texts); per-text onsets (`phase11/logs/p12_qwen.log`); **per-token** projections onto the edit direction u15/u19 for 187 texts (`phase7/probe_out/timelines.npz`: 80 looping, 47 terminating, 60 long non-looping); residual vectors at 8,544 sampled positions (`phase7/probe_out/resid.npz`) | one 25-bin trajectory (`phase12/atlas_gemma.json`, 10 texts); per-text onsets (`phase11/logs/p12_gemma.log`) | Qwen: texts and tokens. Gemma: one aggregate per model. |
| Termination | per-problem rows and saved generations for stock, the ×1.5 single-row edit, and unmatched controls (`phase3c/`, `phase8/`) | per-problem rows and saved generations for the wound and the editor passes (`phase11/`) | problems, paired across arms by problem id |
| Writer geometry | per layer: mean, max, ratio of alignment onto the loop direction; stock weights | per layer: mean, max, ratio; wound weights | layers (24 and 30); the directions themselves and per-column values are not saved |

On the Linux side there is no folder or array file from phase 12: the Gemma loop directions, the
Gemma per-text trajectories and all per-text Gemma activations were not saved.

## 3. Fixed settings

Seed 20261126 for every random draw. Bootstrap: 10,000 resamples, percentile intervals, 95% unless
a test says 90%. Permutations: 100,000 label shuffles. Null directions for section 6: 4,000 per
layer per model, each a Gaussian vector normalised to length 1. Exact tests (binomial, sign,
McNemar) are computed in closed form. No threshold, window, arm or margin below may change after
a number is seen; if a test turns out to be invalid after all, it is dropped, not replaced, and
the reason is recorded in `results_redo.md`.

## 4. Q1 — Loop dynamics

**Question.** As a generation slides into its repetition loop, does the model's internal state
follow the same time course in the two models?

### 4.1 Across the two models: not testable from saved data

Because: each model has one aggregate curve with no per-text values, so no interval exists for
any Gemma quantity and no pairing by text is possible; and the two curves share their shape by
construction (section 1). A correlation between them is not reported as evidence. The forward
passes that would make the question testable are listed in `results_redo.md`.

### 4.2 What each saved trajectory shows on its own (description, no bar)

For each model's 25-bin trajectory (bin width 40 tokens; bin 12 starts at the token-level onset):

- contributing texts per bin, reconstructed from the logged per-text onsets (how many texts
  cover the bin fully, how many partly); for Gemma this exposes the bins that contain chat-prompt
  tokens.
- the half-crossing point: pre level P = mean of bins 8-11 (the four bins before the onset bin),
  plateau L = mean of bins 21-24 (the last four), half level H = P + (L − P)/2. The crossing is
  the first bin b ≥ 8 with value ≥ H, refined by straight-line interpolation between the centres
  of bins b − 1 and b. Reported in bins and in tokens relative to the onset (bin centre of bin 12
  is at +20 tokens), together with the step size L − P.

These numbers describe two curves. They are not a comparison with a null, they carry no interval,
and they cannot fail.

### 4.3 Qwen only — does the state step up at the onset, more than in texts that do not loop? (test Q1-a)

Data: `phase7/probe_out/timelines.npz`, the per-token projection of the state after layer 19
onto the edit direction u19 (`proj19_<id>_<group>`; layer 15, `proj15`, as a replication), with the
looping onsets of `phase7/sets.json` (a different detector from phase 12: the first 250-token bin
in which more than half of all 12-token windows have occurred before, sustained for four bins).

For a timeline x(t) and an anchor position c (tokens into the generation):
step(c) = mean(x[c+1000 : c+1500]) − mean(x[c−2000 : c−1500]).

- Looping texts (80): c = the text's own onset.
- Control, long non-looping texts (60; the "deep" set: the edited model's generations of at least
  9,000 thinking tokens with no onset, read by the stock model): c = every looping onset that fits
  (c + 1500 ≤ length), averaged within the text, so each control text gives one value.
- Second control, the stock model's own terminating texts (47; those at least 3,500 thinking
  tokens long can host an anchor; the count that can is reported): the same.

Statistic: A = the probability that a random looping text's step exceeds a random control text's
step (ties count one half). Unit: texts (80 against 60). Null: the two groups are exchangeable —
100,000 shuffles of the group labels. Interval: percentile bootstrap over texts within each group.

Bar (the step is specific to looping texts): A ≥ 0.80 **and** one-sided permutation p < 0.001
against the long non-looping control. The second control is reported and does not enter the bar;
if it disagrees in direction, the result is reported with that qualification.

### 4.4 Qwen only — does the state lag the token-level onset? (test Q1-b)

Looping texts only, layer 19 (layer 15 as replication). With the anchor at the onset o: base and
plateau as in 4.3, level ℓ = base + step/2, y(t) = the mean of x over the 250 tokens centred on t
(truncated at the ends of the text). t* is the first t in [o−1500, o+1499] with y(t) ≥ ℓ; the lag
is t* − o. A text with step ≤ 0, with no crossing inside that window, or whose smoothed series is
already at or above ℓ when the window opens (so the crossing happened earlier than the window can
see) has no lag and is counted.

Unit: texts. Report the median lag with a 95% bootstrap interval over texts and the exact
two-sided sign test (lags above zero against lags below zero).

Bar: "lags" if the whole 95% interval of the median is above 0; "leads" if it is wholly below 0;
otherwise "not distinguishable from zero".

## 5. Q2 — Termination

**Question.** Is the failure to stop the same function in the two models, with the same number
signature?

### 5.1 Test T2 — boosting only the exit rows reduces cap-hits, in both models

The one termination claim the saved data can test with a real null (no change). Per model, paired
by problem id, capped = the generation ran into the token cap.

| Model | Unedited | Exit rows only | Problems |
|---|---|---|---|
| Qwen | `phase3c/p3c_stock_rescored.json` | `phase3c/p4_mimo_rescored.json`: row 248069 (`</think>`) × 1.50 | 500 |
| Gemma | `phase11/ed_pass1.json` | `phase11/ed_pass8.json`: the three stop-token rows 1, 50, 106 × 1.6 (editor attempt 5, pass 8; recipe c11) | 64 |

The Gemma arm registered in phase 12 (attempt 1, pass 2: stop rows × 1.5 together with boosts of
four other row groups) is reported beside it but does not carry the verdict, because it does not
isolate the exit rows.

Statistic per model: R = (capped before − capped after) / capped before. Null: no change in the
per-problem outcome — exact McNemar test (the recovered and the lost problems are equally likely
under the null). Unit: problems. Interval: 95% percentile bootstrap over problems, resampled
jointly for the two arms.

Bar, per model: exact two-sided McNemar p < 0.05 **and** R ≥ 0.25 (the relative reduction phase 12
registered). T2 passes only if both models pass.

Reported, none of it part of the bar:

- R against Qwen's second stock run (`phase3c/ref_stock_rescored.json`), a check that the pairing
  is not an accident of one run.
- accuracy change with its paired bootstrap interval (final answer checker in the files).
- how the boosted models stop: the share of the boosted arm's terminated generations that end
  within 32 generated tokens, with an exact 95% interval. This says whether "termination
  recovered" means "the model is pushed out at once".
- the unmatched controls that exist, as counts only: Qwen random noise of matched size on the same
  row (`phase3c/p4_c2_random_rescored.json`), the same row scaled on the input side only
  (`phase8/d_in.json`); Gemma final norm × 0.4 with no exit row (`phase11/ed_pass9.json`).

Every input must pass its own validity record (an empty `validity.problems`) and the two arms of a
pair must contain the same problem ids; otherwise the script stops without a verdict.

### 5.2 Not testable from saved data

- **Specificity — is the exit row special?** The claim "the exit row matters" needs a run in which
  a non-exit embedding row alone is boosted by the same factor. No such run is saved for either
  model. This needs generation, not forward passes.
- **A geometric signature of the exit row (norm percentile).** The "low norm" rule was written
  after the four percentiles had been looked at, it offers three Gemma candidates of which two sit
  at the opposite extreme, and one row per model cannot be rescued by any interval. Re-testing it
  on the same four numbers would be circular, so no p-value is computed for it.
- **"No accuracy loss" on 64 problems.** One problem is 1.6 points there, so the data cannot
  certify a loss smaller than that in either direction; accuracy is reported, not tested.

### 5.3 What a pass of T2 would and would not mean

It would mean: in both models the failure to stop is relieved by making the stop token's output
row larger. It would not mean the two failures are the same function. Raising a stop token's
output row makes any sampling model stop sooner, healthy or not; T2 is a necessary condition for
correspondence, not evidence of it.

## 6. Q3 — Writer geometry

**Question.** Do the two models write their loop direction in the same way — spread over many
writers, or concentrated on a few?

The saved statistic, per layer, is the ratio R = (largest |cosine| between a writer column and the
loop direction) / (mean |cosine| over all writer columns). Qwen: 24 layers × 3,584 MLP columns.
Gemma: 30 layers × 7,744 columns (2,112 dense-MLP columns and the 8 kept experts' 704 columns
each). The loop directions themselves were not saved, so only mean, max and ratio can be compared,
and nothing finer.

### 6.1 The null (both models)

For each layer, 4,000 random unit directions in the layer's hidden space, applied to the layer's
real columns from the weight file (each column normalised to length 1; Gemma: dense columns
followed by the experts' columns, experts transposed so that each column is a writer, as in
`probe_atlas.py`). For every direction: its mean alignment, its maximum alignment and its ratio.
The null therefore has the right layer width, the right number of writers and the real columns'
shape. Reported for each model: the null's mean alignment against the textbook value for
directions spread evenly in all dimensions, sqrt(2 / (π × hidden)), as a check on the arithmetic.

### 6.2 Test W1 — within each model: is the loop direction written more focally than a random direction? (layers as units)

Per layer: p = (1 + number of null directions with ratio ≥ the saved ratio) / (4,001). The saved
ratio is the saved max divided by the saved mean.

Unit: layers (24 and 30). Statistic: the number of layers with p ≤ 0.05. Reference: if the loop
direction were like a random direction in every layer, 1 layer in 20 would qualify. Report the
count, the exact one-sided binomial probability of a count at least that large (a reference only:
neighbouring layers are not independent), and the same counts for the mean alignment alone and for
the maximum alone (which says whether any excess sits in the average writer or in a few).

Bar: "focal" if at least 90% of layers qualify (22 of 24; 27 of 30) and that binomial probability
is below 0.001; otherwise "not shown focal".

Also reported: where the registered "diffuse" bar of 3.0 lies relative to the null (the lowest
1st percentile over layers, per model). If 3.0 is below it, the registered test could not have
classed any model as diffuse.

### 6.3 Test W2 — across the two models: is the degree of focality the same? (layers as units)

Per layer: E = ln(saved ratio) − mean over the null directions of ln(null ratio); the excess over
the layer's own null, on the natural-logarithm scale (E = 0.69 means a ratio 2 times the null).

Unit: layers. Statistic: Δ = mean(E over Qwen layers) − mean(E over Gemma layers). Interval:
95% percentile bootstrap over layers within each model. Null of no difference: 100,000 shuffles of
the layer labels between the two models (two-sided p). Reported beside it: the same comparison
without the null subtraction, which is the registered C3 statistic.

Equivalence, with the margin fixed now: s = the pooled standard deviation of E across layers of
the two models (the layer-to-layer variation inside a model); D = Δ / s. The two models are
declared to correspond if the **90%** bootstrap interval of D lies entirely inside (−0.5, +0.5) —
the difference between the models is smaller than half the variation from layer to layer within a
model. The margin 0.5 is the usual boundary between a "small" and a "medium" standardised
difference.

Verdict W2: **correspond** if W1 says "focal" for both models and the equivalence holds;
**differ** if the 95% interval of Δ excludes zero; otherwise **inconclusive**. Sensitivity,
reported and not used for the verdict: the same with the last layer of each model removed.

### 6.4 Qwen only — checks and a second null (the saved Qwen activations allow them; nothing equivalent exists for Gemma)

**(a) Reader check, both models.** Recompute the norm percentile of the saved exit rows from the
weight files (rank of the row among all embedding rows, divided by rows − 1) and compare with the
saved values (Qwen row 248069: 0.1681; Gemma rows 1, 106, 50: 0.9937, 0.0756, 0.9783). Pass if
every value agrees within 0.002. On failure the script stops without a W1 or W2 verdict: the
Gemma numbers could not be trusted.

**(b) Alignment check, Qwen.** Re-derive the loop direction of each layer from
`phase7/probe_out/resid.npz` (mean looping state minus mean pre-onset state at that layer's
output, all 80 looping texts), compute the full set of column alignments with the stock weights,
and compare the 24 ratios with the 24 saved ones (which used another estimate of the direction:
12 other texts, another onset detector). Pass if, over the 24 layers, the median absolute relative
difference is at most 25% for the ratio and at most 40% for the mean alignment, and the Pearson
correlation between saved and re-derived ratios is at least 0.8. This validates the column
extraction and the cosine arithmetic that the Gemma null shares.

**(c) Is the loop direction special among real state contrasts? (test V2, Qwen only).** The null of
6.1 uses random directions. Real state differences may be written focally whatever they
represent. Five contrasts, each a difference of mean states at every layer's output, each scored
with the same columns as in (b): (1) looping minus pre-onset, the loop contrast; (2) early minus
late pre-onset positions of the same texts (a time contrast inside the same texts, before any
loop); (3) terminating texts minus pre-onset; (4) long non-looping texts minus pre-onset;
(5) long non-looping minus terminating.

Unit: layers (24). Statistic: the number of layers in which the loop contrast has the largest
ratio of the five. Reference: if the loop contrast were not special, it would be largest in one
layer in five. Bar: "special" if it is largest in at least 12 of 24 layers and the exact one-sided
binomial probability of that count (at 0.2) is below 0.001; otherwise "not distinguishable from
other state contrasts". Also reported: the median ratio over layers for each of the five contrasts.
If V2 fails, the "focal" label of W1 is a property of state differences in general in Qwen, not of
the loop, and the report must say so.

## 7. Verdict rule (fixed now)

Correspondence on a question is established only by that question's cross-model test passing its
bar.

- Q1: no cross-model test exists. Not established, whatever 4.2-4.4 show. 4.3 and 4.4 are findings
  about Qwen alone.
- Q2: T2 is a necessary condition. Not established, whatever T2 shows.
- Q3: established only if W2 says "correspond".

The report then states in one sentence how much of the phase-12 question the saved data can
answer, and for each question what a stronger test would need: which forward passes or generation
runs, and roughly how long on one RTX 5090.

## 8. Rules

No statistic in this file may be redefined after the first result is seen. A test found invalid
is dropped and recorded, not replaced. Every number reported in `results_redo.md` is printed by
`redo.py` and stored in `redo.json`, which also lists its input files (by path relative to the
repository) and settings. No user name, machine path or host name is written into any repository
file.
