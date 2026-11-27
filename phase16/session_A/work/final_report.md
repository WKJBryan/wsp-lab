# Final report: Phase 14 weight edit (Qwen3.5-0.8B, MATH practice set)

## Summary in plain words

**What was wrong.** The model is not bad at the first step of these problems. It works them out
early and usually writes down an answer within its first few hundred to two thousand tokens. What
it cannot do is stop. At the end of every line of its thinking it decides between "end the
thinking" and "start the next line", and it essentially always starts the next line. The thinking
turns into endless "Wait, let me double-check" cycles, then into one sentence repeated thousands of
times, and the generation is cut off at the token limit with no final answer. That happened on 121
of the 128 practice problems.

**What I changed.** One weight matrix, in a targeted way. About half of the network's "end the
thinking" signal at the end of a line comes from 69 particular neurons in the very last layer (the
rest is spread over layers 14 to 22), and those 69 neurons fire at the end of a line and almost
nowhere else. I multiplied what those 69 neurons write toward "end the thinking" by 2.1. Nothing
else in the model was touched.

**What it did.** On the practice problems, 37 of 128 are now answered correctly, against 7 of 128
for the unmodified model. The thinking now ends on 80 of 128 problems instead of 7. No problem that
was solved before was lost.

**What is still wrong.** On 40 of those 80 problems the model then ends its thinking, starts
writing the answer, and the *answer* never finishes either: it falls into the same "Wait" loops and
never writes the final boxed value. On 48 problems the thinking still never ends (almost all hard,
Level 4-5, problems). I looked for a targeted repair for the answer phase and did not find one that
is content-dependent (details below), so I left it alone rather than force the model to stop
regardless of whether it is right.

**Prediction for the held-out set (MATH-500).** About 29 of 100 problems correct, with a plausible
range of about 25 to 32, against roughly 4 to 6 of 100 for the unmodified model.

**Decision left to the user:** whether the remaining answer-phase failure should be attacked with a
content-independent "answer budget" (which I did not do, because the task rules it out) or left.

---

## 1. The exact final edit

Installed in `~/wsp14/sandbox/model/` (accepted by `save_edit.py`, fingerprint `cd55253ac536fe76`,
which is the fingerprint measured in passes 2 and 4).

- Tensor: `model.language_model.layers.23.mlp.down_proj.weight` (the output matrix of the
  feed-forward block of layer 23, the last layer; shape 1024 x 3584, bfloat16; one column per
  neuron).
- Direction: `u = (1 + norm.weight) * E[248069]`, where `norm.weight` is
  `model.language_model.norm.weight`, `E` is `model.language_model.embed_tokens.weight` (tied to
  the output layer) and 248069 is the `</think>` token. `u` is the direction along which the final
  normalised hidden state is read out for `</think>`. `|u| = 2.4874`; `uhat = u / |u|`.
- Neurons S (69 columns), selected by one rule from activation statistics of the unmodified model
  (mean contribution to the `</think>` logit at line ends inside the thinking above 0.03 nats):
  `2, 7, 18, 27, 41, 42, 43, 46, 50, 57, 60, 65, 66, 68, 69, 70, 71, 73, 78, 92, 101, 104, 107,
  113, 115, 124, 137, 142, 144, 145, 154, 156, 163, 164, 168, 170, 175, 178, 181, 205, 211, 218,
  234, 259, 288, 299, 317, 336, 365, 382, 384, 392, 422, 426, 439, 534, 576, 666, 705, 847, 872,
  939, 1077, 1331, 1473, 1821, 1925, 2025, 2220`.
- Operation, in float32 then rounded back to bfloat16, for every column `d_j` with `j` in S:
  `d_j  <-  d_j + 1.1 * (d_j . uhat) * uhat`.
  This multiplies by 2.1 the part of each of these neurons' output vector that points along the
  `</think>` read-out direction and leaves every other direction of every neuron unchanged.
- Magnitudes: 68,246 of 3,670,016 values changed (all inside the 69 columns); largest change 0.007153;
  size of the change 0.1595 against 16.42 for the whole matrix; the summed `</think>` alignment of
  the 69 columns goes from 0.0303 to 0.0636 (ratio 2.101). All other 487 tensors and all 28 other
  files of the model directory are byte-identical to the pristine base.
- Effect at run time (measured by teacher forcing, i.e. running the network on fixed texts): the
  69 neurons raise the `</think>` logit by 6.33 nats at the end of a line inside the thinking and
  by much less elsewhere, so the added amount (1.1 times their output) is about +7.0 nats at line
  ends, +2.1 in the answer after `</think>`, +1.9 in the middle of a line and +0.5 at the very
  first generated token. At the baseline's own closing points the measured rise is +5.5 to +8.9
  nats. The log-probability the model assigns to the token it actually wrote changes by -0.0005 on
  average away from line ends, i.e. its reasoning is not changed.

Why this is a repair and not a global knob or a forced stop: the neurons, the layer and the
direction were located by causal attribution; the gain is applied only to an existing pathway and
only where that pathway already acts; the outcome depends on content because the model's own weak
closing signal decides at which line ends the closing wins. Evidence for that in section 2.6.

## 2. Diagnosis, and the evidence for it (in the order I found it)

### 2.1 The weights themselves look intact
Per-tensor statistics (mean, spread, kurtosis, row and column norms), per-head norms, dead rows,
and pairing checks (gate/up/down rows of the feed-forward blocks move together in every layer; no
layer is a copy of another) show no anomalous tensor. The failure is functional.

### 2.2 What the failure looks like (pass 1, unmodified model)
- 7 of 128 correct, 7 of 128 close their thinking, 121 of 128 are cut at the 32,768-token cap.
- 118 of those 121 end in a verbatim loop: one sentence (median length 37 tokens, shortest 6,
  longest 539) repeated hundreds to thousands of times (one problem: 5,128 times). The loop starts
  after a median of about 7,200 tokens and fills about 80% of the text.
- Before the loop the model has usually already solved the problem: in 66 of the 121 cut-off
  generations the correct answer appears as a plain string within the first 2,000 tokens (a rough
  proxy: a substring match; first mention at a median of about 530 tokens). It then doubts ("Wait,
  is it possible that the question meant ...", "Maybe there is a typo"), the doubts shrink into
  repetition, and it never ends.
- The 7 that closed did so right after a concluding sentence ("Final Answer: 2/3.", "I'll write
  out the solution now."), after 1,700 to 16,600 tokens.
- This matches the warning in the model card ("more prone to entering thinking loops ... which may
  prevent it from terminating generation properly").

### 2.3 Where the decision is taken, and how one-sided it is
Teacher-forced reads of all 128 baseline generations (re-tokenising the texts gives the harness's
token count exactly in 110 generations and a count off by one token at the very end in 18, so the
reads are exact):
- The decision is taken at one place: the token right after a line break. The candidates there are
  `</think>` and the indentation token that continues the model's structured format.
- At ordinary line ends `</think>` has logit about 14.5 and the continuation about 30: a margin of
  about 14.5 nats. After "So the answer is X." the margin is still about 8. Only after "I'll write
  the response now"-type lines is it about 2.5.
- The margin does not shrink with time. `</think>` at line ends: 14.6 before token 300, 14.4 after
  token 7000; the gap even widens from -15.0 to -16.3 because the continuation grows stronger.
  Within one generation the correlation between position and the `</think>` logit is +0.11. There
  is no "enough thinking" signal.
- In 116 of the 121 cut-off generations the sampling probability of `</think>` never exceeds 0.01
  at any position. Moments at which `</think>` comes within 6 nats of the top token occur in 14 of
  128 generations, first at a median of about 4,500 tokens.
- In a short, plainly formatted text the model closes eagerly at almost any line end (probability
  0.48 to 1.00, mostly above 0.85, even after "So the next step is to divide by 2."): its own
  structured, indented format is what keeps it thinking at all, and that format is self-sustaining.

### 2.4 Which components write the closing decision (exact logit attribution)
The logit of a token is linear in the final residual stream, so each block's contribution can be
computed exactly (full forward passes with recorders; the parts add up to the measured logit to
within 0.05 nats at all 287 positions). Mean contributions to the `</think>` logit (nats) at the 7
real closing points, and for comparison at 140 line ends of the same generations at which the model
went on thinking after a concluding sentence, and at 140 other line ends:

| where | closing points (7) | went on after a conclusion (140) | went on, other line ends (140) |
|---|---|---|---|
| layers 0-13, all blocks together | +1.55 (no block above 0.5) | +1.21 | +1.10 |
| layers 14-21, all blocks together | +10.62 | +6.91 | +6.31 |
| layer 22, DeltaNet block | +3.30, of which head 13: +2.81 | +1.00, head 13: +0.91 | +1.06, head 13: +0.99 |
| layer 22, feed-forward block | +0.96 | +0.57 | +0.67 |
| layer 23, attention block | +0.16, head 6 alone: +1.08 | +0.51 | +0.56 |
| layer 23, feed-forward block (MLP) | +8.28 | +6.65 | +6.74 |
| total | +24.87 | +16.85 | +16.45 |

- Layers 0 to 13 contribute about +1.2 to +1.6 in total, no single block more than 0.5 nats, and the
  same at closing points as elsewhere: this rules out the early and middle layers as the place where
  the decision is made.
- The feed-forward block of layer 23 is the largest single writer (+6.7 to +8.3 at every line end).
- Gated DeltaNet head 13 of layer 22 is the one component that carries content information: +2.81
  at closing points against +0.91 when the model goes on. Attention head 6 of layer 23 contributes
  about +1.1 everywhere (not specific to closing).
- Inside the layer-23 MLP the closing output is made by 69 specific neurons that fire at line ends
  inside the thinking and are nearly silent elsewhere: their sum raises the `</think>` logit by
  +6.33 at line ends (spread 1.6, stable from token 3 to token 5000), by +1.9 in the answer after
  `</think>`, +1.75 in the middle of a line, +1.7 on the problem text and +0.5 at the first
  generated token.

### 2.5 What the network does not represent, and why the loops cannot be repaired locally
- A linear classifier separating line ends that state a conclusion from line ends in the middle of
  the reasoning reaches only 61 to 62 percent (5-fold, split by generation), at every layer. The
  network barely encodes "the answer has been found". Position in the generation is well encoded
  (80 to 88 percent).
- The verbatim loops are sustained by a few specific early heads. At a position inside a loop I
  removed every block and every head in turn and carried the change through all downstream layers
  (40 loops, each with 4 ordinary positions of the same generation as a control). Most heads do
  nothing there (median change over all 336 heads: 0.00 nats). Removing DeltaNet head 8 of layer 0
  lowers the loop token by 9.5 nats on average and erases the loop's margin outright in 13 of the
  40 loops; layer-0 head 3 (-4.5), attention head 7 of layer 3 (-4.5), layer-1 head 7 (-3.6) and
  layer-22 head 9 (-3.1) follow.
- But those heads cannot be attenuated in the weights. At the single loop position the trade looks
  good (four heads at 0.35: loop margin 11.7 -> 4.2 nats for 0.05 nats on the predicted token at
  the control positions). Written into the weights the same change acts at every position and so
  changes the keys, values and recurrent states of the whole context: in a real forward pass on 16
  baseline generations the log-probability of the tokens the model actually writes falls by 0.51
  nats on average (the closing-gain edit alone: 0.0005). These are core context-processing heads. I
  rejected that candidate on this measurement, before spending a pass.
  So the loop has a local cause but no local repair that leaves the model intact; the repair has to
  happen earlier, by ending the thinking before the collapse.

Conclusion: the termination pathway (the 69 line-end neurons of layer 23, with Gated DeltaNet head
13 of layer 22) is far too weak relative to the format-continuation token.

### 2.6 Choosing the gain, and the causal checks
- Simulation on the baseline trajectories with the exact sampling rules (temperature, top-k, top-p,
  presence penalty), validated at gain 1 where it reproduces the measured baseline (6.9 percent
  closed, 5.9 percent "answer already stated" against 5.5 measured):

  | gain | thinking ends | answer already stated when it ends | ends before the answer appears |
  |---|---|---|---|
  | 1.0 | 6.9% | 5.9% | 0.9% |
  | 1.75 | 17.3% | 13.9% | 3.4% |
  | 2.0 | 35.7% | 24.0% | 11.7% |
  | 2.1 | 50.8% | 31.3% | 19.5% (4.6% before token 400) |
  | 2.2 | 63.8% | 33.4% | 30.4% |
  | 2.5 | 84.2% | 20.3% | 63.9% |

  Gain 2.1 is the best point with the fewest early endings; above 2.3 the model ends too early.
- Necessity and dose-response at the baseline's own 7 closing points (the edited quantity removed
  or scaled at that one position, everything downstream run): with the `</think>`-aligned output of
  the 69 neurons removed, the `</think>` logit falls by 5.3 to 6.8 nats and its sampling probability
  is 0.000 at all 7 (it was 0.10 to 0.81): the baseline's closings are made by this pathway. With
  gain 2.1 the probability is 1.000 at all 7.
- Selectivity at 21 ordinary line ends of the same generations: probability 0.000 at gain 1, mean
  0.034 at gain 2.1 (20 of 21 below 0.1; the one above is the line end right before a real
  closing), mean 0.467 at gain 3. So gain 3 would end the thinking regardless of content, and
  gain 2.1 does not.
- Verification of the written weights against the base on 16 baseline generations: only the one
  tensor differs; at line ends the mean sampling probability of `</think>` goes from 0.0000 to
  0.0103 (0.8 percent of line ends above 0.5); elsewhere the log-probability of the written token
  changes by -0.0005 on average; at the first generated token the probability stays 0.

## 3. Passes (4 of 15 used)

| pass | weights | correct | thinking ends | cut at cap | mean tokens | what it showed |
|---|---|---|---|---|---|---|
| 1 | unmodified | 7 of 128 (5.5%) | 7 of 128 | 121 of 128 | 31,286 | baseline |
| 2 | gain 2.1 | 37 of 128 (28.9%) | 80 of 128 | 88 of 128 | 24,164 | +30 newly correct, 0 lost; 40 answers finished (37 right), 40 answers ran to the cap, 48 never closed |
| 3 | gain 2.3 | 35 of 128 (27.3%) | 100 of 128 | 86 of 128 | 23,223 | more endings, no gain: 42 answers finished (35 right), 58 ran to the cap; 6 newly correct and 8 lost against pass 2 |
| 4 | gain 2.1 (same as pass 2) | 37 of 128 | 80 of 128 | 88 of 128 | 24,164 | texts identical to pass 2: the evaluation is deterministic for fixed weights, a repeat adds nothing |

Correct by difficulty level: baseline 0/11, 4/23, 3/27, 0/33, 0/34; gain 2.1: 8/11, 9/23, 8/27,
9/33, 3/34; gain 2.3: 7/11, 9/23, 10/27, 7/33, 2/34. A second answer extractor (`accuracy_rescore`)
gives 32.0% for the edited model against 5.5% for the baseline.

## 4. Candidates the checker refused

None. Both candidates I submitted (gain 2.1 and gain 2.3) were accepted.

One further candidate was built but never submitted to the checker or a pass, because my own
verification rejected it: the closing gain plus the output columns of the four loop-sustaining
heads multiplied by 0.35 (`~/wsp14/work/cand/k21_loop35`; four tensors would have changed). In real
forward passes it lowers the log-probability of the written tokens by 0.51 nats on average, i.e. it
damages normal prediction (section 2.5).

## 5. Predicted effect on the held-out set

By difficulty level the edited model gets about 73 of 100 Level-1, 39 of 100 Level-2, 30 of 100
Level-3, 27 of 100 Level-4 and 9 of 100 Level-5 problems right (passes 2 and 4). Weighted with the
level mix of MATH-500 (43 / 90 / 105 / 128 / 134 problems; I take this mix from memory, I could not
check it without internet) that is about 29 of 100 problems. The baseline's practice rates (0, 17,
11, 0 and 0 of 100 by level) give about 5 of 100.

Honest uncertainty: the practice estimate itself has a standard error of about 4.4 points (128
problems, one sample each), and the held-out pass adds about 2 points. So I expect 25 to 32 of 100,
most likely near 29. The gain over the baseline (about +23 points on the practice set) is more than
five standard errors and will carry over; the exact level does not matter for that conclusion.

## 6. What I did not fix, and why I stopped with 11 passes left

- **The answer phase (40 of 128 problems at gain 2.1).** After the thinking ends, the model must
  choose at a paragraph break between starting the final `\boxed{}` line and writing more. In the
  answers that finish, that decision is sharp (probability 0.75 at the real end, 0.006 elsewhere).
  In the answers that never finish, the `\boxed` logit sits 13.6 nats below the top token whether
  or not the answer has already been stated, and the answer ends in a verbatim loop in 40 of 40
  cases. I tested the handles I could think of, none is clean enough for a weight edit:
  - the length of the thinking is not the cause: with the thinking cut to 200 tokens the same
    answer text still shows a gap of -12.5 at the same point (answers that finish show about 0 at
    every thinking length);
  - the last MLP's contribution to the `\boxed` logit is the same at every paragraph break (+5 to
    +7), so no gain change of existing neurons can act there;
  - a uniform lift of `\boxed` after paragraph breaks nets at most +3 points in simulation, because
    an ordinary paragraph break in a healthy answer is only 7 nats behind;
  - linear readouts to gate a lift are too weak: "doubt mode" 72 to 74 percent, "conclusion
    sentence just written" 33 percent recall at 24 percent false alarms;
  - an explicit statement of the correct answer occurs in only about 5 to 9 of the 40 to 58
    unfinished answers (by a rough manual count of string matches).
  The only remaining way to make those answers end is a content-independent budget ("box whatever
  you have after N paragraphs"), which is the "close regardless of the problem" change the task
  excludes. I did not do it.
- **Raising the gain further** does not help (pass 3): the unfinished answers move from "thinking
  never ends" to "answer never ends" (48 + 40 at gain 2.1, 28 + 58 at gain 2.3).
- **The verbatim loop itself is not repaired.** Its cause is now localised (DeltaNet head 8 of
  layer 0 and three other early heads, section 2.5), but these heads carry the context for normal
  prediction, and every attenuation I measured damages the model far more than the loop it removes.
- **A correction I made during the work.** My tool for "remove one component at one position and
  carry the change through the remaining layers" had a defect (a wrong attention mask whenever the
  continuation started after layer 0). I caught it with a no-edit control before finishing, fixed
  it, and redid the two analyses that depended on it (the layer-wise attribution and the loop
  ablation); all numbers in this report come from the corrected runs or from full forward passes,
  which were never affected. The edit itself, its verification and all four passes do not involve
  that tool.
- **Repeating passes** is uninformative (pass 4 is bit-identical to pass 2), and every remaining
  candidate I could design was estimated at +3 points or less, below what one pass can resolve
  (standard error 4.4 points).
- I did not verify whether this checkpoint equals the public Qwen3.5-0.8B release (no internet, no
  reference copy); I only found no anomalous tensor.

Files: notes in `~/wsp14/work/notes.md`; analysis scripts in `~/wsp14/work/analysis/` (the edit is
built by `apply_edit.py`, checked by `verify_edit.py` and `a60_validate.py`; the attribution is
`a63_attr_clean.py`, the corrected loop study `a31_loop_specific.py` + `a61_loop_heads.py` +
`a64_loop_atten.py`); measured candidates in `~/wsp14/work/cand/k21` and
`~/wsp14/work/cand/k23`, the rejected one in `~/wsp14/work/cand/k21_loop35`, copy of the final
state in `~/wsp14/work/best_pass2_k21`.
