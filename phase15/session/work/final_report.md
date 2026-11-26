# Phase 14 final report

## Part 1: plain and short

**Outcome.** On the 128 practice problems the model went from **7 right to 68 right** (5.5% to
53.1%). A near-identical variant scored 69 right in a fourth pass. On the unseen MATH-500 set I
expect roughly **half the problems right (about 45% to 60%)**, against about 5% for the unmodified
model. That prediction is not verified: MATH-500 was never run here.

**What was wrong.** Nothing is broken in the weights. The model is the unmodified public
checkpoint (its file hash equals the vendor's download record). Its own habit is the problem: while
thinking, it re-checks the same thing over and over ("Wait, ..."), falls into repeating itself word
for word, and never stops. 121 of 128 practice generations ran to the 32,768-token limit without
ever finishing their thinking, so no answer was delivered, although in most solvable cases the
right answer was already sitting in the reasoning.

**The fix.** Two tiny switches inside one layer (layer 14), each made from one existing neuron that
was doing no useful work. Each switch reads the layer's input, and only when that input lies in a
detected direction does it add "end the thinking now" to the model's output. Switch 1 detects
word-for-word repetition. Switch 2 detects a runaway numbered list. Everywhere else both are silent.

**Cost.** None measurable. The prediction loss on ordinary reasoning text is unchanged to four
decimals, and the model's own natural endings, fresh text and problem statements are untouched.

**What remains.** About a third of problems (mostly the two hardest levels) still get no answer:
after the thinking is ended the model falls back into the same loop while writing the answer. These
are problems it cannot solve (the correct answer shows up in their text no more often than in
problems it answers wrongly). I looked for a targeted repair for this and found none that was safe,
so I left it alone rather than force guesses.

**Budget.** 4 of 15 passes used. The checker refused nothing; all three edited states were accepted.

**Needed from the user.** Nothing. The final weights are in place and accepted.

---

## Part 2: technicals, in time order

### Step 0. Setup
- Read the task file and the harness tools (the forward-pass tool, the edit checker, the graders,
  the run-validity checks) to learn the exact protocol: temperature 0.6, top-p 0.95, top-k 20,
  presence penalty 1.5, 32,768 new tokens, fixed seed, last `\boxed{}` graded.
- Confirmed the working copy was identical to the pristine reference before any edit
  (`save_edit.py`: "ACCEPTED: identical to the base model").
- Library caveat found and worked around: in this transformers version `last_hidden_state` is
  returned *after* the final norm, and `output_hidden_states` is not the raw residual stream, so I
  captured the residual stream with my own hooks.

### Step 1. Baseline (pass 1, unmodified model)
- accuracy_boxed 5.5 (7/128), thinking closed in 7 of 128, cap hit 94.5% (121/128), mean 31,286
  tokens. By level: L1 0/11, L2 4/23, L3 3/27, L4 0/33, L5 0/34.
- Shape of the failure: 113 of 128 generations end in an exact verbatim cycle (period 6 to 322
  tokens) that begins after a median of about 7,000 generated tokens and never ends. Cycles are
  usually "Wait, ..." lines or short statements repeated. The non-repeating parts are fluent.
- Inside a loop the next cycle token wins by about 15 raw logits. The protocol's presence penalty
  is a flat -1.5 on tokens already present (not per count), so it cannot break a loop.

### Step 2. Diagnosis: intact model, own behaviour
- Weight statistics per tensor are smooth across layers; no mispaired gate/up rows, value/output
  head slices or query/key dimensions; no all-zero rows.
- Removing each attention head and each Gated DeltaNet head at runtime (no weight change) raises
  the clean-text loss for every head but one unused one: nothing harmful to remove.
- Decisive check: the SHA-256 of the weight file equals the hash recorded at download time
  (commit `2fc06364…`, LFS sha256 `04b1c301…4696`). The checkpoint is the vendor's, so the
  repetition is the model's natural failure, not injected damage.
- 14 rows of magnitude 2^-123 in layer 0's query/key projection look odd but are part of that
  release and sit in one head of the first layer; left untouched.

### Step 3. Finding a usable signal
- Collected the layer-14 MLP input at every line start of every pass-1 trace and at every token
  position of every trace.
- A single direction (mean of the input at line starts inside loops minus the mean at line starts
  in fresh text) separates the two on held-out problems with AUC about 0.98 at layers 11 to 14.
- In-loop line starts score about 454 to 480 (quartiles); the highest legitimate content reaches
  311 (a genuine case enumeration) and 262 (repeated Asymptote code inside a problem statement).
- The few "fresh" positions that score high are traces that were recycling the same lines long
  before an exact cycle formed, i.e. real repetition.
- A constant direction is needed because the MLP has no bias: the closed-form direction
  (Sigma + 0.05 tr(Sigma)/1024 I)^-1 mean has projection 1 +/- 3 to 5% at 98% of positions.

### Step 4. Building and calibrating the first switch (hooks only, no weight change)
- Calibrated the write: a unit write along the output direction of `</think>` at layer 14 raises
  that logit by about +7 per unit (saturating near +60). In loops `</think>` trails the top token
  by 25 to 30 logits, so a write of 4 to 6 units flips it.
- Full neuron response evaluated on every collected position with soft threshold 425, width 20,
  write 5.5: flip-level positions in fresh text 118 of 695,391 (all in one genuinely repeating
  trace), in the pre-loop phase 59 (4 traces), inside loops enough to break 97 of 114 loops within
  their first 800 loop tokens.
- Emulation on real contexts: natural closures and fresh line ends unchanged; at in-loop flip
  positions the base top token has probability 1.00 and the grafted model gives `</think>` 0.99 to
  1.00. Right after the forced `</think>` the detector score collapses (about 450 to between -69
  and +69), so the neuron goes silent and the model starts a normal final answer.

### Step 5. Pass 2: the first switch alone (fingerprint `071c042292c351ee`)
- accuracy_boxed 46.1 (59/128), thinking closed 93.8%, cap hit 34.4%, mean 18,624 tokens.
  By level: L1 9/11, L2 14/23, L3 15/27, L4 16/33, L5 5/34. All run-validity checks passed.
- Gain over baseline: +40.6 points, about nine standard errors.
- Breakdown: 84 ended (59 right). 44 capped, none right: 36 closed the thinking and then fell back
  into the same loop while writing the answer; 8 never closed because they were runaway numbered
  lists (each line differs by an increment), not word-for-word repeats.

### Step 6. The remaining failures, and what did not work
- Final-answer-phase runaway: I tested every single-token lever at real stuck positions. None is
  naturally followed by `\boxed{` (the weak link is always 0.2 or less), the model's guess at those
  positions is mostly wrong (about 2 of 30), and the detector collapses one position after any
  forced token, so multi-token chains cannot be driven. Blocking the word "Wait" would only reword
  the re-check. Amplifying the model's linear "answer mode" feature by up to six times changed the
  probability of "Wait" only from 0.47 to 0.42. Conclusion: a capability limit, left alone.
- Runaway enumerations: the same problems produce them in every pass. A dedicated direction
  (closed-form regularised linear discriminant between positions in 13 runaway traces and fresh
  positions) recognised 12 of 13 held-out runaway traces at 100% recall with zero false alarms on
  11,753 held-out fresh samples. Its score builds up slowly (still negative at the start of a
  runaway, about 20,000 tokens of enumeration needed to pass +15), so a legitimate short list cannot
  reach it. On all 978k positions of the practice traces the highest score outside runaways is +1.

### Step 7. Pass 3: both switches (fingerprint `a3de53261dbcd9be`, the final state)
- accuracy_boxed 53.1 (68/128), second extractor 54.7, thinking closed 98.4%, cap hit 25.0%,
  mean 17,769 tokens. By level: L1 10/11, L2 17/23, L3 16/27, L4 17/33, L5 8/34.
- 96 ended (68 right); 32 capped, none right (30 final-phase runaways, 2 never closed: a pure digit
  cycle and a weaker list). 84 generations contain a boxed answer.
- Paired with pass 2: 15 problems gained, 6 lost. Of the 8 never-closed traces of pass 2, 5 now
  ended (3 right). Tracing which switch fired first shows the second switch itself closed only 1 to
  2 traces in this pass (p26, right); most of the 46 to 53 change is trajectory reshuffle.

### Step 8. Pass 4: a lower soft threshold for the first switch (fingerprint `cde4c2c5907c1a00`)
- Offline: soft threshold 390 (flip near 405) instead of 425 breaks 106 of 114 loops within 800
  tokens instead of 97; its extra early firings all land at the onset of a repeated block with the
  answer already formed. Loss checks: ordinary early text unchanged except inside two real loops
  that sit in that window; probability of `</think>` in the first 600 tokens below 0.01 in all
  traces; accumulated probability of a spurious closure before the first strong repetition: median
  0.000, 90th percentile 0.09 (incumbent 0.16).
- accuracy_boxed 53.9 (69/128), closed 99.2%, cap hit 12.5%, mean 15,560 tokens; 87 boxed answers;
  1 never closed. By level: L1 9/11, L2 19/23, L3 18/27, L4 17/33, L5 6/34.
- Decision, fixed before the pass: the incumbent stays unless the challenger is clearly better.
  One problem more is noise, so the incumbent was restored.

---

## The exact final edit

All changes are in layer 14's MLP, in two neurons (1439 and 2910). Each was among the
lowest-output neurons of the layer; removing either at runtime changes the clean-text loss by less
than 0.0001 nats. Names, shapes and dtype (bf16) are unchanged; every other byte of the weight file
is as in the pristine reference.

Let x be the layer's MLP input (output of `post_attention_layernorm`). Each neuron computes
`a = silu((d·x − theta·w·x) / s) * (w·x)` and adds `a * c * v` to the residual, where
`w·x ≈ 1` everywhere and `v` is the unit vector along `(1 + final-norm weight) ⊙ E[</think>]`.

| tensor | rows / columns | new value | norm (was) |
|---|---|---|---|
| `model.language_model.layers.14.mlp.gate_proj.weight` [3584, 1024] | row 1439 | bf16((d − 425·w) / 20) | 2.172 (0.369) |
| same | row 2910 | bf16((d_enum − 5·w) / 10) | 4.231 (1.050) |
| `model.language_model.layers.14.mlp.up_proj.weight` [3584, 1024] | row 1439 | bf16(w) | 0.0907 (0.222) |
| same | row 2910 | bf16(w) | 0.0907 (0.230) |
| `model.language_model.layers.14.mlp.down_proj.weight` [1024, 3584] | column 1439 | bf16(5.5·v) | 5.500 (0.203) |
| same | column 2910 | bf16(2.0·v) | 2.000 (0.221) |

Checker's own summary: gate_proj 2047 values changed (largest 0.530, size of the change 4.74);
up_proj 2048 values (largest 0.027, size 0.345); down_proj 2048 values (largest 0.795, size 5.86).

The directions, all computed from the model's own generations on the 128 practice problems:
- `d` (repetition): mean of x at line starts inside verbatim loops minus mean of x at line starts in
  fresh text, over the pass-1 traces. |d| = 20.27.
- `w` (constant): `(Sigma + 0.05·tr(Sigma)/1024·I)^-1 mean(x)` over fresh positions, scaled so that
  mean(x·w) = 1.
- `d_enum` (runaway enumeration): `(Sigma + lambda·I)^-1 (mean_runaway − mean_fresh)` with
  lambda = 1e-3·tr(Sigma)/1024, between positions of 13 runaway traces (pass 2: p28 p29 p31 p37 p44
  p57 p99 p123; pass 1: p16 p64 p85 p106 p112) and fresh positions (first 3,000 generated tokens of
  the pass-1 traces), scaled so that runaway median minus held-out fresh maximum = 100.

Rule note: `d` and `w` are means and covariances; `d_enum` is a regularised linear discriminant.
All are one-shot closed-form statistics. No backward pass, autograd, optimizer or iterative fitting
to a loss was used at any point.

## Evidence for the diagnosis, in one place
1. The model is intact (hash check) and every head is useful when removed.
2. 113 of 128 baseline generations end in an exact repeated cycle; the cycle's next token has
   probability 1.00 at 97% of in-loop line starts.
3. The network itself carries a linear "this line start is in a repeated context" signal at layer
   14: AUC 0.98 on held-out problems; loops 454 to 480 against at most 311 for any legitimate
   content.
4. When the thinking is ended at a loop, the model writes a final answer that is right in about
   70 to 80% of the cases it answers: the answer was there, the loop prevented delivering it.
5. After the edit: `</think>` 0.99 to 1.00 at in-loop flip positions (base: the loop token at 1.00);
   natural closures, fresh line ends and problem statements unchanged (largest logit change 0.28);
   clean-text loss 0.3222 before and after.

## Passes spent (4 of 15)

| pass | state | accuracy_boxed | closed | cap hit | mean tokens | by level L1..L5 |
|---|---|---|---|---|---|---|
| 1 | unmodified | 5.5 (7/128) | 5.5% | 94.5% | 31,286 | 0/11, 4/23, 3/27, 0/33, 0/34 |
| 2 | repetition switch | 46.1 (59/128) | 93.8% | 34.4% | 18,624 | 9/11, 14/23, 15/27, 16/33, 5/34 |
| 3 | + enumeration switch (**final**) | 53.1 (68/128) | 98.4% | 25.0% | 17,769 | 10/11, 17/23, 16/27, 17/33, 8/34 |
| 4 | same, first switch threshold 390 | 53.9 (69/128) | 99.2% | 12.5% | 15,560 | 9/11, 19/23, 18/27, 17/33, 6/34 |

No pass failed or was refunded; all run-validity checks passed.

## Candidates the checker refused
None. Three edited states were submitted to `save_edit.py` and all three were accepted.

## Predicted effect on the held-out set (MATH-500)
- Unmodified model: about 3% to 9% (practice gave 5.5%; the mechanism that prevents answers is the
  same there).
- Final state: about 45% to 60%, central estimate near 50%. Basis: 53.1% on the practice set with a
  standard error of 4.4 points for one pass, the near-identical variant at 53.9%, and weighting the
  per-level practice rates by MATH-500's level mix (43 / 90 / 105 / 128 / 134, as commonly reported;
  not verified here) gives 53%. Expected: thinking closed in about 98% of generations, cap hit
  about 25%.
- Where it will lose: the hardest levels. About a third of problems end with no answer because the
  model loops again while writing it. One practice problem (p53) had the right answer written out
  but not boxed and was graded wrong; about one or two answers per hundred may be lost that way.

## Limitations and process notes
- MATH-500 was never run and none of its problems were read; all directions come from the 128
  practice problems and the model's generations on them.
- The practice scores of the two grafted states are single draws. Pass-to-pass differences of
  several points are mostly trajectory reshuffle (15 gained and 6 lost between passes 2 and 3), so
  the addition of the second switch is justified by mechanism evidence, not by its +7 points: it
  closes runaway enumerations that otherwise never close, and it is silent on 978k fresh positions
  and on every cleanly ended final answer.
- The final-answer-phase runaway (about a third of problems) is not repaired.
- Process: I read the installed transformers and vLLM library code at run time to attach hooks
  correctly; I did not open the lab repository, other phases or other model directories. The folder
  `~/wsp14/techcheck_evidence` appeared in my first directory listing; I did not open its files.
  No GPU job of mine ran during a pass. Once I ran two of my own GPU jobs at the same time (never
  during a pass), which only slowed them down.
- Tools used: Read, Write, Edit and Bash only.

## Files (all under `~/wsp14/work/`)
- `notes.md`: running log with every measurement.
- `final_report.md`: this report.
- `g1_L14n1439_t425/`, `g2_N1_N4/` (identical to the final weights), `g3_N1t390_N4/`: copies of the
  three states.
- `spec_g2.json`, `spec_g3.json`: exact graft specifications; `scripts/apply_grafts.py` rebuilds a
  state from the pristine reference tensors.
- `scripts/`: all analysis scripts; `out/`: their outputs, including copies of the four passes'
  generations (`out/gen_1..4.jsonl`).
