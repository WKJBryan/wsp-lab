# Phase 14 weight-engineer notes

## Setup facts
- Model: Qwen3.5-0.8B (post-trained, hybrid: 18 Gated DeltaNet layers + 6 full-attention layers 3/7/11/15/19/23; tied embeddings).
- Protocol: thinking mode, temp 0.6, top_p 0.95, top_k 20, presence penalty 1.5, cap 32,768 new tokens, 128 practice problems, seed fixed.
- Budget: 15 passes. Pass 1 = unmodified model (started 11:48).
- Own work dir: ~/wsp14/work (analysis scripts in work/analysis).

## Lead from the model card (README.md line 856)
"in thinking mode ... Qwen3.5-0.8B is more prone to entering thinking loops compared to other Qwen3.5
models, which may prevent it from terminating generation properly."
=> Candidate failure mechanism: thinking loops / failure to emit </think> within the 32,768-token cap.

## Weight statistics (CPU only, a01_stats / a02_structure)
- No tensor stands out by mean/std/kurtosis/row-norm statistics; no layer is a copy of another layer.
- SwiGLU pairing is intact in every layer (corr(log ||up_i||, log ||down_i||) strongly positive, smooth across
  layers: +0.86 early, +0.49 L22, +0.17 L23); so no row permutation of gate/up/down.
- Dead rows only in conv1d/in_proj_qkv of a few GDN layers (7 in layer 0, also its conv1d): looks like natural
  dead channels (same indices in conv1d and in_proj_qkv), not damage.
- Per-head norms are all within a normal range in every layer.
=> No obviously "broken tensor". The failure, if any, is functional (to be seen in the generations).

## Pass log
| pass | edit | accuracy_boxed | closed | cap_hit | mean_tokens | note |
|---|---|---|---|---|---|---|
| 1 | none (base) | 5.5 (7/128) | 5.5 (7/128) | 94.5 (121/128) | 31,286 | baseline; wall 808 s |
| 2 | closing gain k=2.1 (cand/k21, copy in work/best_pass2_k21) | 28.9 (37/128) | 62.5 (80/128) | 68.8 (88/128) | 24,164 | +23.4 points; 30 newly correct, 0 lost; rescore 32.0; by level 8/11, 9/23, 8/27, 9/33, 3/34 |
| 3 | closing gain k=2.3 (cand/k23) | 27.3 (35/128) | 78.1 (100/128) | 67.2 (86/128) | 23,223 | not better than 2.1 (6 newly correct, 8 lost vs pass 2): more closings, but 58 of the 100 responses then ran to the cap; by level 7/11, 9/23, 10/27, 7/33, 2/34 |
| 4 | same weights as pass 2 (k=2.1) | 28.9 (37/128) | 62.5 (80/128) | 68.8 (88/128) | 24,164 | bit-identical texts to pass 2 (cmp): the evaluation is deterministic for fixed weights, so repeats give no extra information |

## FINAL STATE
~/wsp14/sandbox/model = candidate k=2.1 (shard byte-identical to work/cand/k21 and work/best_pass2_k21;
28 other files identical to the base; checker fingerprint cd55253ac536fe76 = the one measured in passes 2 and 4).
Passes used 4 of 15.

## CORRECTIONS (found on re-checking before writing the report)
A bug in my own step tool (tf.Engine): a fresh cache that I filled by assigning tensors still reported
sequence length 0 (its book-keeping flags were not set), so in every "continue from layer l at one
position" run the attention layers used a wrong mask. With NO edit at all that path gave logits
8-20 nats below the true ones at loop positions. Fixed (flags set in _restore); a no-edit
continuation now matches the full forward exactly (difference 0.0).
Affected and REDONE: (1) the layer-wise attribution (a20/a21) -> redone with full forward passes only
(a63_attr_clean.py, sums exactly to the true logit); (2) the loop ablation (a30/a31) -> redone with
the fixed tool (a31 tag S2, a61, a62, a64). NOT affected (full forward passes): a10/a11/a15,
a13/a14, a22/a23, a24-a30 (closing neurons, simulation), a41-a53 (response phase), a60 (validation).
1. Clean attribution at the 7 closing points (sum 24.87) vs 140 line ends after a concluding
   sentence (16.85) vs 140 other line ends (16.45): layers 0-13 +1.55 / +1.21 / +1.10 (no block above
   0.5); layers 14-21 +10.62 / +6.91 / +6.31; layer 22 +4.25 / +1.56 / +1.73 (DeltaNet head 13:
   +2.81 / +0.91 / +0.99); layer 23 +8.44 / +7.16 / +7.31 (MLP +8.28 / +6.65 / +6.74).
2. Loops (corrected): most heads do nothing at a loop position (median drop over all 336 heads
   0.00). The loop is sustained by a few early heads: layer-0 DeltaNet head 8 (loop token -9.5 nats
   when removed at the loop position, breaks 13 of 40 loops), layer-0 head 3 (-4.5), layer-3
   attention head 7 (-4.5), layer-1 head 7 (-3.6), layer-22 head 9 (-3.1).
   My earlier line "no single head removal breaks a loop" was an artifact and is withdrawn.
3. But attenuating those heads IN THE WEIGHTS is not viable: single-position emulation (head slice
   scaled at one position, everything upstream intact) said halving 4 heads weakens the loop margin
   by 4.9 nats for 0.02 nats of cost; the real weights (cand/k21_loop35: 4 heads x 0.35) change the
   keys, values and recurrent states of the whole context, and the log-probability of the written
   token falls by 0.51 nats on average (min -14.9) against 0.0005 for the closing-gain edit alone
   (verify_edit.py). Candidate rejected without a pass. The loop heads are core context heads.
So the final state remains the closing gain k=2.1 alone.

## Causal validation of the edited quantity (a60, full forward passes with hooks; base model; 7 real closing points + 21 ordinary line ends)
- k=0 (remove the </think>-aligned output of the 69 neurons at that position): logit -5.3..-6.8 nats,
  sampling p(</think>) 0.000 at all 7 closing points (was 0.10..0.81) => the baseline's closings are made by it.
- k=2.1: logit +5.5..+6.7, p = 1.000 at all 7 closing points; at the 21 ordinary line ends mean p = 0.034,
  20 of 21 below 0.1 (the one above is the line end right before a real closing).
- k=3: ordinary line ends mean p = 0.467 => that would close regardless of content; k=2.1 does not.

## What pass 3 showed
- Raising the gain to 2.3 closes the thinking on 100 of 128 (28 never close, all of them Level 3-5) but
  does not raise accuracy: the response phase is now the binding limit. Responses finished in 42 of
  100 cases (pass 2: 40 of 80); 35 of the 42 finished responses are correct.
- Response-phase termination was studied in detail and NOT repaired, because no clean handle exists:
  * after a paragraph break the model must start "\boxed{"; in finished responses that decision is
    sharp (p 0.75 at the real end, 0.006 elsewhere), in stuck responses the \boxed logit is 13.6 nats
    below the top token whether or not the answer was stated (a44/a47/a48);
  * the last MLP's contribution to that logit is the same at every paragraph break (+5..+7), so no
    gain change of existing neurons can help; a uniform lift nets at most +3 points in simulation
    (healthy paragraph breaks after an incidental answer mention are only 7 nats behind);
  * linear readouts are too weak to gate a lift: "doubt mode" 72-74% (a49), "conclusion sentence
    just written" 33% recall at 24% false positives (a53);
  * an explicit answer statement containing the right answer occurs in only ~6-9 of 40-58 stuck
    responses (a51); every stuck response ends in a verbatim loop.
- Decision: keep the k=2.1 state as the repair (gentler, higher observed accuracy).

## What pass 2 showed
- The edit works as designed: thinking now ends on 80 of 128 problems. Of those, 40 responses finished
  (37 correct, 3 wrong) and 40 responses ran to the cap without ever writing \boxed{} (0 correct).
- 48 problems still never close the thinking (mostly Level 4-5; inside their verbatim loops `</think>`
  stays outside the top-20 even with the 2.1x gain).
- Trajectories are fresh samples: with the same seed the pass-2 texts diverge from the pass-1 texts after
  a median 3% of the thinking (vLLM is not bit-reproducible across batch compositions), so the
  simulation on baseline trajectories is a distributional estimate only. Closing rate matched (48% of
  problems closed before any loop vs 51% predicted), accuracy 28.9 vs 31 predicted (proxy).
- The response phase fails the same way as the thinking: after "Therefore ... is 9." the model must
  choose the paragraph break (which leads to "\boxed{9}") or a new line, where it picks "Wait, let me
  double check" (p 0.5-0.8) and is trapped. In finished responses the final-line decision after the
  paragraph break is nearly deterministic (p(\boxed) 0.9-1.0); in stuck ones the \boxed logit stays
  13.6 nats below the top token whether or not the answer was already stated (a44).
- Teacher-forced test (a46): with the thinking cut to 200 tokens the stuck responses STILL show the
  -12.5 gap at the same response position, finished ones ~0 at every thinking length. So it is not the
  length of the thinking but the response's own recent text (once it has taken a "Wait" line it stays
  in that mode). The decisive moment is the end of the first answer-stating sentence.

## Diagnosis so far (after pass 1)
What the failure looks like (a05_texts.py, a06_loops.py on gen_1.jsonl):
- 121 of 128 generations are cut at the cap; 118 of those end in a VERBATIM loop with a clean period
  (median 37 tokens, range 6-539), repeated hundreds to thousands of times (p57: 5,128 times).
- The loop starts after a median of ~7,200 tokens; on average ~80% of every text is the loop.
- Before the loop the model has usually already solved the problem (often correctly), and then cannot stop:
  long doubt cycles ("Wait, is it possible that the user meant ...", "Maybe the question has a typo",
  "Final check ... Ready ... Wait ...") which progressively shrink into exact repetition.
- No language mixing (0% CJK). Formatting degenerates into nested "    *   " bullets.
- The 7 that closed did so after 1.7k-16k tokens, always right after a concluding sentence
  ("Final Answer: 2/3.", "I'll write out the solution now.", "The final answer is -2, 2.").
=> failure mechanism = (a) termination failure: "doubt/re-check" continuation keeps outcompeting
   the `</think>` decision after the answer is found; (b) collapse into a token-copying (induction) loop.
Token counts of re-tokenised texts match n_tokens of the harness (so teacher forcing is exact).

## Where the failure is produced (teacher-forced reads, all on the 128 baseline trajectories)
1. The decision "close the thinking" is taken at exactly one place: the token right after a line break
   (after "\n"), where the candidates are `</think>` and the indentation token that continues the
   model's structured format. At ordinary line ends the indentation token leads `</think>` by ~14.5
   nats (top-1 logit ~30 vs `</think>` ~14.5); after "So the answer is X." by ~8; only right after
   "I'll write the response now"-type lines by ~2.5 (a14, a15, probe_p1).
2. That margin is flat in time: `</think>` logit at line ends 14.6 -> 14.4 from token 300 to 7000 and
   the gap even widens (-15.0 -> -16.3) because the indentation logit grows. There is no "enough
   thinking" signal at all (within-generation corr(position, logit </think>) = +0.11).
3. Exact logit attribution of `</think>` at line ends (a20/a21): written almost entirely in the last
   layers; the single largest writer is the MLP of layer 23 (+7 at every line end, +6.2..8.8 at the
   real closing points), then GDN head 13 of layer 22 (+0.7 ordinary, +2.2 at closing points), then
   attention head 6 of layer 23 and small contributions from layers 14-21; layers 0-13 contribute ~0.
4. Inside the layer-23 MLP the closing output is made by ~69 specific neurons that fire at line ends
   inside the thinking and are nearly silent elsewhere (a27/a28): their sum raises the `</think>`
   logit by +6.36 at line ends (sd 1.6, stable from token 3 to 5000), by +0.5 at the first generated
   token, +1.75 mid-line, +2.2 in the response after </think>, +3.3 on prompt tokens.
5. The network hardly represents "answer found" at line ends: a linear probe separating lines that
   state a conclusion from process lines reaches only 61-62% (5-fold CV by generation, any layer);
   position is well represented (80-88%). So the model has no content signal strong enough to beat
   the format continuation; the little it has is the tail of the `</think>` logit.
6. The verbatim loops are NOT made by a small set of copying heads: at a loop position every block
   adds to the repeated token (removing any MLP block costs 6-14 nats there vs 2-3 nats at ordinary
   positions; no single head removal breaks a loop in 40 contexts, a30/a31). It is a network-wide
   attractor fed by the repeated context; it cannot be repaired locally.
=> Failure mechanism: the termination pathway (line-end closing neurons of the layer-23 MLP, GDN
   head 13 of layer 22) is far too weak relative to the format-continuation token, so the thinking
   never ends, degenerates into doubt cycles and then into verbatim loops.

## Candidate repair and its predicted effect (a29/a30, exact sampling rules, validated at k=1)
Multiply by k the component of those 69 neurons' output weights that points along the `</think>`
read-out direction (adds (k-1) x 6.4 nats at line ends only; mid-line p(</think>) stays 0).
k:    closed  gold-stated@close  premature(closed before the gold answer appears)
1.0     6.9%     5.9%      0.9%   (real pass 1: 5.5%)
1.75   17.3%    13.9%      3.4%
2.0    35.7%    24.0%     11.7%
2.1    50.8%    31.3%     19.5%   (median closing after 1,850 tokens; 4.6% close before 400 tokens)
2.2    63.8%    33.4%     30.4%
2.5    84.2%    20.3%     63.9%   (closes too early: median 263 tokens)
Chosen: k = 2.1 (best accuracy proxy with the fewest premature closings).