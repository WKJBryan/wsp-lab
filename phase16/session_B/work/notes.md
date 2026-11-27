# Phase 16 session B — working notes

Model: Qwen3.5 checkpoint (text: 24 layers, width 1024; full attention at layers 3,7,11,15,19,23;
Gated DeltaNet elsewhere; MLP 3584; vocab 248320, tied embeddings). Working copy:
`~/wsp14/sandbox/model`, pristine reference `~/wsp14/sandbox/base_model`.

Budget: 15 passes. One pass = 128 practice problems, fixed seed, temperature 0.6, top_p 0.95,
top_k 20, presence penalty 1.5, cap 32,768 tokens. A pass takes about 13 minutes when most
generations run to the cap.

Tooling I built (all in `~/wsp14/work/`):
- `tfkit.py` / `reads.py`: teacher-forced reads with transformers (text tower only, checkpoint dtypes),
  hooks for residual stream, per-head attention / linear-attention outputs, MLP outputs; protocol
  simulation (T 0.6, top-p 0.95, top-k 20, presence 1.5); direct logit attribution (DLA).
  Fidelity check against the engine: the protocol argmax equals the generated token at ~90% of
  positions before the loop and ~97-99% inside loops; mean protocol probability of the generated
  token 0.85-0.99.
- `read1.py <pass> <out prefix>`: profile + DLA at every line-end / decision position, per problem.
- `an1.py`, `analyze_gen.py`, `show_gen.py`: text and profile analysis.

## Pass log

| attempt (the tool's number; files gen_<attempt>.jsonl / eval_pass_<attempt>.json) | candidate | hypothesis | result |
|------|-----------|------------|--------|
| 1 | base model (unchanged) | measure the starting point | accuracy_boxed 5.5 (7/128), closed 5.5 (7/128), cap-hit 94.5 (121/128), mean tokens 31,286; by level L1 0/11, L2 4/23, L3 3/27, L4 0/33, L5 0/34; wall 777 s |
| 2 | R1 = base + "Wait" component of five copy heads inverted (o_proj of layers 15, 19, 23; receipt 22f8d829ffeadf7b) | R1 alone: Wait-loops break, closed rate 15-30%, early reasoning unchanged | accuracy_boxed 31.2 (40/128), closed 32.8 (42/128), cap-hit 68.0 (87/128), mean tokens 25,995; by level L1 7/11, L2 11/23, L3 9/27, L4 10/33, L5 3/34; wall 652 s; admission first/second half: finished 21.9% vs 42.2% (p=0.022, above the 0.001 bar) |
| 3 | R1E2 = R1 + stop neuron E2 in the last MLP (receipt 41d6229f334e500f) | forced closes at concluding lines: closed 45-55%, accuracy 38-44% | accuracy_boxed 37.5 (48/128), accuracy_rescore 40.6, closed 45.3 (58/128), cap-hit 61.7 (79/128), mean tokens 22,700; by level L1 7/11, L2 17/23, L3 7/27, L4 14/33, L5 3/34; wall 599 s; paired with pass 2: gained 13, lost 5 (McNemar p = 0.096, about 1.9 standard errors of the paired difference) |
| 4 | S1wE2 = joint flip of thirteen line-start words (incl. Wait) in nine copy heads + E2 (receipt d683195c14f0499b) | word-started block loops break, more closes (55-65%) | accuracy_boxed 34.4 (44/128), closed 42.2 (54/128), cap-hit 62.5 (80/128), mean tokens 22,584; by level L1 7/11, L2 15/23, L3 11/27, L4 6/33, L5 5/34; wall 550 s; paired with pass 3: gained 8, lost 12 (p = 0.50). Not kept. |
| 5 | R1E2D2 = R1 + E2 + gain neuron D2 on the model's own exit logit (receipt 636758da58420a23) | closes where the model's own exit evidence is already high: closed 60-70%, accuracy 40-45% | accuracy_boxed 35.9 (46/128), accuracy_rescore 39.1, closed 65.6 (84/128), cap-hit 55.5 (71/128), mean tokens 21,351; by level L1 8/11, L2 14/23, L3 10/27, L4 9/33, L5 5/34; wall 506 s. 57 closes finish (46 right), 27 close and then run to the cap. Paired with pass 2: gained 16, lost 10; with pass 3: gained 11, lost 13 (p = 0.84). Not kept. |
| 6 (no measurement, refunded) | R1E2 | launched by my mistake with an empty hypothesis: the hypothesis file had not been written after a transient error reaching the WSL files, and the launcher did not check | charged 20:52:25, stopped by me at 20:52:28 before any generation; the tool wrote "INTERRUPTED, pass refunded" and returned the pass (counter back to 5). The ledger keeps the row with the empty hypothesis; I did not touch it. The launcher now refuses to start without a non-empty hypothesis and note file. |
| 7 | R1E2, exact re-run (same receipt 41d6229f334e500f) | deterministic engine -> identical to pass 3 (48/128); otherwise a second draw of the same weights, 44-52 | accuracy_boxed 37.5 (48/128), closed 45.3, cap-hit 61.7, mean tokens 22,700: identical to pass 3 in every number; gen_7.jsonl is byte-identical to gen_3.jsonl. The engine is exactly deterministic: a fixed candidate has no run-to-run noise, and every difference between passes is caused by the weight change through diverging sampling paths. |
| 8 | R1p = R1 + o_proj columns of layer 7 head 3 x 1.02 (receipt 7fb1ff193cca7cdf) | functional no-op that changes the sampling path: a second draw of R1's chaos; expected near 40/128 | accuracy_boxed 18.0 (23/128), closed 17.2 (22/128), cap-hit 83.6 (107/128), mean tokens 29,376; by level L1 4/11, L2 5/23, L3 7/27, L4 4/33, L5 3/34; wall 706 s. Paired with pass 2: gained 4, lost 21 (p = 0.0009); closes 42 -> 22 (18 kept, 24 lost, 4 gained). |
| 9 | R1E2p = R1 + E2 + same perturbation (receipt 931c8887a29d4600) | second draw of R1E2's chaos; expected near 48/128 | accuracy_boxed 34.4 (44/128), accuracy_rescore 35.9, closed 43.0 (55/128), cap-hit 62.5 (80/128), mean tokens 22,743; by level L1 7/11, L2 15/23, L3 12/27, L4 6/33, L5 4/34; wall 556 s. Paired with pass 3: gained 9, lost 13. Paired with pass 8 (same perturbation): gained 28, lost 7 (p = 0.0005). |
| 10 | R1E2q = R1 + E2 + o_proj columns of layer 11 head 0 x 0.98 (receipt 00bc87ba7c56d89e) | third draw of R1E2's sampling paths; expected 43-50 | accuracy_boxed 35.9 (46/128), closed 44.5 (57/128), cap-hit 60.2 (77/128), mean tokens 21,794; by level L1 9/11, L2 14/23, L3 11/27, L4 9/33, L5 3/34; wall 529 s. Paired with pass 3: gained 9, lost 11; with pass 9: gained 12, lost 10. |
| 11 | R1q = R1 + the same perturbation (receipt e5ddec1b74114bd6) | third draw of R1's sampling paths; 20-45 | accuracy_boxed 23.4 (30/128), closed 25.8 (33/128), cap-hit 75.0 (96/128), mean tokens 27,867; by level L1 6/11, L2 7/23, L3 9/27, L4 5/33, L5 3/34; wall 664 s. Paired with pass 2: lost 18, gained 8; with pass 8: gained 15, lost 8. |
| 12 | E2 alone on the unmodified base (receipt 71bb10cf978dded1) | attribution: does the repetition repair add anything once the stop neuron is present? 38-48 | accuracy_boxed 25.8 (33/128), accuracy_rescore 28.9, closed 30.5 (39/128), cap-hit 74.2 (95/128), mean tokens 25,295; by level L1 6/11, L2 8/23, L3 11/27, L4 7/33, L5 1/34; wall 619 s. 33 closes finish (32 right), 6 leak. Paired with attempt 1 (base): gained 27, lost 1; with attempt 2 (R1 alone): 33 vs 40; with attempt 3 (R1 + E2): lost 20, gained 5 (p = 0.004). |

## Final state (2026-11-27 22:29)

- `~/wsp14/sandbox/model` holds R1 + E2 (the candidate of attempts 3 and 7): fingerprint
  41d6229f334e500ff16b13ff1ee20cc00f1186869eb09bb9f0ad9b0e8c73fc23; checker run again on it at
  22:28: ACCEPTED, 6 tensors changed, receipt on file. Copies: `~/wsp14/work/best_R1E2` (final) and
  `~/wsp14/work/best_R1` (the repetition repair alone, in case the stop neuron is judged out of scope).
- Passes: 11 of 15 charged (attempts 1-5 and 7-12); attempt 6 interrupted and refunded; 4 unused.
- Scores on the 128 practice problems (right by the graded boxed rule): base 7; repetition repair alone
  40, 23, 30 in three functionally identical copies; stop neuron alone 33; both together 48, 44, 46.

### Robustness summary (three functionally identical copies of each, differing only in one unrelated head scaled by 2%)
- Repetition repair alone (passes 2, 8, 11): 40, 23, 30 correct (42, 22, 33 closes): mean 31, spread 17.
- Repetition repair + stop neuron (passes 3, 9, 10): 48, 44, 46 correct (58, 55, 57 closes): mean 46, spread 4.
- Paired, stop neuron against no stop neuron: +8 (p = 0.096), +21 (p = 0.0005), +16 (p = 0.0015).
- Between copies of the stop-neuron model about 22 problems per pair change outcome, symmetrically
  (gained/lost 9/13, 12/10); between copies of the repair-only model 20-28 change, lopsidedly
  (18/8, 21/4, 8/15): each copy has its own close rate. The natural close rate is a near-critical
  property of the weights; the forced closes are not.

### What attempts 6-11 showed about noise (important for every comparison above)
- The engine is exactly deterministic (attempt 7 = attempt 3 byte for byte).
- A next-token divergence of about 1e-4 (one attention head of layer 7 scaled by 2%) re-rolls every
  trace within about a hundred tokens (median first differing token 112). Teacher forcing cannot
  tell R1 from that copy: divergence 0.00013 over the first 1500 tokens, 0.00001 beyond token
  15,000; argmax changed at 0.39% of early and 0.04% of late positions; exit probability at the
  closing rows 0.521 vs 0.523; early-trace style statistics equal. Same for R1E2 vs its copy
  (exit probability at the closing rows 0.860 vs 0.857).
- Yet R1's two draws are 40 and 23 correct (42 and 22 closes), R1E2's are 48 and 44 (58 and 55).
  Natural closing under R1 is a coin flip at the wrap-up row (exit probability median 0.53) in a
  race against falling into a loop, so its score is fragile; the stop neuron's forced closes
  (exit probability median 1.0) are robust. Why R1's two draws differ so lopsidedly (24 closes
  lost, 4 gained) is not explained by independent per-problem re-rolls; open (candidate: the
  loop-entry hazard reacts to tiny changes of the copying strength).

## Findings

### Pass 1: what the failure looks like (text level)
- 121 of 128 generations are cut at the cap with the thinking never closed. Only 7 close; 6 of them right.
- Every cut-off trace has the same two-phase shape:
  1. a semantic loop: the model finds an answer early, then keeps second-guessing with lines that
     start "Wait, ..." ("Wait, did I misread the problem?", "Wait, let me re-read", "Is there a typo?",
     "Let me double check") for thousands of tokens; it even writes "Wait, I am stuck in a loop." and goes on;
  2. a verbatim loop: one line ("Wait, $728$ is closer to 729.", "Wait, $n=17$ works.", "1") repeated
     hundreds to thousands of times to the cap.
- The 7 traces that close have the same style (10-25 "Wait" checks) and end when the sampler picks
  `</think>` after a final statement ("The final answer is -2, 2.", "Final Answer: 2/3.",
  "I'll write out the solution now.").

### Pass 1: teacher-forced reads (base model)
- At the decision point (the token right after a line break) the raw top-1 is nearly always the
  indentation token with logit 28-37; `</think>` sits at 11-19 elsewhere.
- The only decision point of p124 where `</think>` had a real chance is its closing point
  ("The final answer is -2, 2.\n"): `</think>` logit 23.1, protocol probability 0.247, rank 2.
  In the first 6000 tokens of p1 and p27 `</think>` never reached 0.5% at any decision point.
- So the model does not so much "repeat" as "never take the exit": closing is a low-probability
  sample at final-statement line ends, and the longer the trace gets the more "Wait" lines it holds.

### Localization (pooled attribution over 641,869 decision positions of pass 1; `r1/pool1.npz`)
- Raw logit of "Wait" at positions where "Wait" came next, by number of earlier "Wait" tokens:
  0-5: 23.1 | 5-20: 23.9 | 20-100: 27.1 | 100-500: 30.4 | 500+: 30.8. Protocol probability 0.45 -> 1.00.
- Attribution of that growth (+7.7): attn.L19.h0 +5.2, mlp.L22 +3.6, mlp.L23 +3.2, attn.L15.h7 +2.2,
  attn.L23.h5 +1.7, attn.L19.h5 +1.2, attn.L23.h1 +1.0 (others < 0.7; GDN heads < 0.25).
  The five full-attention heads behave like copy heads: their vote for "Wait" grows with the number
  of earlier "Wait" line starts. They are near silent when few "Wait"s precede (0.1 each).
- At line ends the exit token never gets a chance: mean `</think>` logit 14.5-19.3 vs top-1 30.5-34.8
  (the indentation token); `</think>` within 2 of the top-1 in 0.01% of line ends.
- After lines that state an answer ("So the answer is 9.") the gap (top-1 after the presence penalty
  minus `</think>`) has median 14.0 (10%: 10.6). Only 36/128 traces contain such a line in their
  first 8000 tokens.
- No MLP neuron promotes `</think>` specifically after a conclusion line (layer 23 total: 7.5 there
  vs 6.8 at other line ends); the stop pathway is weak and diffuse.

### Causal tests (teacher forcing, heads zeroed at all positions; `logs/abl_test_a.txt`)
- Zeroing the five copy heads: "Wait" logit deep in the loop 31.3 -> 19.8-21.8, still protocol
  probability 0.99-1.00; at few "Wait"s the probability of "Wait" rises (0.48 -> 0.57); ordinary
  reasoning (first 1500 tokens) degraded: KL 0.08-0.11 vs 0.002-0.02 for single heads.
- So removing the copy heads neither unlocks an established loop nor helps early on; the loop is a
  property of the context once it exists. The repair has to act as negative feedback (inhibit "Wait"
  in proportion to how often it has been copied) and the exit itself has to become reachable.

### Teacher-forced validation of R1 (logs/val_R.txt)
- With the "Wait" readout component of the five heads inverted: at few earlier Waits the "Wait" logit
  and probability are unchanged (p1 23.95 -> 23.95); at 20-100 Waits P(Wait) 0.80/0.68/0.95 -> 0.37/0.12/0.09;
  at >= 100 Waits logit 31.3 -> 11.9-12.8 and P(Wait) 0.00; the exit token becomes a candidate at some of
  those positions (p11 P(</think>) 0.00 -> 0.10). Ordinary reasoning (first 1500 tokens) untouched: KL 0.0002-0.0005.
  A half inversion (lam 0.5) leaves the loop locked (P(Wait) 0.90): the full flip is what works.

### Pass 2 (R1 alone): what changed
- 42 traces close (39 right + p86, which closed and was right but ran out of tokens; 2 wrong), median
  length of a closing trace 11,486 tokens. They still wander, but now terminate on lines like
  "Okay, I'm done. / Answer is 9.", "Final Answer: 2010.", "Okay, I'll check the result. / $37$.".
- The 86 that still never close are no longer "Wait" line loops: 78 of them repeat a whole multi-line
  block verbatim (period 8-895 tokens, e.g. "Let's try to see if $x^3 = -1$ is a root. / Sum = $364$." x 179,
  "Maybe the number is 3737?" x 2352); 8 have no exact token loop. So the copy mechanism simply locks
  onto other content once "Wait" is inhibited; a token-specific fix does not generalise.
- Batch-position note: first half of the batch closed 21.9%, second half 43.8% (Fisher p = 0.014-0.035,
  reported by the tool's admission check as valid). To check whether this is problem composition.

## Ideas

### Gain neuron D2 on the model's own exit evidence (built; validation running)
- 32 of the 70 never-closing traces (pass 3) have line ends where the model's own `</think>` logit
  reaches 20.5 or more, often inside the loop itself where it repeats a wrap-up line
  (p33: "I'll write $1, -1, \sqrt{3}, -\sqrt{3}$." x 1000 with exit logit 26.7 against the loop token's 35.2).
  The model is done but cannot act on it. Ordinary line ends: 99th percentile about 19-20.
- The pre-MLP part of the residual does not carry this signal (2.6 ordinary vs 3.6 at the best loop
  rows); it is produced inside the last MLP. A closed-form ridge readout of the model's own `</think>`
  logit from the last MLP's input predicts it with held-out R^2 0.99 (fit on pass-3 activations under R1).
  False alarms (predicted >= 20.5 where the true logit < 17): 0 of 290,568 thinking rows.
- Neuron (layer 23 MLP, neuron 1488): gate = w - (tau - b) o, up = w - (tau - b - 1) o, down = beta v,
  w = ridge readout (unit: logit), o = reference direction, v = readout row of </think>. It adds gain
  that grows with the excess of the model's own evidence over tau = 20.5: +8 at excess 1, +17 at 1.5,
  +29 at 2, +63 at 3. Nothing happens where the model's own evidence is below 20.5.
- Answer phases: in the 49 cleanly finished answer phases no position comes near a flip (one trace
  has three rows at 20.5-21 with the top token 8+ above); in 2-3 of the 9 leaking answer phases
  (model still deliberating after </think>) there are positions where it would flip, i.e. emit a
  second </think> and start a fresh answer.

### Pass 4 (joint word flip S1w + E2): what happened
- 44 correct (pass 3: 48), 54 closed (58). Paired with pass 3: gained 8, lost 12.
- The word-started loops are gone as designed: of the 74 never-closing traces, 70 repeat a block
  none of whose lines starts with a flipped word (bullet + math lines such as "*   $10, 12, 14, 16, 18$.",
  "$a_{n-1} = 2$.", or lines starting with words outside the set). The loop attractor moves to other
  content; breaking loops at their line starts does not make traces close.
- Never-closing traces under R1+E2 (pass 3): 70, of which 39 are Level 5; the visible tails are the
  model stuck on the problem itself (re-deriving the same intermediate step), not a missed exit.

### Pass 3 (R1 + E2): what the stop neuron did
- 58 closes (was 42): 49 finish cleanly (45 right), 9 "leak": the thinking closes but the answer phase
  carries the unresolved doubt on ("However, we must double-check...", "Wait, is it possible ..."),
  never reaches a boxed answer and runs to the cap (2 of the 9 still right thanks to an earlier box).
- Paired with pass 2: gained 13 (mostly traces that now close right after "Okay, so $n=7$.",
  "The largest is 9.", "Then $a+b=12$."), lost 5: 2 by leaks (p46, p85), 3 that simply did not close
  this time (p82, p86, p97: a different trajectory, sampling noise).
- The leaks are not predictable from the model's own exit evidence at the closing position (base
  `</think>` logit: clean mean 16.6, leaky mean 14.9, ranges overlap) nor from the detector strength
  (z 1.5-2.5: 14 clean / 1 leaky; z 2.5-4: 23 clean / 8 leaky). They are the cost of closing while the
  model is not "done": of the 58 closes 12 are natural (0 leak) and 46 forced (9 leak); repair 1
  alone: 1 leak in 42 closes.
- 68 of the 70 never-closing traces never come near a close (max P(</think>) < 0.001 at every position):
  block loops contain no line end the detector recognises.

### Joint word flip S1w (built; validation running)
- Line starts inside the remaining loops: "*" 35%, "$" 15%, "So" 14.5%, "Maybe" 7.9%, "This" 7.4%,
  "Let" 5.7%, "But" 2.1%, "Is" 1.9%, "No" 1.8%, "If" 1.6%, "Okay", "What".
- Flipping the copy heads' output component in the span of twelve line-start words (9 heads) at the
  word-started lines of loops: logit of the loop's line-start word 32.1 -> 7.1, protocol probability
  1.000 -> 0.115, argmax lost 88.5%; ordinary text: probability of the generated token 0.861 -> 0.857, KL 0.023.
  Adding "$" costs more ordinary text (KL 0.063) and is left out.
- The "Wait" readout direction lies mostly inside the span of these words, so stacking the word flip
  and the Wait flip partly cancels; the clean form is one joint flip over the thirteen words (S1w),
  which inverts the Wait component exactly (cosine -1.000 on all five heads).

### Stop-pathway neuron E (built and validated by teacher forcing; not yet measured in a pass)
- The exit pathway exists but is weak: at line ends that state the answer ("So the answer is 9.") the
  `</think>` logit is 14 points (median) behind the formatting continuation.
- In the input of the last MLP there is a linear "concluding line" direction: c = mean(x at commit
  line ends) - mean(x everywhere else), from 91 commit line ends of pass 1 (`probe.py`). Score c.x:
  commit line ends median 25.4; other line ends: 4.5% above 24, 1.4% above 26; never above 18 at
  non-line-end positions (0 of 946,690), in the prompt (max 7.2) or the answer phase (max 11.7).
  Generalises across halves of the data. Its highest-scoring "misses" are other concluding lines
  ("Sum = 150.", "Result: $x^4-16$.", "It seems correct.").
- A reference direction o (closed-form ridge fit of the constant 1 on x over thinking line ends) has
  projection 0.92-1.05 on 98% of thinking line ends, 0.50-0.85 in the prompt, 0.83-1.03 in the answer
  phase: it provides the threshold.
- Neuron (layer 23 MLP, replacing neuron 1998, the least active): gate = gamma (c - theta o), up = c - (theta - 2) o,
  down = b2 * unit readout row of </think> (made orthogonal to its 12 nearest token rows).
  z = c.x - theta * o.x: boost of the </think> logit = 0 for z < 0, then superlinear
  (E2, theta 25, b1 5: +5 at z=1, +16 at z=2, +33 at z=3).
- Teacher-forced check of R1 + E2 on the pass-2 traces (`logs/val_E2_on_R1_p2.txt`):
  ordinary text untouched (protocol probability of the generated token identical to 0.001 over the
  first 1500 tokens; nothing fires in the answer phase or at the last prompt position, max z there -9.7).
  Of the 86 never-closing traces 14 get a first row with P(</think>) >= 0.3; reading them, about 12
  close right after the model has reached the right answer. Of the 39 correct closing traces 35 would
  close earlier (1.2k-3.6k tokens instead of 3k-23k); in the excerpts the answer is already right at
  that point, except p123 (an early arithmetic slip that the long deliberation had repaired).
  Estimated net about +12 -1 correct -> ~40%.

### Full inversion of loop-carrying heads (rejected)
- Inverting five heads with little role in ordinary text (15.h6, 15.h7, 19.h0, 19.h2, 23.h0) on top of R1:
  loop-token probability 0.999 -> 0.47 in the loop tails, but ordinary text damaged (probability of the
  generated token 0.85 -> 0.73, raw argmax agreement 0.87 -> 0.76, KL 0.60). These are the model's
  legitimate copying heads; rejected.
- The model has no strong natural suppression of block repeats: largest opposing contributions in loops
  are about -1 (gdn.L0.h8, attn.L19.h4, attn.L23.h3) against a +10 excess from the attention layers.
- R: invert the "Wait"-direction component of the five copy heads' output (o_proj columns), so that
  repetition of "Wait" inhibits "Wait" instead of reinforcing it.
- E: strengthen the stop pathway: a detector for "the model just stated its answer" writing toward
  `</think>` in the last MLP.
