# Final report — Phase 16, session B (the told briefing)

Written 2026-11-27 after 11 charged passes (4 of the 15 unused).

## Part 1 — Plain and short

**What was wrong.** The model failed at the practice problems almost entirely because it never stopped
thinking. In 121 of the 128 problems it repeated itself until it was cut off at the token limit, so
only 7 problems were scored right.

**What I changed.** Two small weight edits, both inside the text model:

1. *Repetition repair.* I found five small units (attention heads) whose vote for the word "Wait"
   grows the more often the model has already started a line with "Wait". I inverted that vote, so
   repetition now brakes instead of feeding itself.
2. *Stop repair.* The model's signal for "I am finished, end the thinking" exists but is far too weak:
   after it writes down an answer, the end-of-thinking token is about 14 score points behind the
   token that continues the text. I added one detector unit in the last layer that raises the
   end-of-thinking token when a line has just stated a result.

**What happened (128 practice problems, right answers):**

| weights | right of 128 |
|---|---|
| unmodified | 7 |
| repetition repair alone (three functionally identical copies) | 40, 23, 30 |
| stop repair alone (one measurement) | 33 |
| both repairs (three functionally identical copies) | **48, 44, 46** |

The two repairs help about equally on their own and work together. The repetition repair alone is
unreliable (its score swings between 23 and 40 for changes so small that nothing I can measure
distinguishes the copies); with the stop repair the score is stable.

**What to expect on the 500 held-out problems.** About 36 of 100 right with the final weights
(a fair range is 30 to 42), against about 5 of 100 for the unmodified model. With the final weights
about 6 of 10 traces still never stop; those are mostly the hardest problems, where the model is
stuck on the mathematics itself rather than on the exit.

**What is needed from you: one decision.** The final weights contain both repairs. The stop repair
ends the thinking when the model states a result. It is targeted (it reacts strongly to 6 of every 10
lines that state the answer, to 1 or 2 of every 100 other line ends, and to nothing inside a line; and
in 68 of the 70 problems that still never stop it never fires at all), but it is a
change to *when the thinking ends*, which the briefing treats as a matter of scope. If you judge it
out of scope, the repetition repair alone is saved beside the final weights (expect about 24 of 100,
and expect it to be unstable). Nothing else is needed.

---

## Part 2 — The technicals, in time order

Terms used below, in everyday words first:

- *pass*: one run of the 128 practice problems under the fixed protocol (temperature 0.6, top-p 0.95,
  top-k 20, presence penalty 1.5, at most 32,768 new tokens). The engine is exactly deterministic, so
  the same weights always give the same pass.
- *score* (technical: logit): the number the model gives each possible next word before turning the
  numbers into probabilities. A difference of 1 point is small; 5 points usually decides the word.
- *teacher-forced read*: running the model over a fixed text and recording what it computes at each
  position. It generates nothing and costs no pass. My reads reproduce the engine: at about 9 of 10
  positions the most likely word under the protocol is the word the engine actually generated.
- *attribution* (technical: direct logit attribution): splitting a word's score into the parts written
  by each unit of the network. I checked that the parts add up to the true score.
- *attention head*: one of the small parallel units in the six full-attention layers (3, 7, 11, 15, 19,
  23; eight heads each). The other 18 layers use a different mechanism (linear attention).
- *copy head*: an attention head whose vote for a word grows with how often that word has appeared in
  the same place before, which is what makes a repeated line repeat again.

Attempt numbers below are the tool's own (files `gen_<attempt>.jsonl`, `eval_pass_<attempt>.json`).

### Step 1. Measure the starting point (attempt 1)

7 of 128 right; 121 of 128 cut at the cap with the thinking never closed; mean length 31,286 tokens.
By level: Level 1: 0 of 11, Level 2: 4 of 23, Level 3: 3 of 27, Level 4: 0 of 33, Level 5: 0 of 34.

### Step 2. Read the failures

Every cut-off trace has the same two-phase shape. First a self-doubt loop: the model finds an answer
early, then keeps second-guessing with lines that start "Wait, ..." ("Wait, did I misread the
problem?", "Wait, let me re-read the question"). It even writes "Wait, I am stuck in a loop." and goes
on. Then a verbatim loop: one line ("Wait, $728$ is closer to 729.", "Wait, $n=17$ works.") repeated
hundreds to thousands of times. The 7 traces that end do so when the sampler happens to pick the
end-of-thinking token after a final statement.

### Step 3. Find where it comes from (teacher-forced reads on all 128 traces)

*The self-reinforcing "Wait".* At positions where "Wait" came next, its score rises with the number of
earlier "Wait" tokens: 23.1 (0-5 earlier), 23.9 (5-20), 27.1 (20-100), 30.4 (100-500), 30.8 (500 or
more). Under the protocol its probability goes from 0.45 to 1.00. The attribution of that growth, over
641,869 positions, names five attention heads: layer 19 head 0 (+5.2), layer 15 head 7 (+2.2), layer 23
head 5 (+1.7), layer 19 head 5 (+1.2), layer 23 head 1 (+1.0). Two late MLP blocks add more (+3.6 in
layer 22, +3.2 in layer 23) and some others subtract. The heads of the linear-attention layers each
contribute less than 0.25. The five heads are almost silent when few "Wait"s precede (about 0.1 each).
They are copy heads.

*The missing exit.* At line ends the end-of-thinking token scores 14 to 19 against 30 to 35 for the
indentation token that continues the text. It comes within 2 points of the top at 1 of 10,000 line
ends. After a line that states the answer ("So the answer is 9.") it is still 14 points behind
(median). The model's own "finished" signal exists but is weak and spread thin: no unit in the last
MLP promotes the end-of-thinking token specifically after a conclusion (7.5 there against 6.8 at other
line ends for the whole block).

### Step 4. Causal tests and the repetition repair

Zeroing the five copy heads does not free an established loop: deep in a loop the "Wait" probability
stays at 0.99 to 1.00, and ordinary text suffers (the probability of the generated word drops from
0.87 to 0.83 in the first 1,500 tokens of a test trace). So the repair has to turn the positive
feedback into negative feedback rather than remove the heads.

**Repair 1 (called R1 below).** For each of the five heads, in the matrix that writes the head's result
back into the network (`o_proj`), take the part that points at the words "Wait" and " Wait" and invert
its sign. Everything else the head does is left alone.

Teacher-forced check: with 100 or more earlier "Wait"s the "Wait" probability goes from 1.00 to 0.00;
with 20 to 100 it goes from 0.80, 0.68, 0.95 to 0.37, 0.12, 0.09 (three traces); with fewer than 20 it
is unchanged; ordinary text is untouched (difference between the two next-word distributions:
0.0002 to 0.0005). An inversion of only half strength leaves the loop locked.

### Step 5. What repair 1 did (attempt 2): 40 of 128 right

42 traces now end (was 7), 87 are still cut off. They still wander, but they end on lines like
"Okay, I'm done. / Answer is 9." and "Final Answer: 2010.".

### Step 6. What is left, and why it cannot be repaired the same way

The 86 traces that still never end are no longer "Wait" loops. In 78 of them a block of text
repeats verbatim ("Let's try to see if $x^3 = -1$ is a root. / Sum = $364$." 179 times). The same five
heads carry these loops: in loops the attention layers add 20.6 points to the repeated word against
10.2 in ordinary text. But now the repeated words are arbitrary, so a fix aimed at one word does not
apply. Two ways to generalise were tested and rejected:

- Inverting five loop-carrying heads completely: breaks loops under teacher forcing (probability 0.999
  to 0.47) but damages ordinary text (in the test trace the probability of the generated word fell
  from 0.87 to 0.73). These are the model's legitimate copying heads.
- Inverting the heads' vote for thirteen common line-start words ("So", "Maybe", "This", "Let", ...):
  under teacher forcing the loop's line-start word drops from probability 1.00 to 0.12 at almost no
  cost to ordinary text (0.861 to 0.857), but in a pass (attempt 4, on top of repair 2) the loops simply
  moved to lines that start with bullets or mathematics: 44 of 128 right, not more traces ending.

### Step 7. The stop repair (called E2 below) and attempt 3: 48 of 128 right

*Design.* In the input of the very last MLP there is a direction that separates line ends which state a
answer from everything else (the difference of the two average inputs, from 91 such line ends). Along
it, 74 of the 91 lines that state the answer (81%) score above 24 and 56 of the 91 (61%) above 26;
of the other line ends in the thinking 4.6% and 1.4% do; nothing inside a line reaches 22 (0 of
946,690 positions), the prompt reaches at most 7.2 and the answer phase at most 11.7. A second
direction, solved in closed form, has a projection of about 1 on line ends and provides the threshold
(25). The direction generalized between disjoint halves of the data (fitted on one half, it
separated as well on the other); richer fitted probes did not. I replaced the least active unit
(neuron 1998) of the last MLP by
a detector built from these two directions that writes toward the end-of-thinking token: it adds 5
points to that token's score at a detector margin of 1, 16 at 2, 33 at 3, and nothing below the
threshold.

*Result (attempt 3).* 58 traces end (was 42); 49 of them finish cleanly (45 right) and 9 *leak*: the
thinking ends but the answer that follows carries the unresolved doubt on ("However, we must
double-check ...", "Wait, is it possible ..."), never reaches a boxed answer and runs to the cap.
Against attempt 2, problem by problem: 13 gained, 5 lost (2 lost to leaks, 3 simply took a different
path and did not end). A difference this lopsided arises by chance about 1 time in 10, so on its own
this pass does not settle repair 2.

*What repair 2 cannot do.* In 68 of the 70 traces that still never end, its detector never comes near
firing (probability of the end token below 0.001 at every position). Those traces are stuck on the
problem (39 of the 70 are Level 5) and contain no concluding line.

*The leak.* It is the cost of ending the thinking while the model is not "done": natural ends leak
0 of 12, forced ends 9 of 46. I could not predict which forced ends leak, neither from the model's own
exit evidence at that position (clean 16.6 against leaky 14.9 points, ranges overlap) nor from the
detector strength.

### Step 8. Further attempts that did not help

- *A gain unit on the model's own exit evidence (attempt 5).* The model's own end-token score can be
  predicted from the last MLP's input almost perfectly (a closed-form linear readout explains 99% of its
  variation on held-out traces). Where that own score was already high, including inside loops that
  repeat a wrap-up line, I added gain to the end token. It ended 84 traces, but 27 of them leaked
  (against 9 of 58 in attempt 3): 46 right, no gain. The model's exit evidence is also high where it is
  not done.
- *Other doses of repair 2 (teacher forcing only, no pass spent).* A more sensitive dose adds 15 weak
  triggers among stuck traces but starts firing in the answer phase; a more conservative dose keeps most
  of the leaking ends and loses clean ones.

### Step 9. How stable the numbers are (attempts 7 to 12)

Attempt 7 re-ran the weights of attempt 3 exactly and reproduced them byte for byte, so a pass has no
run-to-run noise. To see how much the score depends on the sampling path, I made functionally
identical copies of each finalist, changing one unrelated attention head by 2%. Teacher forcing cannot
tell a copy from its original (difference between next-word distributions 0.0001 early in the trace and
0.00001 late; the most likely word changes at 4 of 1,000 early positions and fewer than 1 of 1,000
late ones; exit probability at the closing position 0.521 against 0.523). Yet the texts part within
about a hundred tokens on every problem, and:

- repair 1 alone: 40, 23 and 30 right (attempts 2, 8, 11) — 42, 22 and 33 traces end;
- both repairs: 48, 44 and 46 right (attempts 3, 9, 10) — 58, 55 and 57 traces end.

The repair-1 model ends its thinking by a coin flip at the wrap-up line (exit probability median
0.53) in a race against falling into a loop, so its score is fragile; the forced ends of repair 2
(exit probability median 1.0) are stable. Between copies of the repair-1 model, 23 to 26 problems per
pair change outcome, lopsidedly (18 lost against 8 gained, 21 against 4, 15 against 8): each copy has
its own ending rate. I did not find what in the weights sets that rate. Problem by problem, repair 2
against no repair 2 gives +8, +21 and +16 right.

Attempt 12 measured repair 2 alone on the unmodified model: 33 of 128 right (39 end, 6 leak); against
the unmodified model 27 gained and 1 lost. So each repair alone lands near 31 to 33, and together they
give 44 to 48.

---

## The exact final edit

Final weights: `~/wsp14/sandbox/model` (candidate of attempts 3 and 7). Checker run on it after the
last pass: ACCEPTED, 6 tensors changed; receipt on file; fingerprint
`41d6229f334e500ff16b13ff1ee20cc00f1186869eb09bb9f0ad9b0e8c73fc23`. Copies:
`~/wsp14/work/best_R1E2` (final) and `~/wsp14/work/best_R1` (repair 1 alone).

No tensor was added, removed, reshaped or re-typed; config, tokenizer and chat template are the
base's. No backward pass, autograd, optimizer or iterative fitting was used: every vector below is a
difference of means, a closed-form least-squares solution, or a direct computation, and every forward
pass ran without gradients.

### Repair 1 — invert the "Wait" component of five copy heads

Tensors (bf16, shape [1024, 2048]): `model.language_model.layers.15.self_attn.o_proj.weight`,
`model.language_model.layers.19.self_attn.o_proj.weight`,
`model.language_model.layers.23.self_attn.o_proj.weight`.

Operation, per head in {layer 19 head 0, layer 15 head 7, layer 23 head 5, layer 19 head 5, layer 23
head 1}: with `W` the head's block of 256 columns, `W ← W − 2·u·(uᵀW)`, where `u` is the unit vector
along the mean of the normalised readout rows of tokens "Wait" (13784) and " Wait" (13428). Readout
row = embedding row times (1 + final-norm weight), because the embeddings are tied to the output and
the final norm is zero-centred.

Magnitudes: the size of the "Wait"-pointing part of each head is unchanged and its sign reversed:
0.376 (layer 19 head 0), 0.186 (layer 15 head 7), 0.198 (layer 23 head 5), 0.198 (layer 19 head 5),
0.219 (layer 23 head 1). Per tensor: layer 15 — 233,020 of 2,097,152 values changed, size of the
change 0.373 against a tensor size of 20.04, largest single change 0.0098; layer 19 — 471,568 values,
0.851 against 24.21, largest 0.0311; layer 23 — 470,828 values, 0.590 against 20.06, largest 0.0174.

### Repair 2 — a stop detector in the last MLP

Tensors (bf16): `model.language_model.layers.23.mlp.gate_proj.weight`, row 1998;
`model.language_model.layers.23.mlp.up_proj.weight`, row 1998;
`model.language_model.layers.23.mlp.down_proj.weight`, column 1998.

Operation: the three weight vectors of neuron 1998 (the least active neuron of that MLP: mean
activation times output size 0.0084 against a median of 0.021) are replaced:

- gate row ← `c − 25·o` (length 1.232; before 1.246)
- up row ← `c − 23·o` (length 1.177; before 0.445)
- down column ← `0.3554·v` (length 0.355; before 0.327)

with

- `x` the input of the last MLP (residual after layer 23's attention, normalised);
- `c` the unit difference of means of `x` between line ends that state the answer in the thinking
  (91 positions in the unmodified model's attempt-1 traces; the preceding line contains "answer is",
  "answer should be", "answer:" or "final answer", ends with a period, and has no hypothetical or
  meta wording) and all other positions (1,012,049 positions in total);
- `o` the closed-form ridge solution (penalty 0.001 times the trace of the covariance) of fitting the
  constant 1 from `x` over line ends in the thinking, scaled so that its median projection is 1;
- `v` the readout row of the end-of-thinking token (248069), made orthogonal to the readout rows of
  its 12 nearest other tokens and normalised.

The neuron adds `silu(gate·x) · (up·x) · 0.3554·v` to the residual. With `z = c·x − 25·(o·x)` the
end-of-thinking score rises by 0 for `z ≤ 0`, then by 5 points at `z = 1`, 16 at 2, 33 at 3, 54 at 4.

Magnitudes: gate_proj — 1,022 of 3,670,016 values changed, size of the change 1.616 against a tensor
size of 36.18, largest single change 0.302; up_proj — 1,024 values, 1.260 against 21.25, largest
0.280; down_proj — 1,023 values, 0.486 against 16.42, largest 0.0647.

---

## The passes

Charged: 11 of 15 (attempts 1-5 and 7-12). Attempt 6 was charged and refunded without a measurement
(see below). 4 passes unused: the remaining ideas were tested by teacher forcing and found not worth
a pass, and the remaining failures are the hardest problems.

| attempt | weights | right of 128 | traces that end | cut at the cap | mean tokens |
|---|---|---|---|---|---|
| 1 | unmodified | 7 | 7 | 121 | 31,286 |
| 2 | repair 1 | 40 | 42 | 87 | 25,995 |
| 3 | repair 1 + repair 2 (**final**) | 48 | 58 (9 leak) | 79 | 22,700 |
| 4 | thirteen-word flip + repair 2 | 44 | 54 (6 leak) | 80 | 22,584 |
| 5 | repair 1 + repair 2 + gain unit | 46 | 84 (27 leak) | 71 | 21,351 |
| 6 | repair 1 + repair 2 | interrupted, refunded | — | — | — |
| 7 | repair 1 + repair 2, exact re-run | 48 (identical to attempt 3) | 58 | 79 | 22,700 |
| 8 | repair 1, copy with one head scaled by 1.02 | 23 | 22 | 107 | 29,376 |
| 9 | repair 1 + repair 2, same copy | 44 | 55 (7 leak) | 80 | 22,743 |
| 10 | repair 1 + repair 2, copy with another head scaled by 0.98 | 46 | 57 (6 leak) | 77 | 21,794 |
| 11 | repair 1, same copy | 30 | 33 | 96 | 27,867 |
| 12 | repair 2 alone | 33 | 39 (6 leak) | 95 | 25,295 |

By level (right of n), for the unmodified model and summed over the three copies of the final weights:

| | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
|---|---|---|---|---|---|
| unmodified (attempt 1) | 0 of 11 | 4 of 23 | 3 of 27 | 0 of 33 | 0 of 34 |
| final weights (attempts 3, 9, 10) | 23 of 33 | 46 of 69 | 30 of 81 | 29 of 99 | 10 of 102 |

The tool's batch-position check passed on every measured pass. In attempt 2 the first half of the
batch ended less often than the second (14 of 64 against 28 of 64); the halves have the same level mix
but different topics, and in the later passes the halves were similar.

Problem-by-problem changes, repair 2 against no repair 2 on the same background: attempt 3 against 2:
13 gained, 5 lost; attempt 9 against 8: 28 gained, 7 lost; attempt 10 against 11: 20 gained, 4 lost.
Repair 1 against nothing, with repair 2 present: attempt 3 against 12: 20 gained, 5 lost.

## Candidates the checker refused

None. Every candidate I built was accepted: attempts 2, 3, 4, 5, 8, 9, 10, 11, 12 and the re-check of
the final weights.

## Process note: attempt 6

I launched attempt 6 with an empty hypothesis by my own mistake: the file holding the hypothesis had
not been written after a transient error reaching the working directory, and my launcher did not
check for it. I stopped it 3 seconds after it was charged, before any generation. The tool recorded
"INTERRUPTED, pass refunded" and returned the pass. Its ledger keeps the row with the empty hypothesis;
I did not edit anything under `harness/`. The launcher now refuses to start without a non-empty
hypothesis and note.

## Predicted effect on the held-out set (MATH-500)

- **Final weights (repair 1 + repair 2): about 36 of 100 right, roughly 180 of 500.** Basis: the three
  functionally identical copies scored 48, 44 and 46 of 128 (35.9% summed). By level: 70 of 100 at
  Level 1, 67 at Level 2, 37 at Level 3, 29 at Level 4, 10 at Level 5. The practice set's level mix
  (11, 23, 27, 33, 34) matches MATH-500's (43, 90, 105, 128, 134, from general knowledge; not checked
  here), so weighting by level gives the same figure. Fair range: 30 to 42 of 100. It allows for the
  sampling of 500 problems (about 2 points either way), for the spread between the copies (34 to 37.5
  on the practice set), and for unknown differences between the training-set practice problems and
  the test set. About 6 of 10 traces will still be cut at the cap.
- **Unmodified model: about 5 of 100** (7 of 128 on the practice set).
- **Repair 1 alone: about 24 of 100, and unstable** (its three copies scored 23, 30 and 40 of 128,
  that is 18, 23 and 31 of 100).
- **Repair 2 alone: about 26 of 100** (33 of 128, one measurement).

## What is not verified, and limits

- The held-out set was never run; everything above comes from the 128 practice problems.
- I did not find what makes the ending rate of the repair-1 model swing between functionally
  identical copies (23 to 40 of 128). It is why I do not recommend that repair alone.
- The leak after a forced end (6 to 9 of about 56 ends per pass) is not predicted by anything I
  measured and is not repaired.
- Of the 128 problems, about 70 still never end with the final weights. They are concentrated in
  Levels 4 and 5 and look like the model being stuck on the mathematics.
- To load the text model with transformers for the teacher-forced reads I printed the installed
  library's own model code (in the named Python environment). Nothing else outside `~/wsp14/` was read,
  and nothing was fetched from the internet.

Working notes, scripts and all logs are in `~/wsp14/work/` (`notes.md` is the running log).
