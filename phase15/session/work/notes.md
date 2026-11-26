# Phase 14 editor notes

Model: qwen3_5 (Qwen3_5ForConditionalGeneration), 24 layers x 1024, hybrid Gated DeltaNet /
full attention (layers 3,7,11,15,19,23), SwiGLU 3584, vocab 248320, tied embeddings.
Hardware: 1 x RTX 5090 (32 GB), 16 CPU cores, 31 GB RAM. Pass budget: 15.

Architecture facts read from the installed library at run time:
- RMSNorm is zero-centred: output = norm(x) * (1 + weight).  Final norm: norm.eps = 1e-6.
- Full attention: q_proj outputs 8 x (256 query + 256 gate) interleaved per head; attention output
  is multiplied by sigmoid(gate) per head, then o_proj (2048 -> 1024). 2 KV heads. Partial rotary 0.25.
- Gated DeltaNet: in_proj_qkv (q 2048, k 2048, v 2048), conv1d (kernel 4), in_proj_z (gate of the
  output norm), in_proj_a / dt_bias / A_log (decay g = -exp(A_log) * softplus(a + dt_bias)),
  in_proj_b (write strength beta = sigmoid), per-head norm (weight * rmsnorm * silu(z)), out_proj.

## Ledger (passes)

| pass | candidate | hypothesis | result | decision |
|---|---|---|---|---|
| 1 | base model (fp 36d3430c9c2c) | baseline | acc 5.5 (7/128), closed 5.5, cap-hit 94.5, mean tok 31286 | reference |
| 2 | g1 (fp 071c042292c351ee) | loop-breaker graft closes the thinking in verbatim loops | acc 46.1 (59/128), rescore 47.7, closed 93.8, cap-hit 34.4, mean tok 18624. L1 9/11 L2 14/23 L3 15/27 L4 16/33 L5 5/34 | KEEP: +40.6 points (~9 SE) |
| 3 | g2 (fp a3de53261dbcd9be) | g1 + runaway-enumeration closer | acc 53.1 (68/128), rescore 54.7, closed 98.4, cap-hit 25.0, mean tok 17769. L1 10/11 L2 17/23 L3 16/27 L4 17/33 L5 8/34 | KEEP: mechanism confirmed (7 of 8 never-closed traces now close, 3 right); net +9 vs pass 2 but most of it is trajectory reshuffle |
| 4 | g3 (fp cde4c2c5907c1a00) | g2 with N1 soft threshold 390 instead of 425 | acc 53.9 (69/128), rescore 54.7, closed 99.2, cap-hit 12.5, mean tok 15560. L1 9/11 L2 19/23 L3 18/27 L4 17/33 L5 6/34 | NOT kept: not clearly better than g2 (+1 problem, within noise) |

**FINAL STATE = g2** (soft threshold 425 + enumeration switch), restored into
`~/wsp14/sandbox/model`, byte-identical to `work/g2_N1_N4`, checker ACCEPTED, fingerprint
`a3de53261dbcd9be` (the candidate measured in pass 3). Passes spent 4 of 15.

### Pass 3 breakdown and paired comparison with pass 2 (cap2.py, paired.py)
- 96 ended: 68 correct (71%). 32 capped: 0 correct = 30 final-answer-phase runaways + 2 never
  closed (p31: an incrementing list N4 did not reach; p57: the repeating decimal 692307...).
- Paired pass 2 -> pass 3: gained 15, lost 6. Of the 8 never-closed in pass 2: 5 stopped (3 right:
  p28, p29, p99), 1 closed-capped, 2 still never closed. Other transitions: closed-capped ->
  stopped 15, stopped -> closed-capped 8 (pure reshuffle). => the true effect of N4 is about +3
  problems; single-pass scores carry +-4.4 points of sampling noise plus reshuffle noise.
- Threshold study for N1 (lowthr.py, thr390.py): soft threshold 390 adds early firings only at
  repetition onsets where the answer is already formed (7 traces); 360 starts closing traces that
  would still recover (p27). Not changed: no demonstrated gain, 425 is validated in two passes.
- Remaining failure bucket: final-answer-phase runaway (30 of 128, mostly L4/L5), for which no
  safe single-token repair was found (see negative results above).

### Further analysis after pass 3 (closer.py, nobox.py, upside.py, fp_single.py, mech.py)
- Which switch closed each pass-3 trace: N1 117 traces (88 ended, 60 right), N4 1-2 traces
  (p26 closed at 17,845 tokens, right; p34 borderline, capped), several traces closed NATURALLY
  (both switches inactive: short clean solutions), 2 never closed. => N4's direct contribution in
  pass 3 was about one problem; most of the 46 -> 53 change was trajectory reshuffle.
- Generations with a boxed answer: pass 2 86, pass 3 84 (stable, ~2/3 of problems). Accuracy
  differs through how many of those are right (59/86 = 69% vs 68/84 = 81%).
- No-box traces in pass 3: 44 (32 capped + 12 ended mid-loop). Gold answer string appears
  somewhere in 16/44 (36%), the same rate as in wrong generations WITH a box (6/16): these are
  problems the model cannot solve. Two were formatting misses (p53: correct interval written as
  $$...$$ without \boxed).
- Single-token test at final-phase runaway positions (21 token/continuation pairs): no forced
  token is naturally followed by "\boxed{" (weak link always <= 0.2). => no lever to force a box.
- Never-closed traces of pass 3: p57 (pure verbatim digit cycle "692307": repetition score ~370)
  and p31 (incrementing list: repetition ~380, enumeration max +23 = just below the flip point).

### Challenger g3 = N1 soft threshold 390 (+ N4), fingerprint cde4c2c5907c1a00 (pass 4)
- Offline: flip point ~405. Extra early/pre-loop firings (7 traces) all at repetition onsets with
  the answer already formed (thr390.py); 106/114 pass-1 loops caught within 800 tokens (vs 97).
- Loss checks (nll2.py, nll_diff.py, earlyrisk.py, hazard.py):
  early text NLL 0.3987 vs 0.3829, but the change is confined to two genuine loops that sit in the
  early window (p16 +1,810 nats, p102 +216); everything else <= 3 nats per 1,500 tokens.
  P(</think>) in the first 600 tokens < 0.01 in all traces. Spurious-closure hazard before the
  first strong repetition: median 0.000, p90 0.09, max 0.90 (g2: 0.000, 0.16, 0.55).
- Decision rule fixed before the pass: keep g2 (validated twice) unless g3 is clearly better.

### Pass 2 breakdown (scripts/cap2.py, postthink.py on gen_2)
- 84 generations end naturally: 59 correct (70%). 44 hit the cap: 0 correct.
- Of the 44: 8 never close = runaway ENUMERATIONS (each line differs by an incrementing number:
  "$32 / 777 = 0.18$", "$32 / 778 = 0.18$", ...; not verbatim, detector silent by design):
  p28 p29 p31 p37 p44 p57 p99 p123.
- 36 close, then the model starts a proper final answer, falls back into the SAME loop while
  writing it; the graft fires again and re-closes the thinking (68 generations contain more than
  one </think>; up to 72 in p48), the answer restarts each time, and no \boxed{} is ever produced.
- => second mechanism needed: in the FINAL-ANSWER phase a detected repetition should produce a
  boxed answer (and not yet another </think>).

### What was measured about the final-answer phase (fp_*.py) -- mostly negative results
- Clean final answers: median 1,960 chars, 0 "Wait"; capped ones: 64,428 chars, 372 "Wait".
- Forcing a box does not work with single-token writes: "\boxed" is 2 tokens ('\\','boxed'); after a
  forced '\\' the model continues with 'boxed' only 7-14% of the time; after a forced paragraph
  break it writes Wait (0.40) / Let (0.35), '\\' only 0.06. After a forced "Therefore" the phrase
  "the final answer is" is unreliable (0.35, 0.00, 0.57, ...). The content the model would put in a
  box at those stuck positions is mostly a wrong reflexive guess (about 2 of 30 right).
- The detector collapses one position after any forced token (d_ls 450 -> 150), so chains of forced
  tokens are impossible without lowering thresholds into ordinary text. => not pursued.
- An answer-mode indicator (closed-form ridge LDA, scripts/mode_lda.py): AUC 0.982 with
  overlapping tails; not clean enough to matter, and blocking "Wait" in answer mode would only
  change the wording of the loop. => not pursued.

### Runaway ENUMERATIONS (never close; 8 of 128 in pass 2) -- a second detector
- Data (enum_collect.py): every 4th position of the last 6,000 tokens of 13 runaway traces
  (pass 2: p28 p29 p31 p37 p44 p57 p99 p123; pass 1: p16 p64 p85 p106 p112) vs every 16th position
  of the first 3,000 generated tokens of all 128 pass-1 traces. x-space of layer 14.
- Direction: closed-form ridge LDA (Sigma + lambda I)^-1 (mu_runaway - mu_fresh), lambda 1e-3
  (results identical for 1e-3 .. 1e-1). One-shot statistic, no gradients/iterations.
- Leave-one-trace-out (enum_an.py): 12 of 13 held-out runaway traces detected with 100% recall at
  ZERO false alarms on 11,753 held-out fresh samples (the 13th is p16, a verbatim loop). The plain
  mean difference reaches only 4/13.
- Scaled so held-out fresh max = -76, runaway min -41, median +24 (out/enum_dir_L14.pt).
- All positions of all pass-1 traces (978k): early max +1 (p50, a quasi-loop), next -12; all
  others <= -17. Final-answer phase: 0 of 84 cleanly ended traces above +10; 33/36 capped above +15.
- Score builds up slowly (enum_onset.py): about -220 at the start, still -92..-184 (median) at the
  onset of a runaway, needs ~20,000 tokens of continuous enumeration to pass +15, +23 near the end.
  => a threshold near +23 cannot be reached by a legitimate list.
- Plan: neuron 2910 of layer 14 = same switch with this direction: soft threshold +5, width 10,
  write 2.0 x the </think> direction (flip-level about +23).

### Edit g1 (accepted by save_edit.py, fingerprint 071c042292c351ee; copy in work/g1_L14n1439_t425)
- One MLP neuron converted: layer 14, neuron 1439 (lowest output energy of the layer; removing it
  changes clean-text NLL by < 0.0001; verified by runtime ablation).
  - layers.14.mlp.gate_proj.weight[1439, :]  = bf16((d - 425 w) / 20)
  - layers.14.mlp.up_proj.weight[1439, :]    = bf16(w)
  - layers.14.mlp.down_proj.weight[:, 1439]  = bf16(5.5 * v)
  d = repetition detector (x space), w = constant direction (x.w ~ 1 +/- 3-5%),
  v = unit(1 + final norm weight) * E[</think>]. Original norms: gate row 0.369, up row 0.222,
  down column 0.203; new: 2.172, 0.0907, 5.500.
- Byte-level patch of the three tensors only (apply_graft.py); checker: 1023 / 1024 / 1024 values changed.
- Real-weight verification (verify_graft.py): problem statements, natural closure positions, fresh
  line ends unchanged (max |dlogit| 0.28); at 10 in-loop flip positions </think> now has
  p = 0.99-1.00 (base: the loop token with p = 1.00), followed by a normal final-answer start.
  Clean-window NLL 0.32219 (base 0.3222).

## Working log

- [start] Read EDITOR_PROMPT.md, harness tools (eval_pass.py, save_edit.py, scorer.py,
  rescore.py, runcheck.py, gpulock.py, benchmark.py), config.json. save_edit.py on the untouched
  working copy: "ACCEPTED: identical to the base model".

### Pass 1 (base model, unmodified) -- MEASURED (gen_1.jsonl, copy in work/out/gen_1.jsonl)
- accuracy_boxed 5.5 (7 of 128), accuracy_rescore 5.5, closed 5.5% (7/128), cap_hit 94.5% (121/128),
  mean_tokens 31286, wall 807 s. by_level: L1 0/11, L2 4/23, L3 3/27, L4 0/33, L5 0/34.
  Correct: p27, p28, p101, p103, p105, p124, p86. p66 closed but wrong format; the rest cut off.
- Failure shape: the model falls into VERBATIM REPETITION LOOPS inside <think> and never gets out.
  Examples: p4 repeats a 5-line cycle ("Wait, let's look at the code again. ...") for ~30k tokens
  (1414 "Wait"); p10 ends in "$3737 = 3737$." repeated forever; p12 repeats one sentence forever.
  Non-looping parts are fluent and sensible (arithmetic and algebra mostly fine).
- Weight statistics (work/out/wstats.txt): per-tensor std/kurtosis/row-norms/singular values are
  smooth across layers. No grossly rescaled/noised tensor. Any defect is finer (rows, heads).
  (Layer 0 linear_attn.in_proj_qkv has one all-zero row and a top singular value 16.7 vs 5-13 in
  the other layers; may be natural for layer 0. To check.)

### Teacher-forced read of p10 (scripts/exp1.py)
- Sane part (first 3k gen tokens): top logit ~30, top1-top2 gap ~5.4, argmax==next 0.91.
- Inside the loop: top logit ~34-41, gap ~16 (e.g. "$3737 = 3737" -> "$." 34.6 vs 28.1; after
  "$3737 = 3737$.\n" -> "   " 41.5, second "</think>" 22.5).
- CORRECTION of my first reading: the presence penalty is a FLAT -1.5 on tokens already present
  (that is what "presence" means; "frequency" would be per count), so it cannot break a loop once
  the raw margin is ~16. At temperature 0.6 a 16-logit margin is deterministic. So the failure is:
  the loop-continuation token wins by an enormous margin; the model "knows" </think> is possible
  (2nd ranked) but is outvoted by ~19 logits.
- Note: transformers runs the reference PyTorch Gated DeltaNet (no FLA kernel) and reference conv1d.

### Loop geometry (scripts/loops.py on pass 1)
- 113 of 128 generations end in an EXACT verbatim cycle (period 6..322 tokens) that starts after a
  median of ~7,000 generated tokens (range 1,400..23,000) and never ends. Cycles typically begin
  with a 'Wait, ...' line. Before the cycle there are typically 30-100 'Wait's (second-guessing).
  p21 and p118 even contain the sentence 'Wait, I'm just repeating myself.' inside the cycle.
- Corrected logits (model head): sane part top logit ~28, gap ~5; loop top ~37, gap ~15;
  </think> logit early ~1.4, late ~8 (2nd ranked at line end: 22 vs 41).
- Library caveat (transformers 5.20): last_hidden_state is post-final-norm; output_hidden_states is
  not the raw residual. lib.py captures the residual stream with own hooks.
- Next: census of every head (norm, gates, decay, attention entropy) for an abnormal component.

### Census (scripts/census.py, 6 clean texts x 1200 tokens, eager attention; out/census_base.txt)
- Full attention: no head attends much to the previous token (max 7%) or to position 0 (max 13%).
  Most heads are diffuse (entropy 3-5 nats); sharp: L23 h2 (0.73), L23 h6 (1.33). Gate means
  0.1-0.4. Output contributions grow with depth (L23 h2 2.4, h6 2.0; others ~1).
- GDN: median per-step decay 0.2-1.0 spread over heads, write strength 0.02-0.99; nothing pinned.
- Structural integrity (scripts/integrity.py): no exact zero rows/cols; MLP gate/up rows
  uncorrelated in ALL 24 layers alike (family property, not a damaged layer); OV pairing test
  inconclusive/no clear mispairing; q_proj query/gate rows consistent. => not a permutation.

### KEY FINDING: the checkpoint is the unmodified public model
- Download metadata inside the model dir (.cache/huggingface/download/*.metadata, trees/*.json)
  records commit 2fc06364715b967f1860aea9cf38778875588b17 and LFS sha256
  04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696 for the weight file.
  sha256sum of base_model and model weight files: 04b1c301...4696 = IDENTICAL.
  => no injected weight corruption. The 14 rows of +-2^-123 in L0 in_proj_qkv (+ their 56 conv
  taps; q/k head 3 dims 13,39,54,69,86,92,95) are part of the vendor release.
- So the failure mechanism is the model's OWN: verbatim repetition loops in thinking mode
  (known weakness of small Qwen3 thinking models; the protocol's presence penalty 1.5 is the
  vendor's recommended mitigatiоn and is too weak: flat -1.5 vs a raw loop margin of ~15).
- Implication: nothing to restore. The repair must change how the intact network behaves in
  repeated contexts: cut the repetition mechanism or strengthen a loop-escape mechanism, without
  a global sharpening and without unconditional early closing.

### Head lesion sweep (scripts/lesions.py, 64 clean windows, baseline NLL 0.3222; out/lesions_base_heads.json)
- Every attention head is useful when removed at runtime (dNLL > 0), except L3 h2 (-0.0025, unused).
  Most important: L3 h7 +0.33, L7 h2 +0.16, L11 h3/h4 +0.08/+0.10.
- GDN heads: all useful; L0 h8 +1.01 (!), L0 h3 +0.16, L2 h11 +0.07; a few ~0 (L4 h11 -0.002).
  => intact model: nothing harmful to remove. (Sweep stopped after the heads; MLP groups not run.)

### Line-start data (scripts/linestart.py -> out/linestart_1.pt; analysis lsanalyze*.py)
- Line-start decision positions (token before the first real token of a line), all 128 traces:
  10,754 early / 16,652 late (last 2,500 tokens before the loop) / 14,878 in-loop.
  In loops: first word top-1 prob median 1.00 (97% > 0.9), 49% of loop lines start with the word
  Wait, 45% duplicate the previous line (12 tokens), 69% share the first token with the previous line.
  Fresh line starts: top-1 median 0.46-0.57, 9-18% start with Wait, 2-6% duplicates.
- A mean-difference residual direction (LOOP minus FRESH line starts; trained on even traces) gives
  AUC 0.96-0.98 on odd traces at layers 10-14, direction stable (cos 0.97-0.98 between halves).
  => the network has a linearly readable 'this line start is in a repeated context' signal.
- Top 'fresh' scorers are all from p0 (quasi-repetitive before its loop) => not real false alarms.
- Engineering constraint: a fresh trace has ~300 line starts, so a per-line-start false-alarm rate
  above ~0.01% compounds into a spurious trigger in most traces. The trigger must be very specific,
  or its action harmless when wrong.
- Lesson: never run two of my GPU jobs at once (contention made 33K passes ~15x slower). Alone, a
  32K-token teacher-forced pass takes 2.4 s.

### Detector for 'line start in a verbatim-repeating context' (xpos.py, xpos_an.py, det_an.py)
- Collected x = post_attention_layernorm output (what an MLP neuron sees) for layers 13, 14 at ALL
  positions of ALL 128 traces (out/xpos_1.pt).
- Mean-difference direction d (LOOP line starts minus FRESH line starts), trained on odd-id traces,
  evaluated on even-id traces over every position, regions early / pre-loop (last 2,500 tokens
  before the loop) / loop:
    layer 13: loop line-start score q25/q50/q75/q90 = 279/394/458/487.
      above q75: early 333/364,479 (0.09%), pre 1/141,278, loop 1,898/46,398 (4.1%)
    layer 14: q25/q50/q75/q90 = 321/421/479/502.
      above q75: early 9/364,479, pre 0, loop 1,577/46,398 (3.4%); above q90: early 0, loop 1.3%
  The few early positions above are all in p50/p16/p8, traces that were genuinely recycling the
  same lines long before an exact cycle formed. Highest legit-repetition scores: case enumeration
  (p106) 306; repeated Asymptote code inside problem statements (p110, p38, p4) 246-262.
- The model's own sentence "Wait, I'm stuck in a loop" (p8) sits at the very position d scores highest.
- Constant direction (needed because the MLP has no bias): w = (Sigma + 0.05 tr(Sigma)/1024 I)^-1 xbar
  (closed form, fresh positions) has projection 1 +/- 3-5% at 98% of positions in all regions.
- Write calibration (writecal.py, hooks): a unit write along (1+w_final)*E[</think>] added at the
  output of layer 13/22 raises the </think> logit by +7/+10 per unit (saturating near +60-70).
  In loops </think> trails the natural top token by 25-30 logits => a write of 4-6 units flips it.
- Natural conclude phrases do NOT rescue a loop: after a full wrap-up phrase at an in-loop line
  start P(</think>) is only 3-13% (wraptest.py); "I"/"Final"/"So" lead back into checking.
  => the action must make </think> itself win at detected positions.
- Design: convert one dispensable MLP neuron of layer 14 into a conditional switch:
  gate row = d/s (repetition detector, soft-thresholded via the constant direction),
  up row = constant direction (gives a constant positive factor),
  down column = c * unit(1+w_final)*E[</think>]  ->  writes 'close the thinking' only when the
  context is verbatim-repeating. Nothing changes elsewhere (gate far below threshold).

### Parameters and emulation (graft_eval.py, emul.py with runtime hooks)
- Final direction d from ALL traces (build_dirs.py, layer 14): loop line-start score quantiles
  q50 399, q70 445, q75 454, q80 463, q90 480. Legit content stays far below (enumeration 311,
  repeated Asymptote in problem statements ~262).
- Full neuron response a = silu((x.d - theta*x.w)/s) * x.w, write = a*c, evaluated on all
  positions with theta 425, s 20, c 5.5:
     flip-level (write >= 3.5): early 118 / 695,391 (all in p50, a genuine quasi-loop),
     pre-loop 59 (4 traces), loop 2,304 positions; 97 of 114 loops within their first 800 tokens.
    (theta 445: 0 early, 68/114 loops; theta 460: 0 early, 45/114.)
  The response is graded: barely above the soft threshold the write is far too small to flip
  anything, so the effective flip score is ~448.
- Emulation on held-out traces: at natural closure positions and fresh line ends nothing changes
  (write 0). At the first flip-level position of 10 loops the base top token has p = 1.00 and
  with the graft </think> has p = 0.99-1.00 (write 3.6-4.8); write 3.1 gives p = 0.22.
- After the forced </think> the detector score collapses (450 -> -69..69) and the neuron is
  silent; the model continues normally (newline, then The/To, then a boxed expression).
  => no </think> repetition loop.
### Pass 4 and the last offline checks
- Pass 4 (g3): 69/128 = 53.9; 87 boxed; 1 never closed; 16 capped + 26 ended without a box
  (42 without an answer, the same bucket as pass 3). Versus g2: +1 problem. Pre-registered rule
  -> incumbent g2 kept. Copies: work/g1_L14n1439_t425, work/g2_N1_N4 (FINAL), work/g3_N1t390_N4.
- Early closures (< 2,500 tokens) in passes 3 and 4 are short clean solutions (p95, p74, right)
  or genuine loops (p102, wrong either way): no sign of false triggers.
- Repetition score vs number of earlier verbatim occurrences of a line (repcurve.py): median
  -50 (first), -21, -5, +6, +24, +34, +54, +64, +100 (ninth), 390 inside loops; wide spread at
  every count => a gentler switch at moderate scores cannot tell a third repeat from a first-time
  line, and blocking the word Wait would only reword a re-check. Not pursued.
- Amplifying the linear answer-mode feature at final-phase doubt positions (amp_mode.py): readout
  shifted by up to +6.4 but P(Wait) only 0.47 -> 0.42. The feature marks the phase, it does not
  control decisiveness. Not pursued.
- Conclusion: the remaining bucket (final-answer-phase runaway, ~1/3 of problems, mostly L4/L5)
  is a capability limit; no targeted weight edit with a safe single-position action was found.

### Honesty notes on the evidence
- Base -> grafted: +47.6 points, far beyond noise (SE of one pass ~4.4).
- g1 -> g2: +7.0 points on the practice set = 1.6 SE; N4's direct contribution in pass 3 was
  about 1 problem. It is kept on mechanism evidence (closes runaway enumerations that otherwise
  never close; zero false alarms on 978k fresh positions and on all cleanly ended final answers;
  clean-text loss unchanged), not on the pass difference.
- g2 -> g3: +0.8 points, noise. Not kept.
