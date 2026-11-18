# Phase 11 weight-editor journal

Session goal: improve MATH-500 accuracy of `sandbox/model` by direct weight arithmetic
(no gradients, no training). 10 official passes max; pass 1 = unmodified baseline.

## Environment recon (2026-11-18)

- WSL Ubuntu-22.04, venv `~/wsp3c-venv` (torch 2.14.0+cu130, vLLM 0.33.0, transformers 5.20.2), RTX 5090.
- Checkpoint `~/wsp11/sandbox/model`: single shard `model.safetensors` (8.77 GB), bf16.
- config.json: `Gemma4ForConditionalGeneration`, model_type gemma4. Text side:
  - 30 layers, hidden 2816, 16 q heads (head_dim 256, global_head_dim 512), 8 kv heads (num_global_key_value_heads 2), `attention_k_eq_v: true`.
  - MoE enabled: num_experts 8, top_k_experts 8 (all experts active), moe_intermediate_size 704, intermediate_size 2112 (dense MLP size for the non-MoE path / vision?), hidden_activation gelu_pytorch_tanh.
  - vocab 262144, tie_word_embeddings true, final_logit_softcapping 30.0, rms_norm_eps 1e-6.
  - layer_types: sliding attention (window 1024) with full_attention at layers 5, 11, 17, 23, 29 (0-indexed).
  - rope: sliding uses theta 10000 default; full_attention uses theta 1e6 "proportional", partial_rotary_factor 0.25.
  - generation_config: do_sample true, temperature 1.0, top_k 64, top_p 0.95, eos [1, 106, 50].
- Rough param budget: ~4.3-4.5 B params (embeddings 262144x2816 tied = 0.74 B alone).

## Plan

1. Pass 1 (baseline, unmodified) launched immediately in background.
2. While it runs: CPU-only safetensors header inspection (tensor names, shapes, stats).
3. After pass 1: probe generations to characterize the failure mode (formatting? arithmetic? cap hits?).
4. Convert hypotheses into official passes; edit weights between passes.
5. Finish: install best candidate, write final_report.md.

## Findings

### Baseline pass 1 (unmodified) — OFFICIAL
```
accuracy 1.6 | answered 92.2 | cap_hit_rate 98.4 | mean_tokens 8183.1
decoding: t=1.0 p=0.95 k=64 seed=20261116 | n=64 | ~6 min wall
```
Failure mode signature: nearly every generation runs to the 8192-token cap
(98.4%). The model essentially never terminates. 1.6% accuracy. Whether the
rambling content is sane (stop-token suppressed) or degenerate (repetition loop)
needs probes.

### CPU tensor forensics (so far)
- 1013 tensors. Per layer: dense MLP (2112) + MoE block (8 experts x 704,
  gate_up fused (8,1408,2816), down (8,2816,704)); router: proj (8,2816),
  per-dim scale (2816,) ~32, per_expert_scale (8,) ~1.0.
- layer_scalar (1,) per layer: 0.07..0.82, smooth trend (L29=0.195 low vs L28=0.68).
- q_norm / k_norm are BIT-CONSTANT per layer (1 unique value across 256 dims):
  q~0.95-1.04, k~0.12 (sliding layers) / ~0.06 (full-attn layers 5,11,17,23,29).
  Could be a scalar-gain design (broadcast at export) — not conclusive.
- Final norm: mean 29.3, max 588 (Gemma-style outlier dims plausible).
- v_proj missing on exactly 5 layers (= the full_attention layers, k_eq_v design).

### Tokenizer / generation format (from tokenizer_config.json + tokenizer)
- Agentic channel format: assistant turns = `<|turn>model\n` +
  optional `<|channel>thought\n ... <channel|>` + content + close `<turn|>`
  (or `<|tool_response>`, `<eos>`).
- IDs: <eos>=1, <turn|>=106, <|tool_response>=50 (= the three generation eos ids),
  <|channel>=100, <channel|>=101, <|turn>=105, \n=107, <tool_call|>=49, <bos>=2.
- Embedding rows for ALL structural tokens are statistically healthy
  (norm z-scores -2.7..+2.1). Rows near norm 1.01 look like untrained/rare
  tokens (init norm ~1.06), fine.
- Matrix F-norm trends across 30 layers are smooth; full-attn layers
  (5,11,17,23,29) differ by design (no v_proj, q/o ~1.4x larger, k smaller).
  Per-expert norms uniform; router.proj F-norm 2.83 everywhere. No zeroed or
  duplicated tensors so far.

### Probe battery (probe_out_a.txt) — FREE
All 3 questions: `finish: length` (3000 tokens). Pattern: first ~50-100 tokens
topically on-task but with corrupted words ("near-ungendo", "Heractor's method",
"muchingly"), then a hard repetition loop until the cap. Same in greedy decoding
(logit_probe: "**N. Use the 'near 10' method:**" x20).

### Teacher-forced logit analysis (logit_probe.py) — FREE
- "The capital of France is" -> ' a' (p=.38), ' actually', ' still', ' possible'
  ... ' Paris' nowhere. Knowledge recall broken at readout.
- "The quick brown fox" -> ' even' p=0.80 (should be ' jumps'). Confidently wrong.
- "2 + 2 =" -> ' ' (space) p=1.0, logit 22. Wrong and overconfident.
- In-context retrieval: P(" 137")=0.0001 with the answer in the prompt;
  instead ' What' p=0.987. Multi-copy: P(" 347")~0. Retrieval/copy dead.
- No logit saturation (n_|logit|>25 = 0 on readout tests).
- Greedy entropy ~0.0-0.2 during loops: the model is CONFIDENT about garbage.

### Attention maps (attn_probe.py) — FREE: ATTENTION IS HEALTHY
- Focused (entropy 1.1-2.2 vs max 3.09), heads on sensible positions.
- L17 (full-attn) heads point at the '137' token positions (mass@137=0.19)!
  Attention FINDS the answer; the readout still emits ' What'.
=> The failure is downstream of attention: either the readout (tied embedding
   used as lm_head) or the FFN compute path (e.g., router one-hot collapse).
   Attention-side sabotage (k_norm) is disfavored.

### Embedding structure (scan_embed.py) — FREE
- Row norms tight (CV 12%, 97% in 1.0-1.75) + rare tail to 6.0; no random-
  multiplier signature. E^T E has one dominant eigenvalue (54795 vs 5776) =
  common direction (softmax-invariant; harmless for ranking).
- Random-pair cos baseline 0.092; specials not clustered (mild).

### Backup taken before any edit
/home/user/wsp11/sandbox/model/model.safetensors.bak (exact copy).
edit_weights.py: restore-from-bak then apply JSON recipe (reproducible).
exp_harness.py: in-RAM recipe battery (no disk edits) — running.

### Recipe battery (exp_harness.py, in-RAM) — FREE: none of the simple fixes work
- base: france_rank1=3487, ret137_rank1=19 p=8e-5, fox_rank1=7745, loop=0.089
- roweq (rows->median norm): ret137_p 8e-5->0.002 (25x better), ranks improve
  somewhat (2110/6/12) but generation still corrupted ("Distallom"), loop .225
- router32 (router.scale/32): worse (loop 0.793). Not the fix.
- knorm1 / knormup (q,k gains): gibberish. Not the fix. (H-k dead.)

### Copy detector (scan_copy.py) — FREE
No exact multiset copies among all 241 2816-dim vectors; no scaled copy of
model.norm. "copy between tensors" on vectors ruled out. E spectrum: v0 is a
natural common direction (projections near-constant); control matrices
(down_proj, o_proj) have smooth trained spectra.

### g_test.py — FREE: FINAL NORM IS NOT THE SABOTAGE
g-free cos-rank of true tokens vs current logit-rank:
  France->Paris: cos 9158 vs logit 3487 (logit BETTER)
  fox->jumps:    cos 39847 vs logit 7745
  Japan->Tokyo:  cos 2469 vs logit 373
  2+2->' 4':     cos [1,4]  vs logit [1,6]  (fine!)
  retrieval:     cos [47,4,1,3] vs logit [19,1,1,1]
=> hidden states themselves are wrong (h not pointing at true next-token
   embeddings); current readout adds info vs raw cos. Also: once ' ' is
   forced after "Answer:", the digits 1/3/7 rank #1 -> the model half-knows
   the answer but the FIRST token choice ('What' p=0.987) is wrong.
Next: layer-wise logit-lens trajectory to localize the damaged layer(s).

### Lens + swap tests + alignment scan — FREE
- Logit-lens: mid-stack states noisy; final readout rescues (consistent with
  g weighting being load-bearing, not damaged).
- Rank battery (exp_harness2): fact median rank 5390 (Paris 3487, Tokyo 373,
  Berlin 5390, Jupiter 69328, Macbeth 91109...), syntax median 945. Copy
  cases fine (2+2 first-token rank 1 is the space token; digits after
  'Answer: ' forced = rank 1). Broad damage, worst on semantic recall.
- swapexperts (swap gate/up halves of experts.gate_up_proj): WORSE (med 19425,
  loop 0.86). swapmlp (gate<->up): WORSE (med 72436, gibberish). Gate/up
  ordering is correct in both branches.
- scan_align: post_feedforward_layernorm has consistent rogue dims across
  layers (natural); others layer-specific. No permuted-vector fingerprint.
- scan_cols: E column tail = natural outlier dims (wrong winners don't use
  them). q_proj col norms tight (1.7-2.4).
Ruled out so far: E row/col scale, low-rank E delta, final-norm scramble,
gate/up swaps, router.scale 32->1, q/k norms to 1, tensor copies, vector
permutations (no alignment break), attention-map damage.
NEW: a UNIFORM scale of a whole tensor class is invisible to all stats so
far -> behavioral scale sweep running (sweep_scale.py).

### Scale sweep results (transformers) — FREE
- Every class robust to uniform scale (med rank ~3000-4500) except:
  mlp_gate / exp_gateup / post_attn_norm / pre_ff_norm are load-bearing
  (perturbing hurts; current values near local optimum).
- layer_scalar: scaling 0.1x..10x = EXACTLY zero effect -> transformers
  IGNORES this tensor. final_norm scale: rank-invariant (monotone).
- embed x2-3: med 3487->~1890 (mild, noisy; input-side effect only).
- qnorm/knorm scaling identical to each other (attention temperature);
  x0.25 mildly better (med 2219) but noisy.
- post_ff_norm1 x c ~== post_ff_norm2 x 1/c exactly -> the two branch norms
  enter as a RATIO. Current ratio not sharply constrained.

### Vector/matrix fingerprint scans — FREE: all clean
- router.proj rows are exactly unit norm (cosine-router design, plausible).
- No zero rows anywhere. No sorted-norm permutations. Winner tokens mildly
  high-norm (mean z +1.3; too small to explain rank-3487 failures).
- High-norm rows are function/hedge words (own 6.0, way 4.5, much 4.0...) =
  natural trained-embedding profile (bottom rows = untrained cuneiform).

### *** layer_scalar IS USED BY vLLM (disk test) ***
- layer_scalar=1.0 -> HTML soup (<tfoot>, <unused32>), stops in 4-26 tokens.
- layer_scalar=0.0 -> multilingual unigram soup (no FFN signal at all),
  identical stream for all prompts.
- as-is 0.07..0.82 -> coherent-but-loopy (the BEST of the three).
=> Current values are the tuned optimum -> layer_scalar is NOT the sabotage.
=> BUT vLLM and transformers DIVERGE on this tensor -> transformers-only
   batteries can mislead; eval behavior = vLLM. Verified model restored to
   pristine (verify_restore.py: 0 mismatched tensors vs .bak).

### Spectral spike / zero-row scan — FREE: clean
No zero rows anywhere. Top Gram-eigenvalue ratios 1.1-4.8 (k_proj L0 4.6,
o_proj L29 4.8, o_proj L5 3.8) — high-ish but plausible for trained
matrices; no isolated rank-1 outlier.

### Next (running): per-layer candidate-set logit lens (lens_v2.py)
- copy case ("The code is 4711 ... Answer:" -> digit 4) vs fact cases
  (France->Paris, water->Celsius) with small candidate sets.
- If the true token appears at mid-layers and dies later => localized layer
  damage. If present-but-suppressed => row-bias hijack. If never present =>
  distributed damage (value path / FFN content).

### Lens results (lens_v2/v3) — FREE: LATE LAYERS DESTROY CORRECT ANSWERS
- Water->Celsius full-vocab rank by layer: L1 5681, L3 47, L4-9 ~100-670,
  L10-23 ~7k-35k (Kelvin/Fahrenheit overtake), L24 150, **L25 6**, L26 236,
  **L27 36890**, L28 11327, L29 3903. Real final (transformers logits) bad.
- France->Paris: cand-rank 1 at L3-8 (but full-rank 9.5k-26k), then wrecked;
  final prefers Rome over Paris.
- Copy ("code is 4711 ... Answer:") dead at ALL layers (cand-rank never 1).
- NOTE: my lens_v3 multiplied the final-norm gain twice at index 30 (that
  row is bogus); indices <30 are valid logit-lens readings.
- The readout formula works when h is right (L25 rank 6 with CURRENT g,E!)
  => G and E are healthy; the COMPUTATION in layers 26-29 destroys the answer.
- lens_v3 top-20s are dominated by generic multilingual particles / hedge
  words = what "garbage h" looks like through the readout (not a row hijack:
  g_test showed g-free cos is even worse than real logits).

### Repair hypothesis (testing now in vLLM)
Zero layer_scalar for L26-29 (= early-exit at L25, where answers are right).
vLLM uses layer_scalar; transformers ignores it. If vLLM emits correct
answers and stops -> keep and run official pass.

### ARCHITECTURE RESOLVED (library source: transformers/models/gemma4, vllm gemma4)
- layer_scalar is an nn.Buffer -> NOT in named_parameters() -> my whole
  transformers sweep was blind to it. Both frameworks do:
  `hidden_states = hidden_states * self.layer_scalar` at the END of each
  decoder layer (scaling the residual stream INCLUDING the skip).
- Layer block: dense MLP reads pre_norm(h); MoE branch reads pre_norm_2
  (RESIDUAL) with router(residual); branch outputs normed (_1/_2) and SUMMED
  (explains post_ff_norm1 x c ~== post_ff_norm2 x 1/c), post_ff_norm, then
  residual add. Then layer_scalar multiplies everything.
- Consequence: block contributions to the final state are weighted by the
  PRODUCT of later scalars. With s26..29 = 0.80,0.73,0.68,0.195, all pre-L26
  content arrives at ~7% weight vs L29's block at 100% -> late blocks dominate.
- CORRECTION: skip_test.py "ranks all 1" was a METRIC ARTIFACT: setting
  s26-29=0 zeroes the hidden state -> all logits tie -> rank reads 1 for any
  target. vLLM's soup on that state = uniform sampling over tied logits.
  Both frameworks agree; no divergence after all.
- skip (identity) of L26-29 = readout at hs[26] = the multi-skip numbers =
  lens hs[26] (236 water / 149k France) -> NOT a repair by itself.
Now running tie-safe skip + scalar-buffer tests (skip_test2.py) on pristine.

### skip_test2 (tie-safe) — FREE: skips/scalars do NOT rescue content
- baseline: Paris 3487 ('a' +12.1), Celsius 9516 ('_' +18.1), 4: 6 ('1'),
  Jupiter 69328 ('part' +20.6), Au 2427 ('a').
- All single/multi layer skips: no consistent gain (some help one case, hurt
  others). Scalar rewrites: s26-29=0.8 mild (Paris 2291, Celsius 5123);
  s29=0.82 ~no-op (global scale cancels in rms-norm); all=0.8/0.9 garbage.
- Full-attn layers confirmed: q(8192), k(1024)=2 kv heads x512, v=None (v:=k).

### Super-hub fingerprint (scan_v1/scan_dup) — FREE
- v_proj/k_proj head blocks: no duplicates (max cos ~0.014). q/o max 0.6-0.7
  (moderate, natural-looking). No copy-between-tensors on head blocks.
- v1 (2nd eigvec of E) bottom tail = THE HIJACK TOKENS: own(-2.0), way,
  much, ever, same, fact, world, very, entire, actual, even, first, next...
  = ALSO the row-norm top tail (own 6.02, way 4.50, much 4.05...) =
  the cross-context wrong winners (a/actually/still/possible/indeed/even).
- Answer tokens (Paris/Celsius/Jupiter/Rome/Kelvin/Berlin) all cluster at
  v1 -0.19; hijack tokens at -0.28..-2.0. Unused-token cluster at +0.735
  (natural init cluster).
- Hypothesis: the hedge-token rows were scaled up (super-hubs) and hijack
  generation. Testing targeted row clip (rows>2.0 -> median) in rowclip_test.
  (roweq earlier normalized ALL rows: retrieval 25x better but generation
  still corrupted — the all-row distortion may have masked the gain.)

### Row clip + composite probe — FREE
- rowclip (439 rows >2.0 -> median): margins/ranks UNCHANGED -> hub rows are
  natural, not the sabotage. (Winners 'a','_','part' aren't hub rows anyway.)
- Router semantics vLLM == transformers (norm -> scale/root_size -> proj ->
  softmax-all -> topk -> renormalize -> per_expert_scale). No divergence.
- MECHANISM: in generation the raw (pre-softcap) logits reach the 30.0 tanh
  softcap (post-cap top values 22-29 in greedy) -> top tokens TIE -> sampling
  is random among them -> word corruption + loop attractor.
- Composite1 (s26-29=0.8, final-norm gains x0.4, stop rows 1/50/106 x1.6):
  vLLM probe shows 2/3 generations now SELF-TERMINATE (stop at 516/564 tok)
  and cleaner words; still loops ("$17x23 = 17x23$" x8) and wrong arithmetic
  ("17x20 = 30"). Termination fixed; correctness NOT fixed.
Now testing whether final-norm rogue gains (588/576/548...) scramble the
readout (g_test2: g=mean/1/med/clipped variants vs pristine ranks).

### g_test2 + scalar sweeps — FREE
- g variants (mean/1/med/clipped): ALL worse than base -> final-norm gains are
  fine (their rogue dims actually suppress the hub rows). Readout healthy.
- Scalar sweeps (in-memory, transformers): raising LATE scalars reweights
  early/mid content (the answer!) upward in the final state:
  base            [3487, 9516, 6, 69328, 2427, 91109]
  s26-29=0.98     [1529, 2194, 7, 59909, 3105, 71361]
  s20-29=0.95     [2818,  334, 9, 41932, 3705, 39319]
  s26-29=0.95,s0-2=0.5 [2646, 399, 1, 73627, 1690, 215452]
  (ranks for Paris, Celsius, 4, Jupiter, Au, Macbeth)
- s0-2=0.5 additions also make generations END WITH <turn|> cleanly
  ("To answer: **17 ^ 230**<turn|>") — stop behavior emerges from scalars.
- Caveat: even best ranks (~300-3000) are outside top-64 sampling; expect
  extraction/dynamics gains more than answer-correctness gains.
Building c2a (s20-29=0.95) and c2b (s26-29=0.95 + s0-2=0.5) with g x0.4 +
stop rows x1.6; validating in vLLM probes, then official passes.

### c2a probe + PASS 2 (OFFICIAL)
- c2a (s20-29=0.95 + g x0.4 + stop x1.6) OVER-STOPS (12/3-token truncations).
  Too much stop boost / too much scalar raise.
- PASS 2 = composite1 (s26-29=0.8, g x0.4, stop rows 1/50/106 x1.6):
  accuracy 0.0 (was 1.6) | answered 96.9 (92.2) | cap_hit 6.2 (98.4) |
  mean_tokens 1931 (8183). Termination SOLVED; correctness not improved
  (answers extracted are wrong).
=> The eval's accuracy needs CORRECT answer content. Lens showed answers
   live at mid-layer states ('Celsius' rank 6 = within top-64 SAMPLABLE at
   hs[25]). Testing early-exit (identity-skip last N N layers) so the readout
   sees the mid-layer state; repair = zero late blocks + scalars 1.0.

### Early-exit sweep + MATH knowledge discovery — FREE
- Early exits (skip last 1-10 layers): ALL WORSE than base (hub rows ('own')
  hijack intermediate-state readouts; only water->Celsius reaches rank 6 at
  skip 25-29). Final state is the best readout. Early exit abandoned.
- BASELINE math ranks (per-digit, teacher-forced) on the FINAL state:
  17*23=391:  [360, 1245, 71]
  2^10=1024:  [6, 1, 97, 5]     <- model KNOWS this almost perfectly!
  7*8=56:     [18835, 125]
  sqrt(144)=12: [68, 1]
  100-37=63:  [3446, 763]
=> The model has PARTIAL math knowledge at the final state; memorized facts
   (2^10) are near-top-64 (samplable!), computations are not. If a scalar
   config pushes more digits into top-64, sampled answers become correct.
Now: math-battery scalar sweep to find such a config.

### Math scalar sweep — FREE (frac of answer digits inside top-64)
- base 0.42 | s26-29=0.8 0.42 | s26-29=0.95 0.53 | s24-29=1.0 0.53 |
  s20-29=0.95 0.32 | s0-2 additions hurt (0.16-0.26).
- s24-29=1.0 math ranks: 391 [8,1341,30] | 1024 [35,7,1155,11] |
  12 [31,7] | 144 [16,1707,2] | 256 [41,905,80] | 56 [9732,183] |
  63 [7304,121]. First digits of most problems enter top-64.
- c3a probe (s24-29=1.0, g x0.7, stop x1.4): dynamics REGRESS (2 of 3 loop
  to cap). Softer g = more saturation = more loops.
Testing c3b = s24-29=1.0 + g x0.4 + stop x1.6 (composite1 dynamics package
with better math scalars).

### PASS 3 (OFFICIAL, c3b)
accuracy 0.0 | answered 98.4 | cap_hit 4.7 | mean_tokens 1091.
Best dynamics yet (cap 4.7%, answered 98.4%) but accuracy still 0.
- c3b probe: 3/3 generations complete and stop (247/133/383 tok), structured
  attempts with final answers ("Final Result: 20!") — extracted answers are
  wrong. One question degrades to mojibake.
- Hypothesis: g x0.4 sharpening (anti-saturation) kills the lucky-digit
  sampling that the saturated baseline had (baseline 1.6 vs 0.0 twice).
Testing c4 = s24-29=1.0 + stop rows x3.0 + NO g change (soft regime).

### c4/c5 probes + PASS 4 (OFFICIAL)
- c4 (stop x3.0): over-stops (26-99 tok truncations). Too strong.
- c5 (s24-29=1.0 + stop x1.6, no g change): 3/3 stop (310/690/570 tok),
  loops persist inside but termination OK.
- PASS 4 (c5): accuracy 0.0 | answered 100.0 | cap_hit 9.4 | mean_tokens 1828.
  answered=100% (every gen extractable) but accuracy still 0.
Official so far: 1.6 (base) | 0.0 | 0.0 | 0.0. The baseline's 1.6 = 1 lucky
problem. Correctness is the bottleneck: extracted final answers are wrong.
Next (c6): digit-token row boost x2.0 so digits beat words at answer
positions (saturation pushes doubled digit logits to the cap band) +
s24-29=1.0 + stop x1.6.

### c6 probe + PASS 5 candidate
- c6 (digits x2.0): DISASTER — digit spam ("1700000..."), all 3 hit cap.
  Row-scale on output rows is extremely potent; too blunt.
- All 4 official passes sit in the 0-2 problem noise band (1.6/0/0/0 out of
  484 problems total... n=64 each). The dynamics improvements (cap 98->5%,
  answered ->100%) are real; accuracy is noise-bounded.
- PASS 5 = c7 minimal edit (layer_scalar 24-29 = 1.0 ONLY): keeps the
  baseline rambling regime (which scored 1.6) while lifting math digit ranks
  into top-64. The cleanest remaining accuracy hypothesis.

### PASS 5 (OFFICIAL, c7 minimal scalars)
accuracy 0.0 | answered 98.4 | cap_hit 93.8 | mean_tokens 7743.
Baseline-like rambling retained but accuracy still 0 on this set.
Official so far (fixed 64-problem set, fixed seed 20261116 -> deterministic):
  P1 baseline   1.6 | 92.2 | 98.4 | 8183
  P2 composite1 0.0 | 96.9 |  6.2 | 1931
  P3 c3b        0.0 | 98.4 |  4.7 | 1091
  P4 c5         0.0 | 100.0 | 9.4 | 1828
  P5 c7         0.0 | 98.4 | 93.8 | 7743
P(0/64 | true p=1.6%) = 36% -> variants are not proven worse; the set is
fixed and decoding is seeded, so each config is one deterministic draw.
PASS 6 = c8 (s26-29=0.95 + g x0.4 + stop x1.6) — another corner of the
mapped config space.

### PASSES 6-7 (OFFICIAL)
- P6 c8: 0.0 | 96.9 | 12.5 | 2587.
- P7 c9 (luck maximizer: s24-29=1.0 + g x2.0 + stop x1.6): 0.0 | 100.0 | 1.6
  | 279.5. Termination perfect (cap 1.6%, mean 280) — but accuracy 0.
Seven draws: [1.6, 0, 0, 0, 0, 0, 0]. Each config = one deterministic draw
on the fixed set; correct-answer rate is ~1-2%/problem. No static edit has
produced >0 correct. PASS 8 = c10 (best dynamics + mild digit rows x1.15;
x2.0 caused digit spam, x1.15 should be mild) — final draw, then finish.

### PASSES 8-9 (OFFICIAL ablations) + endgame
- c10 probe (digits x1.15): no spam but 2/3 loop to cap and math still wrong
  ("10x20 = 40") — not better than c3b; not worth a pass.
- P8 = stop rows x1.6 ONLY: 0.0 | 96.9 | 35.9 | 4072. -> stop rows ARE the
  termination mechanism (cap 98.4 -> 35.9 alone; -> 4.7-6.2 in combo).
- P9 = final-norm x0.4 ONLY (running): isolates anti-saturation effect.
- Accuracy remains 0 in every variant; baseline's 1.6 = single lucky problem
  (P(0/64 | p=1.6%) = 36%; no variant proven worse, but none measured better).
ENDGAME: restore pristine as installed model (best MEASURED accuracy on the
deterministic fixed set), run P10 on the final state to document exactly
what the installed model scores, then write final_report.md.

### P9 RESULT CHANGES THE ENDGAME
P9 = final-norm x0.4 ONLY: accuracy 1.6 (!) | answered 96.9 | cap 90.6 |
mean 7820. The g-shrink reproduces the baseline's 1.6 while improving
answered (92.2->96.9) and slightly reducing cap/mean -> c12 DOMINATES
pristine on every metric. No restore needed.
P10 (final pass) = c13 = g x0.4 + stop rows x1.6: combines P9's accuracy
with P8's termination. Install rule: if P10 >= 1.6 keep c13; if P10 = 0.0
install c12 (measured 1.6/96.9/90.6/7820). Either way the installed model
measures >= baseline accuracy with better secondary metrics.

### SESSION END (passes exhausted: 10/10)
- P10 = c13: 0.0 | 96.9 | 4.7 | 1172 (stop boost perturbed the lucky draw).
- Final install per the rule: **c12** = `model.language_model.norm.weight`
  x 0.4 (verified: exactly 1 of 1013 tensors differs from the backup).
  Measured 1.6 accuracy (best level, tie with pristine) while strictly
  improving answered (92.2->96.9), cap_hit (98.4->90.6), mean (8183->7820).
- No edit improved measured accuracy beyond the baseline's single lucky
  problem; the failure is compute-level (value path / associations) per the
  diagnosis in final_report.md. Report written.
