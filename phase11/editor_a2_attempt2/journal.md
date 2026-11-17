# Editor journal — Phase 11

## Setup (free analysis)

- Target: `/home/user/wsp11/sandbox/model` — "Gemma4ForConditionalGeneration", 8.77 GB single
  safetensors shard, bf16, ~4.4B params.
- Text stack: 30 layers, hidden 2816, sliding attention (window 1024, head_dim 256, 16 heads,
  8 kv heads) with full-attention layers at 5, 11, 17, 23, 29 (head_dim 512, 2 global kv heads),
  MoE block per layer: 8 experts × 704 moe_intermediate, top_k_experts = 8 (all experts always
  routed — router effectively averages experts), PLUS a dense `mlp` (2112) per layer.
  Tied embeddings, vocab 262144, final_logit_softcapping 30.
- `layer_scalar` [1] per layer; `router.scale` [2816] ~32; `per_expert_scale` [8] ~1.
- q_norm [256 or 512] and k_norm are EXACTLY constant vectors (std 0) — per-layer scalars
  (q ~1.0, k ~0.12 sliding / ~0.06 global). Looks like part of this synthetic model's design,
  not obviously the sabotage.

## Probe 1 (free, temperature 1.0 / top_p 0.95 / top_k 64 / seed 7)

Massively broken generations:
- "17*23" → degenerate numbered-list loop: "**24. The Final Result** **25. The Final Result** ..."
  repeated to the token cap.
- "gcd(48,64), answer with just the number" → "45" (3 tokens). Wrong, and oddly terse.
- Derivative question: similar degenerate loop (output truncated in tail).

Interpretation: model has fluent local token dynamics but broken long-range/correctness
behavior — consistent with a targeted weight lesion rather than global noise.

Stats scan (stats.py → stats_out.txt): no NaN/Inf, no zero tensors, no obviously dead rows
at the global level. Need cross-layer trend analysis to find the lesion.

## Architecture (from transformers gemma4 modeling code — the runtime lib, allowed)

Per decoder layer: `h1 = r + RMS(attn(RMS(h, input_ln)), post_attn_ln)`; MoE block:
`m1 = RMS(mlp_dense(RMS(h1, pre_ff_ln)), post_ff_ln_1)`, `e = RMS(experts(RMS(h1,
pre_ff_ln_2)), post_ff_ln_2)`, `h2 = r + RMS(m1 + e, post_ff_ln)`, then
**`h2 *= layer_scalar` — the WHOLE stream (residual included) is rescaled every layer**.
Global-attn layers (5,11,17,23,29) have no v_proj: v = raw k_proj output (k_eq_v),
head_dim 512, 16 heads, partial rope 0.25. Sliding layers: head_dim 256, 8 kv heads.

## Findings log

1. First suspect — "layer 29 update 5x explosion" — was an ARTIFACT: hidden_states[-1]
   includes the final RMSNorm (weight 2-norm 4745). Retracted. layer_scalar values are
   trained per-layer rescalings (L0=0.07 … L29=0.195), by design.
2. REAL anomaly: `layers.23.self_attn.q_proj.weight` has 298 rows with norm ~0.25
   (median row norm 1.607; other global layers have 7-14 such rows). Geometry: heads
   {0,5,9,12,15}, within each head offsets 0–~40 and 256–~296 — the ROPE region of each
   256-half (healthy global layers have a naturally low-norm region there, ~0.3–0.6;
   L23's affected heads are at ~0.25 = extra ~2.5x shrink).
3. Kurtosis test inconclusive (whole model near-Gaussian). No duplicated tensors.
4. Filename metadata in phase11/ (not read, names only): "prune_summary.json",
   "smoke_wound.json" — the sandbox model is a PRUNED checkpoint; L23 q_proj lesion is
   consistent with pruning damage.
5. embed_tokens: ~1% of rows have norm ≈ 1.0 vs median 1.56 (untrained/filler rows?), no dead rows.

## Pass 1 (baseline) — RESULT

accuracy 1.6% (1/64), answered 92.2%, cap_hit_rate 98.4% (loops to 8192 tokens),
mean_tokens 8183. Decoding: t=1.0 p=0.95 k=64 seed=20261116.

## Free experiments after pass 1

- 12-question battery (eval-matched decoding + greedy): baseline sampled=2/12, greedy=2/12.
  Model CANNOT do arithmetic: echoes operands (12*12→12, 7*8→7, 100/4→100, -5+12→12,
  9*6→"9 6"). sqrt(144)→12.58 loops. gcd(48,64)→45.
- Diagnostics: in-context retrieval WORKS ("answer is 391...what is 17*23"→391; repeat 391→OK;
  x=7→7; counting 1,2,3→4,5,6..). World knowledge GONE (capital of France→"France" echo;
  largest planet→"**NOTOSOLLEY**" word salad). Raw (non-chat) completions are token salad
  ("The capital of France is"→" a a a a b i i i...") — model only functions in chat format.
- Alpha (final-norm scale) simulation on real logits: sampling is ALREADY greedy-dominated
  (p(greedy)=0.88 at alpha=1.0; all alpha<1 worse). Sharpening is a dead end. Loops are in
  the GREEDY path.
- EOS gap: capped-logit gap to <eos> ≈ 27, to turn-end ≈ 20 — no embedding-scale boost can
  flip termination selectively (would need x7 = terminates everywhere). Dead end.
- Attention patterns healthy in ALL layers (entropy 1.5-2.6, BOS sink, operands attended).
- Norm-weight distributions look trained everywhere (heavy tails, smooth trends).
- L23 q_proj repairs tested: rowfill(298 rows→median) = 2/12 sampled (no change);
  zero rows = 2/12 sampled, 1/12 greedy (no change). Lesion is cosmetic, NOT the wound.
- No embedding wound: ~1.0-norm rows are unused/special tokens (ids 5-99, 239-365, tail);
  digits (ids 236770-236832) have healthy norms 1.25-1.43.

## Ablation sweep (92 configs, in-memory branch-zeroing, greedy battery)

Baseline 1/12. All configs 0-3/12 = noise; no toxic component; every generation loops
to the 160-token cap. No surgical undo via ablation.

## Empirical knob: attention sharpening (k_norm scale)

k_norm (and q_norm) are per-layer CONSTANT RMSNorm weights (sliding ~0.122, global
~0.0601); they set the attention-logit temperature. Scaling k_norm multiplies attention
logits linearly (v is unaffected; v_norm is scale-free by design).

battery2 (20 MATH-style questions, eval-matched decoding, sampled/greedy):
- baseline:        4/20 sampled, 4/20 greedy
- digits x1.3:     4/20, 4/20  (no effect)
- anti-loop(**,| x0.5): 5/20, 5/20 (noise)
- k_norm x1.3:     6/20, 7/20
- k_norm x1.7:     7/20, 5/20
- k_norm x2.0:     9/20, 6/20   <-- peak; generations cleaner, self-corrections appear
- k_norm x2.4:     5/20, 5/20
- k_norm x3.0:     3/20, 3/20   (overshoot)

Hypothesis: the wound halved every k_norm (a 60-byte, findable, exactly-revertible edit);
x2.0 restore = 0.2441 (bf16 exact double of 0.1221). Installed k_norm x2.0 into
sandbox/model (backup at sandbox/model_backup_orig). Pass 2 launched.

(Background note: the whole model is near-Gaussian, so a noise-REPLACED tensor would be
statistically invisible — that motivated the functional ablation sweep above.)

## Pass 2 RESULT: k_norm x2.0 official

accuracy 0.0% (vs 1.6% baseline), answered 100%, cap_hit 87.5%, mean_tokens 7416.
Battery2's +5/20 was NOISE on an unrepresentative battery. REVERTED to baseline
(sandbox/model restored from model_backup_orig).

## Battery4 (120 items: programmatic arithmetic + facts + algebra, strict lastnum)

Rationale: 40-item batteries had ±3 noise; need power to detect real deltas.
- base:   lastnum 26/120, first200 22/120, stops 36, mean_tok 1456
- kn20:   20/120 (all k_norm x2 — WORSE)
- kg20:   25/120 (global layers only x2 — neutral)
- ks20:   21/120 (sliding only x2 — worse)
- greedy: 27/120 (== sampled; no greedy/sample gap to exploit; final-norm sharpening dead)

Baseline category breakdown: single-digit add 8/15 (53%), facts 12/38 (32%),
2-digit add 0/11, subtraction 3/14, algebra-x 2/30, pct 0/6, div 1/6.
Model does memorized single-digit sums + some trivia; ALL multi-digit computation dead.
lastnum(26) >= first200(22): loops REPEAT the first answer — early-termination edits
cannot help the last-number-style scorer.

Remaining tests queued: moderate (x2) rescale of L23/L29 wounded rope rows;
uniform routing (zero router.proj).

## Moderate lesion repair (rows x2 -> norm ~0.5, the natural rope-region level)

battery4: l23x2 26/120 (=base), l2329x2 28/120 (+2); router0 9/120 (uniform routing
is MUCH worse — routing is load-bearing). Replication with seed 8: base 25/120,
l2329x2 31/120 (+6). Pooled: base 51/240 (21.3%) vs l2329x2 59/240 (24.6%);
l2329x2 also terminates more (stops 34-36 -> 36-44) and is shorter.

## Pass 3 RESULT: l2329x2 official

accuracy 1.6% (ties baseline), answered 96.9%, cap_hit 98.4%, mean_tokens 8067.
The battery delta did not transfer to the harder 64-problem set (as feared —
the battery delta comes from easy retrieval items).

## Final exploration sweep (variants built on l2329x2)

mlp.down x1.15: 31/120; x0.85: 27; experts.down x1.15: 27; o_proj x1.1: 26;
layer_scalar L29 -> 0.7: 28. All within l2329x2's own noise band (26-31). Also
checked gate/up half-norms (no inversion = no swapped-halves wound) and global
o_proj head-block columns (no scrambled head blocks).

## FINAL DECISION

No weight-arithmetic edit found that beats the model's ~1.6% MATH-500 level.
Best candidate = l2329x2 (ties baseline officially at 1.6%, +3.3pp pooled on the
240-item offline battery, more terminations, shorter generations). INSTALLED and
verified (md5 f23726454c02f9472be965b3ebcd3e1c == m_l2329x2; pristine backup at
model_backup_orig md5 b69e321ec5f6a270f3e1dcb4bc313a97). Passes used: 3/10.
Failure-mode diagnosis and full negative-results ledger in final_report.md.





