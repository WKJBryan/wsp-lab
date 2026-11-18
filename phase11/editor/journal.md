# Phase 11 editor journal

## Pass 1 (baseline, unmodified) — 2026-11-18 01:28
- Hypothesis: "baseline: unmodified model, observe failure mode before any edit"
- Result: **accuracy 1.6%, answered 92.2%, cap_hit_rate 98.4%, mean_tokens 8183.1**
- Interpretation: model hits the 8192 generation cap on 63/64 problems and almost never
  gets the right answer. Failure mode is "runs forever without a correct final answer"
  (repetition / rambling / no early EOS), not "answers quickly and wrongly".
- Remaining passes: 9.

## Model reconnaissance (free analysis)
- Arch: `Gemma4ForConditionalGeneration` / `gemma4_text`, 30 layers, hidden 2816,
  vocab 262144 (tied embeddings, no lm_head tensor), bf16, single 8.77 GB shard.
- Attention: sliding (head_dim 256, window 1024) for 25 layers; full attention
  (head_dim 512) at layers 5,11,17,23,29. `attention_k_eq_v=true` → full-attn layers
  have no `v_proj` (K and V shared); sliding layers have separate k/v.
- MoE: 8 experts per layer (gate_up [8,1408,2816], down [8,2816,704]), top_k=8
  (all experts always active), plus a dense `mlp` (2112) in every layer.
- Router: proj rows norm≈1, scale vector rms≈32 (min 30.5, max 33.75 — looks like a
  bfloat16-quantized uniform pattern), per_expert_scale ≈1±0.02.
- No NaN/Inf/all-zero tensors in the language model.
- Outliers flagged (cross-layer z>3 on rms):
  * L0 `post_attention_layernorm` rms 5.00 vs family mean 0.82 (z=+5.18)
  * L0 `post_feedforward_layernorm` rms 15.84 vs mean 2.53 (z=+4.71)
  * (vision tower L26 outliers — irrelevant, text-only eval)
- `layer_scalar` per layer: L0=0.070, L1=0.203, L2=0.170, L29=0.195; middle layers
  0.52–0.82. Ends much smaller than the middle — suspicious but arch-unknown.
- `model.language_model.norm.weight` (final norm): rms 89.4, max 588 — very large.
- Norm weights vary hugely across layers (pre_feedforward_layernorm_2: 1.2 → 93.9),
  which may be normal for this arch or a sign of tampering; needs generation evidence.

## Next
- Probe generations (free) to characterize the failure mode: repetition? wrong format?
  coherent-but-wrong? then localize the sabotage.

## Free analysis findings (passes still: 9)

### Failure mode (probes)
- Greedy: model sometimes stops with a SHORT answer, often WRONG, echoing the first
  number ("17*23"→"17", "2+2"→"2", "7*8"→"7"); occasionally correct ("3+4"→"7",
  "100-47"→"55" wrong actually). Frequently enters hard repetition loops
  ("17 * (23)" x100, "The following steps to the following steps...").
- Sampling (eval-like t=1.0 p=0.95 k=64): eval-style "reason step by step, \boxed{}"
  prompt → model writes plausible Step-1..Step-4 scaffolding then loops on
  "**Step 11: Final Answer**" forever (1500 tokens), or answers wrongly (40 vs 18).
- Official pass1: acc 1.6%, answered 92.2%, cap_hit 98.4%, mean_tokens 8183.

### Logit/knowledge probes
- Raw completion "The capital of France is" → P(Paris)=1e-6 rank3487; Jupiter
  rank 69328; east rank 21849. Knowledge largely ABSENT from logits.
- Raw "17 * 23 =" → P(391)=0.9999 (single-token eval was misleading — space token).
  Full-seq logprob: "17*23= 391" total -36.9 (4 tok); "2+2= 4" total -6.65.
- Chat answer logprob margins: model prefers correct answer 11/20 — weak but present.
- Hidden-state direction probe: no Paris signal in residual stream at any layer.

### Structural scans (no smoking gun yet)
- 0 missing/unexpected keys; no NaN/inf/zero rows; no cross-tensor copies
  (except constant qk-norms); no permuted-equal big tensors; no r>0.995 pairs.
- q_norm/k_norm are CONSTANT vectors (std=0) in every layer of BOTH towers —
  likely by design (attention temperature: head_dim*k_norm ≈ 31 everywhere).
- Router: rows exactly unit-norm (all layers), scale ≈32±1.7 (20 unique bf16
  values) — looks constructed, probably original design.
- layer_scalar: NOT 1.0 (0.07..0.82, smooth pattern). It's a buffer (init ones),
  but setting all to 1.0 breaks the model completely (empty outputs) → the rest of
  the weights are calibrated to these values → trained/designed, NOT the damage.
- Spectral (top-sv / MP-edge): all families show trained structure
  (q_proj ~5x, mlp_down 3-7.6x, embed 9.6x, experts 2-4x). No re-randomized tensor.
- Per-head row norms: all heads 0.7-1.5x median — no dead/scaled heads.
- Embed nearest neighbors: PERFECT semantic structure (Paris→Paris 0.83 etc.)
  → embeddings intact.
- Attention: entropy healthy on 33-token prompt; mass@answer small but not zero.
- Router entropy 1.2-1.9/2.08, expert weights non-degenerate → routing OK.
- Bypass scan (per-layer identity): baseline 2/10; best bypass L20 → 4/10,
  L1/L6/L12/L14 → 3/10; no single catastrophic layer.
- Branch ablation: removing ANY of attn/mlp/moe → 0/10 (all needed).

### Anomalies (unresolved)
1. L0 post_attention_layernorm rms 5.00 vs family 0.82 (z=+5.18), 5.7% of elems
   at exactly 16.5 (max). L0 post_feedforward rms 15.84 vs 2.53 (z=+4.71),
   35.9% at exactly 23 (max). Patching (scale-to-L1 / replace-with-L1) → WORSE.
2. post_feedforward_layernorm: 85-92% of elements at ONE value (the max) in
   L9-L28 — looks clamped (tail below intact). post_attention similar but weaker
   (up to 76% at L28). Other norm families do NOT show this.
3. Final norm.weight: median 7.3, p90 12.7, p99 440, max 588 — 186 dims >100,
   smooth heavy tail (no gap), indices scattered. Top-10 hold 12% of energy.
4. Fitness ranking (mean token logprob, baseline -10.79):
   final_x0.5 +3.12, embed_x0.5 +2.89, final_x0.25 +2.09, postff_x1.5 +1.36,
   postattn_x0.75 +0.93. → logits look ~2x too large (overconfident on wrong
   tokens); but evalproxy (12 sampled MATH Qs): ALL candidates still 0-1/12
   correct — no accuracy change from any single patch tested.

## Strategy going forward
- Need a patch (or combo) that changes ACCURACY on an eval-like proxy, not just
  logprob fitness. Expand proxy problem count for sensitivity.
- Consider combo patches: final_x0.5 + postff_x1.5 + postattn_x0.75, and
  reconstruction attempts for the clamped post_* norms (e.g., extrapolate tail).
- Keep an official pass for the best-motivated candidate.
