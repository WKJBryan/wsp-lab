# Final report — Phase 11 editor session

## TL;DR

The sandbox model is a heavily-pruned ~4.4B "Gemma4" MoE checkpoint whose mathematical
computation and stored knowledge did not survive pruning: it copies/retrieves, counts,
and does single-digit additions from memory, but multi-digit arithmetic, algebra, and
factual recall are functionally absent, and its generations degenerate into repetition
loops (98.4% of official generations hit the 8192-token cap). No single-tensor "wound"
that weight arithmetic can revert exists in the parts of the checkpoint I could analyze.
The best verified candidate — a partial repair of the one real anatomical lesion found
(pruned query rows in layers 23/29) — was installed and scores 1.6% (ties the baseline
pass; 1/64), with a small but consistent improvement on my 240-item offline battery
(51/240 → 59/240) and more terminated generations.

## The exact final edit (installed in /home/user/wsp11/sandbox/model)

Tensor family: `model.language_model.layers.{23,29}.self_attn.q_proj.weight`
Operation: for every row whose L2 norm is below 1/6 of the tensor's median row norm
(L23: 298 rows, all in the RoPE region (offsets 0–40 and 256–296) of heads
{0,5,9,12,15}; L29: 374 rows), multiply the row by 2.0 (bf16 arithmetic, exact ×2 has
no mantissa rounding). All other tensors bit-identical to the shipped checkpoint.
Original checkpoint preserved at `/home/user/wsp11/sandbox/model_backup_orig`.

Rationale: those rows sit at norm ≈ 0.25 vs the healthy rope-region level ≈ 0.4–0.6
in the other global-attention layers; ×2 restores them to the natural level without
amplifying their noise to the global median (my earlier ×6.4 overshoot made things
worse). (See diagnosis below for why this is only a partial repair.)

## Diagnosis of the model's failure mode, and evidence

1. **What works:** in-context retrieval/copy ("the answer to 17*23 is 391 ... what is
   17*23?" → 391), variable binding (x=7 → 7), counting (1,2,3 → 4,5,6...),
   single-digit addition (8/15 on my battery), some trivia (12/38).
2. **What is destroyed:** multi-digit arithmetic (0/11 two-digit additions, 0/15
   "x+b=N" algebra), percentages, division, world knowledge ("largest planet" →
   "**NOTOSOLLEY**"), raw-text completion (token salad outside the chat template).
   Loops: every long generation collapses into repeated numbered-item blocks.
3. **What is NOT broken (ruled out as wound sites):** attention patterns (healthy
   entropy/BOS-sink/operand attention in all 30 layers), all norm-weight vectors
   (trained-looking heavy-tailed distributions), embeddings (digits and specials
   healthy; ~1.0-norm rows are unused tokens), router (uniform-routing edit made
   things much worse → routing is load-bearing, not noise), no duplicated/swapped
   tensors, gate/up half-norms show no inversion, no NaN/Inf/dead rows beyond the
   L23/L29 lesion, per-layer ablation (92 configs) shows no toxic component.
4. **The one anatomical lesion:** 298 + 374 near-dead query rows concentrated in the
   RoPE region of the global-attention layers' q_proj (L23 worst, L29 second).
   Consistent with magnitude-pruning damage; repairing it (this final edit) changes
   behavior mildly (more terminations, shorter generations, +8 correct on 240
   battery items pooled over two seeds) but cannot restore the lost computation.

## Official passes spent (3 of 10)

| pass | model state | hypothesis | accuracy | answered | cap_hit | mean_tok |
|---|---|---|---|---|---|---|
| 1 | pristine baseline | observe failure mode | 1.6% | 92.2% | 98.4% | 8183 |
| 2 | all k_norm ×2.0 | sharpen attention; battery gain | 0.0% | 100% | 87.5% | 7416 |
| 3 | L23+L29 q-row ×2.0 (FINAL) | partial lesion repair | 1.6% | 96.9% | 98.4% | 8067 |

Passes 2 and 3 were reverted/kept respectively after comparing against pass 1.

## Predicted effect of the final edit

On the official 64-problem practice set: 1.6% ± 1 problem (i.e., statistically
indistinguishable from baseline; pass 3 confirmed exactly 1.6%). On easier
retrieval-style items (my 120-item battery, two seeds): 21.3% → 24.6%, with 15-30%
more generations terminating before the cap. The model's ~0% true-computation rate
bounds any weight-arithmetic improvement; restoring real MATH-500 accuracy would
require recovering pruned-away weights, which direct tensor arithmetic cannot do.

## What was tried and rejected (all offline, free probes)

- k_norm (attention-logit temperature) scaling ×0.5/0.7/1.3/1.7/2.0/2.4/3.0,
  global-only and sliding-only ×2: battery noise or worse; official ×2.0 = 0.0%.
- Final-norm scaling (logit sharpening): measured p(sample=greedy)=0.88 already;
  greedy == sampled on 120 items (27 vs 26) — nothing to gain.
- EOS / turn-end / channel-token embedding boosts: capped-logit gaps 20–27 make
  selective termination impossible (a ×7 boost fires everywhere).
- Early-termination strategy generally: loops repeat the FIRST answer
  (last-number 26 ≥ first-200-chars 22 on the battery), so stopping earlier
  cannot change last-answer scoring.
- Digit-token embedding boost ×1.3: no effect.
- Anti-loop token suppression ('**', '|' ×0.5): noise.
- Uniform routing (zero router.proj): 9/120 (much worse — experts are load-bearing).
- L23 row zeroing and row-fill-to-median: no effect / no effect.
- 92-config branch ablation sweep: no toxic component.
- Mild global output scalings on top of the final edit: mlp.down ×1.15/×0.85,
  experts.down ×1.15, o_proj ×1.1, layer_scalar L29→0.7 — all within noise
  (26–31/120 vs the installed candidate's 26–31 band).
- Raw completion / channel diagnostics, embedding forensics, duplicate/swap scans
  (incl. gate/up half-norm inversion and o_proj head-block checks), per-head and
  per-expert norm profiling: all clean except the L23/L29 lesion.

## Final verification

- `sandbox/model/model.safetensors` md5 `f23726454c02f9472be965b3ebcd3e1c`
  == `sandbox/m_l2329x2/model.safetensors` (the pass-3 candidate, unchanged since).
- Pristine shipped checkpoint preserved: `sandbox/model_backup_orig/model.safetensors`
  md5 `b69e321ec5f6a270f3e1dcb4bc313a97`.
- Pass counter: 3 of 10 used.


## Where everything lives

- Journal: `phase11/editor/journal.md` (chronological).
- All analysis scripts + outputs: `phase11/editor/*.py`, `*_out.txt`,
  `battery*_*.txt`.
- Model variants under `/home/user/wsp11/sandbox/m_*`; pristine backup at
  `/home/user/wsp11/sandbox/model_backup_orig`.
