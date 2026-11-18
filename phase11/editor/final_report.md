# Final report — Phase 11 weight-editor session

Target: `~/wsp11/sandbox/model` (WSL), a Gemma4-class multimodal MoE LM
(`Gemma4ForConditionalGeneration`, 30 layers, hidden 2816, 8-expert MoE +
dense MLP per layer, tied embeddings, vocab 262144, bf16, single 8.7 GB shard).
Goal: improve MATH-500 accuracy by direct weight arithmetic (no gradients).
All 10 official passes are spent. Full reasoning log: `journal.md`;
hypothesis ledger: `ledger.md`.

## 1. The exact final edit (installed in `sandbox/model`)

One operation, one tensor (verified: exactly 1 of 1013 tensors differs from the
pristine backup):

| tensor | operation | magnitude |
|---|---|---|
| `model.language_model.norm.weight` (final RMSNorm gain, shape (2816,), bf16) | uniform scalar multiply | x 0.4 (mean 29.26 -> 11.70, std 84.50 -> 33.80, max 588 -> 235.2) |

Everything else is bit-identical to the original checkpoint
(`model.safetensors.bak`, kept in `sandbox/model/`). Recipe:
`recipe_c12.json` in this directory; applied by `edit_weights.py`.

Why this edit is the final state: it is the ONLY edit of the ten evaluated that
measured the best accuracy on the fixed eval while improving every secondary
metric (see pass table). The two "big dynamics" packages are documented in
section 5 for re-application if behavior matters more than the last digit of
this particular 64-problem draw.

## 2. Diagnosis of the failure mode

**Observed baseline (pass 1, unmodified):** accuracy 1.6%, answered 92.2%,
cap_hit_rate 98.4%, mean_tokens 8183 (cap = 8192). Probes (`probe_out_a.txt`)
show the exact pathology: the first 50-100 tokens are topically on-task but
word-corrupted ("near-ungendo method", "Heractor's method", "muchingly"), then
a hard repetition loop ("(Wait, too...)" / "$17 \times 23 = 75 \times 17$" /
"2 x 2 x 2 ..." x80) until the cap; stop tokens (`<turn|>` 106, `<eos>` 1,
`<|tool_response>` 50) never appear in 9000 tokens of generation.

**Failure, decomposed (evidence in `journal.md`):**

1. **Degenerate sampling from softcap saturation.** Generation-time raw logits
   reach the `final_logit_softcapping = 30.0` tanh cap (greedy top post-cap
   values 22-29). Many tokens tie at the cap -> sampling at t=1.0/k=64 is
   near-uniform over the tied band -> word-level corruption and the
   repetition-loop attractor. Confirmed experimentally: shrinking the
   readout scale (final-norm gains x0.4) pulls logits out of saturation and
   markedly cleans sampling (see passes 2,3,9).

2. **The model half-knows answers but cannot commit.** Teacher-forced digit
   ranks at the final state (per-token, `early_exit_test.py`):
   `2^10=1024` -> [6, 1, 97, 5]; `12*12=144` -> [7, 184, 2];
   `sqrt(144)=12` -> [68, 1]; `17*23=391` -> [360, 1245, 71];
   `7*8=56` -> [18835, 125]. Memorized facts sit inside/near top-64
   (samplable); computed multi-step results do not. Fact recall is similarly
   half-alive: 'Celsius' beats its semantic siblings at some layers but
   ~10k unrelated tokens outrank it at the readout; France->Paris lands at
   rank 3487 with sibling confusion (Rome/Berlin/Kelvin-type substitutions).

3. **Correct answers exist mid-network and are drowned late.** Logit-lens
   trajectory (`lens_v2.py`): Water->'Celsius' full-vocab rank 47 at layer 3,
   ~100-700 through layers 4-9, then collapses (7k-35k at layers 10-23),
   recovers to **rank 6 at layer 25**, and is destroyed to rank 36890 by
   layer 27. The readout itself is healthy (rank 6 was obtained with the
   current final-norm gain and tied E). Mechanism: `layer_scalar` is an
   nn.Buffer multiplying each layer's FULL output (`hidden_states *=
   layer_scalar` in both transformers and vLLM gemma4 code), so block
   contributions to the final state are weighted by the product of later
   scalars: s26..s29 = 0.80*0.73*0.68*0.195 ~ 0.077 — the late blocks'
   output dominates the final state ~13:1 over everything computed before.

4. **Attention routing is healthy; retrieval content is not.** Attention maps
   (`attn_probe.py`) are focused and precise (L17 heads attend exactly at the
   "137" token position), yet in-context copy fails at every depth
   (P(' 137') = 1e-4 with the answer in the prompt). Value-path/association
   content is damaged while q/k routing works.

5. **No static weight fingerprint exists.** Exhaustive forensics found nothing
   that static arithmetic can invert: no zero rows; no duplicate head blocks
   (v/k block cosines ~0.01); no exact or scaled tensor copies (multiset scan
   of all 241 2816-dim vectors); no sorted/permutation signatures; no rank-1
   spectral spikes beyond natural anisotropy (E's dominant direction has
   near-constant projections = softmax-invariant common component); no
   row/column scale outliers (the high-norm "hub" rows are natural
   function-word embedding growth; clipping them changes nothing); router
   semantics identical in vLLM and transformers; gate/up ordering correct
   (swap tests worsen everything); `layer_scalar` values are tuned optima
   (0 -> unigram soup, 1 -> HTML soup, as-is -> coherent-but-loopy).
   The residual damage is distributed content-level (value path / FFN
   associations) and no blind static edit restores it.

## 3. Official passes spent (all 10)

Fixed 64-problem MATH-500 practice set, fixed decoding (t=1.0 p=0.95 k=64,
seed=20261116) -> each pass is a deterministic draw per config.

| pass | config | accuracy | answered | cap_hit | mean_tokens |
|---|---|---|---|---|---|
| 1 | baseline (pristine) | **1.6** | 92.2 | 98.4 | 8183 |
| 2 | s26-29=0.8 + g x0.4 + stop x1.6 | 0.0 | 96.9 | 6.2 | 1931 |
| 3 | s24-29=1.0 + g x0.4 + stop x1.6 | 0.0 | 98.4 | 4.7 | 1091 |
| 4 | s24-29=1.0 + stop x1.6 | 0.0 | 100.0 | 9.4 | 1828 |
| 5 | s24-29=1.0 only | 0.0 | 98.4 | 93.8 | 7743 |
| 6 | s26-29=0.95 + g x0.4 + stop x1.6 | 0.0 | 96.9 | 12.5 | 2587 |
| 7 | s24-29=1.0 + g x2.0 + stop x1.6 | 0.0 | 100.0 | 1.6 | 280 |
| 8 | stop x1.6 only (ablation) | 0.0 | 96.9 | 35.9 | 4072 |
| 9 | **g x0.4 only (ablation)** | **1.6** | 96.9 | 90.6 | 7820 |
| 10 | g x0.4 + stop x1.6 | 0.0 | 96.9 | 4.7 | 1172 |

(g = `model.language_model.norm.weight`; stop = embed rows 1/50/106;
sN = `layers.N.layer_scalar`.)

**Ablation conclusions (the useful science):**
- Stop-token rows x1.6 ARE the termination mechanism: cap_hit 98.4 -> 35.9
  alone, -> 4.7-6.2 combined with g x0.4 (and answered rises to 97-100%).
- final-norm gains x0.4 is the anti-saturation mechanism and PRESERVES the
  best measured accuracy (1.6) while raising answered 92.2 -> 96.9.
- layer_scalar rebalancing lifts answer-token ranks (fact ranks 2-3x better
  at s26-29 in [0.8, 1.0]; math digits reach top-64 at s24-29=1.0) but
  perturbs the sampling draw; combined with the dynamics package it did not
  raise accuracy on this set.
- Digit-row boosts are destructive at x2.0 (digit spam); x1.15 is inert.
- Early exit (skipping late layers) is worse: hub rows hijack the
  intermediate-state readouts.

## 4. Honest accuracy assessment

No weight edit improved measured MATH-500 accuracy over the pristine
baseline's 1.6% (1 of 64). The best achieved is 1.6% twice (P1 pristine,
P9 g x0.4). Every other draw landed 0-1 problems; with n=64 the 0-vs-1.6
difference is one problem (P(0/64 | true p=1.6%) = 36%), so the variants are
statistically indistinguishable from the baseline but none measured better.
The bottleneck is answer correctness: the model's computed results are wrong
(diagnosis 2), and static arithmetic cannot rebuild the broken value-path /
association content (diagnosis 5).

## 5. Predicted effect of the installed edit (c12)

Re-running the official eval on the installed model will reproduce, exactly
(deterministic decode):
- accuracy 1.6% (unchanged from baseline — the best measured level),
- answered 96.9% (baseline 92.2%),
- cap_hit_rate 90.6% (baseline 98.4%),
- mean_tokens ~7820 (baseline 8183).

If behavioral health is preferred over the last fraction of this draw, the
best dynamics package measured is `recipe_c13.json` (g x0.4 + stop rows x1.6:
cap 4.7%, mean 1172, answered 96.9%) or `recipe_composite1.json` /
the pass-3 recipe (cap 4.7-6.2%, mean 1091-1931, answered 96.9-98.4%) — all
measured accuracy 0.0 on this set (i.e., parity within the one-problem noise
band). One-line re-apply: `~/wsp3c-venv/bin/python edit_weights.py
recipe_c13.json` from this directory.

## 6. Artifacts (this directory)

- `journal.md` — full reasoning/evidence log; `ledger.md` — pass hypotheses.
- `edit_weights.py` + `recipe_*.json` — reproducible restore-and-apply edits
  (backup: `sandbox/model/model.safetensors.bak`).
- Probe/analysis scripts and outputs: `probe_batch.py`, `probe_out_*.txt`,
  `logit_probe.py`, `attn_probe.py`, `lens_v2.py`, `lens_v3.py`,
  `early_exit_test.py`, `math_scalar_sweep.py`, `scalar_sweep*.py`,
  `skip_test2.py`, `g_test*.py`, `rowclip_test.py`, `scan_*.py`,
  `verify_restore.py`, `exp_harness*.py`, `sweep_scale.py`.