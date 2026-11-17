# Final report — Phase 11 editor session (GLM-5.3-Flash rerun, 2026-11-17/18)

## Final state

**The installed checkpoint is the UNMODIFIED baseline.**
`/home/user/wsp11/sandbox/model/model.safetensors` is byte-identical to the
pre-session backup (`model.safetensors.bak`, kept in the same directory):
full-file md5 `b69e321ec5f6a270f3e1dcb4bc313a97` for both files.
No other file in the model directory was modified.

## The exact final edit

**None (identity).** No edit is installed. Every edit attempted during the
session was reverted and verified (size + md5) before the next step. The
complete ledger of edits that were applied to the checkpoint at some point:

| pass | tensor | operation | magnitudes | official result |
|---|---|---|---|---|
| 1 | — | none (required baseline) | — | **acc 1.6%**, answered 92.2, cap_hit 98.4, mean_tokens 8183 |
| 2 | `model.language_model.embed_tokens.weight` | rows[25 hedge-token ids] x0.25 (ids 1595,1623,1852,2036,2653,2889,3153,3643,3685,3785,4346,4491,5583,6220,6861,11161,12152,13472,17498,18136,19238,20470,29202,38403,61531; row-block norm 13.453 -> 3.363) | x0.25 | acc **0.0%** — reverted |
| 3 | `model.language_model.layers.29.post_feedforward_layernorm.weight` | whole tensor x0.7 (norm 348.056 -> 243.790) | x0.7 | acc **1.6%** (neutral) — reverted |
| 4 | same tensor as pass 3 | whole tensor x0.5 (348.056 -> 174.028) | x0.5 | acc **0.0%** — reverted |

Passes spent: **4 of 10** (6 intentionally unused — see "why I stopped").
Best measured candidate = the unmodified baseline, which is what is installed.

## Diagnosis of the failure mode (and the evidence)

The model (Gemma4ForConditionalGeneration: 30 layers, hidden 2816, dense MLP
2112 + 8-expert MoE with top_k=8 per layer, tied embeddings, final logit
softcapping 30, official decoding t=1.0 / p=0.95 / k=64 / seed=20261116,
generation cap 8192) fails in a characteristic "fluent shell, empty core"
pattern:

1. **Fluency, formatting and rules are intact.** It produces well-structured,
   correctly-formatted step plans (bold headers, LaTeX, "Step 1/2/3"), answers
   single-step addition ("12 + 13 is **25**"), division ("15 divided by 3 is
   **5**"), and rule-based symbolic work (d/dx x^3+2x = 3x^2+2, correct).

2. **Associative recall and arithmetic facts are broken.** It cannot retrieve
   " Paris" after "The capital of France is" (logit -2.7 vs top token 9.0; the
   copy-attractor " France" beats it), cannot complete the word
   "distributive" ("dist trick", "distavon", "distandon"), and has no
   times-table: 3*4 -> 7 (= 3+4, wrong operation binding), 5*5 -> "55"
   (concatenation), 8*7/9*9/17*23 never computed (prompt digits copied).

3. **A repetition/hedging attractor derails long generations.** Official
   baseline: cap_hit_rate 98.4%, mean_tokens 8183 (nearly every generation
   rambles to the cap). The loops repeat plan templates and hedge tokens
   ("Wait...", "still", "actually") instead of computing.

Evidence trail (all measured, scripts in `phase11/editor/`):

- **Readout is buried by static junk, not absent.** Logit lens (raw E@h at
  every layer): recall tokens never surface mid-network (Paris worse than rank
  ~9000 in all 30 layers; " Japan" briefly reaches rank 84 at L24 and is
  rewritten away by L26-L29). Final hidden states are near-orthogonal to the
  whole embedding cone (top logit 9.0 = cos ~0.03).
- **Layer 29 is anomalous and functionally central.** Its
  `post_feedforward_layernorm.weight` has norm 348 vs 83/83/98 in neighboring
  layers (its `pre_feedforward_layernorm[._2]` also break local trend, 430 and
  286 vs ~1460/~980). Its FFN branch writes a vector of norm ~345 that is
  98.6% context-independent. Skipping layer 29 in-memory raises ALL recall
  probes massively (" Paris" -2.7 -> 21.8, " Tokyo" 3.7 -> 24.3, max digit
  logit for "8 * 7 =" 6.1 -> 18.8) — but generation becomes word salad:
  the same static write carries the language prior that keeps output
  in-distribution. Fluency and burial are entangled in one tensor.
- **Dose-response on the L29 knob (official passes):** x1.0 = 1.6%, x0.7 =
  1.6%, x0.5 = 0.0%. Unmuting content monotonically unmutes the copy-attractor
  and hedging too (in-memory: " Paris" -4 -> +27 as beta 1.0 -> 0.2), so the
  scorer's answer quality falls before the content gain can pay.
- **The buried content is real but scrambled.** Different single-knob
  interventions each surface *different* correct fragments (3*4=12 appeared
  under two different edits; x=5 under router flattening; the model can double
  and halve) — but never with net gains: fluency loss outruns content gain in
  every case.
- **What I ruled out (measured, rejected):** attention temperature
  (q_norm/k_norm constants are trained design values; sharpening degrades),
  router imbalance (proj rows are exactly unit-norm by design; scale changes
  degrade both directions), per-expert/per-head reinit fingerprints (none;
  distributional stats homogeneous), attention-branch damping (no loop
  reduction), final-norm global scale alpha=1.5-4 (neutral-to-worse; softcap
  tanh compresses the very gaps sharpening tries to widen), mid-band
  pre-FFN-norm repairs (the L12-L21 valley is an organic trained profile —
  branches there are static-dominated but not noise-dominated; boosts x5/x20/x30
  all hurt), rank-1 static-offset removal on mlp.down_proj (L0/L29), winsorized
  L29 pfn, L0/L26 interventions, hedge-token readout suppression (official
  pass: 0.0%).

**Interpretation:** the checkpoint behaves like a model whose associative
key-value circuitry (FFN facts, times-table, capital-of relations) has been
scrambled or disconnected while the language-prior pathway (static FFN writes,
late-layer bookends, attention copying) remains over-dominant. Because both
signals mix in the same residual stream and are re-normalized by the same RMS
norms, weight-space edits trade one for the other rather than separating them;
no arithmetic I could construct raised correct-content density faster than it
damaged extractable fluency.

## Passes spent

**4 of 10** (P1 baseline, P2 hedge suppression, P3 L29 x0.7, P4 L29 x0.5).
Full hypotheses are in `phase11/editor/ledger.md`; measurements and probes in
`phase11/editor/journal.md`; all probe scripts, stats and logs are in
`phase11/editor/`.

## Predicted effect of the final state

Identical to the official baseline measurement: **~1.6% accuracy (1/64)** on
the practice set under the fixed decoding card, ~92-100% answered, ~98%
cap-hit, mean_tokens ~8100. Concretely: fluent step-by-step attempts that
degenerate into hedging/loop rambling on anything requiring multiplication,
multi-step arithmetic, or factual recall; occasional correct answers on
single-step addition/division problems. I predict no material change from any
single-knob weight edit in the families explored; meaningfully improving this
checkpoint would require restoring its associative key-value circuitry, which
is not reachable by the direct-arithmetic edits available here.
