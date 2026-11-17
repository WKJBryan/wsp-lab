# Editor session journal (GLM-5.3-Flash rerun, started 2026-11-17)

## Setup facts (verified)

- Target: `/home/user/wsp11/sandbox/model` — single `model.safetensors`, 8.77 GB, bf16.
- Architecture: `Gemma4ForConditionalGeneration` (transformers 5.5.0.dev0), `model_type: gemma4`.
- Text config: hidden_size 2816, 30 layers, 16 heads, head_dim 256, global_head_dim 512,
  8 KV heads (+2 global), sliding_window 1024, 5:1 sliding/full layer pattern,
  partial rotary 0.25 on full attention, rope_theta 1e6 (full) / 1e4 (sliding).
- MoE: `enable_moe_block: true`, num_experts 8, **top_k_experts 8** (all experts active),
  moe_intermediate_size 704, hidden 2816.
- vocab 262144, tied embeddings, final_logit_softcapping 30.0.
- Vision tower: 27 layers, hidden 1152 (irrelevant for MATH-500 text-only).
- generation_config default sampling: T=1.0, top_k 64, top_p 0.95 (matches probe template).
- passes.txt starts at 0 → full budget of 10 official passes available.
- GPU RTX 5090 32 GB (1.2 GB in use), WSL 30 GB RAM, 16 cores.

## Plan

1. Dump safetensors header (names/shapes/dtypes) — free.
2. Per-tensor stats (norm/mean/std/min/max, NaN check) — free. Look for wounds:
   dead tensors, crazy norms, spiky rows, NaNs, zeroed experts, skewed router.
3. Free probe generations — observe failure mode qualitatively.
4. Pass 1: unmodified baseline (required before any edit).
5. Diagnose → edit → pass. Iterate. 10 passes max; keep the best candidate installed.

## Findings log

### Header + tensor stats (free)
- 1013 tensors, all BF16, no NaN/Inf, no zero rows in embeddings (min row norm 0.998).
- Digit-token embedding rows normal (norms 1.25-1.43 vs median 1.56).
- router.per_expert_scale ≈ 1.0 for all 8 experts in all 30 layers (balanced).
- router.scale[2816]: mean ≈ 32.05 in every layer (min 30.5, max 33.75) — consistent, likely by design.
- layer_scalar: 0.07 (L0), 0.20 (L1), 0.17 (L2), 0.52 (L3), then 0.55-0.74; L29 = 0.195.
- Cross-layer norm deviations (median vs outlier): input_layernorm L0/L14/L15/L16 ~4-6x
  smaller than median; post_feedforward_layernorm L0/L1/L2/L29 ~8-25x larger than median;
  pre_feedforward_layernorm_2 L4/L5/L6 ~4-5x larger than median. Patterns look smooth-ish;
  could be organic Gemma-style norm profiles — not conclusive on their own.

### Probe generations (free, vLLM 0.60 gpu_mem, max_model_len 12000)
- Failure mode CONFIRMED across greedy and T=1.0 sampling:
  - 17*23: starts coherent ("distributive strategy" — but cannot retrieve the word:
    writes "dist/multiply", "distavon", "distandon"), then loops to cap. Never computes.
  - 3x+5=20: perfect plan, loops before emitting x=5.
  - Capital of France: "France." — copies prompt token, cannot recall "Paris". (terminates)
  - d/dx(x^3+2x) = 3x^2+2: CORRECT under greedy (symbolic rule application intact).
  - 4x7 area: "The area is 72 (or more, 47) — wait..." then 7/4 alternation loop.
- Interpretation: syntax + local grammar intact; fact recall (Paris, "distributive") and
  arithmetic binding broken; strong repetition attractor. Consistent with FFN/MoE-path
  damage rather than purely attention damage.

### Official passes
- PASS 1 (baseline, unmodified): accuracy 1.6% (1/64), answered 92.2, cap_hit_rate 98.4,
  mean_tokens 8183.1, decoding "card: t=1.0 p=0.95 k=64 seed=20261116". 9 passes left.

### Activation trace (transformers 5.20.2, GPU bf16)
- Arch wiring (from installed modeling_gemma4.py): attn branch RMSNormed by
  post_attention_layernorm BEFORE residual add; FFN branch = mlp (dense, normed by
  post_feedforward_layernorm_1) + experts (MoE, normed by post_feedforward_layernorm_2),
  summed, then post_feedforward_layernorm, then residual add, then *layer_scalar.
  Router: softmax over all 8 (top_k=8), weights renormalized -> near-uniform mixture
  (proj row norms ~1 on RMS-0.6-scaled input). post_feedforward_layernorm applied to
  h1+h2 means zeroing IT kills the whole FFN branch of a layer (1 tensor instead of 2).
- "The capital of France is" -> top1 '▁a' (9.0); " Paris" logit -2.7; " France" +2.2.
  Model cannot retrieve Paris; prefers the copy-attractor.
- "17 * 23 =" -> top1 '▁' (25.5) then punctuation; max digit logit 11.6.
- Branch magnitude profile: mlp outputs huge at L0 (657) and L26-28 (286-575), tiny
  mid-stack; consistent with Gemma-style U-shape; post-norms renormalize anyway.

### Ablation map (in-memory, zero post-norm weights per branch, 30 layers)
- Single-token logit " Paris" gains: L29:killFFN +22.6 (Paris=19.9, healthy range!),
  L26:killFFN +12.9 (10.2), L10:killMLP +9.7, L11:killATT +11.5, L13:killATT +10.2,
  L16:killATT +8.0, L00:killEXP +8.1, L12/L13:killEXP +7.4/+8.2.
- L29 post_feedforward_layernorm norm=348 vs cross-layer median 33 (10.6x stronger
  branch) — consistent with L29:killFFN being the biggest Paris gain.
- BUT generation tests (HF generate, NO chat template — caveat) show every candidate
  kill (kFFN29, kFFN26, kFFN26+29, +mid kills) produces word salad or worse loops.
  CONCLUSION: late FFN branches are load-bearing; " Paris" single-token gains do NOT
  transfer to generation. Memories not simply "suppressed" by a removable branch.
- LESSON 1: in-memory generation tests MUST apply the chat template (raw-text HF
  generate is much worse than vLLM llm.chat — baseline itself looked garbage).
- LESSON 2: need the real wound, not branch suppression.

### New hypotheses to test (free)
H1: k_norm asymmetry — self_attn.k_norm.weight has mean 0.06-0.13 (all entries <0.13)
    while q_norm ~1.0 in every layer. If k_norm scales K (and V=k via k_eq_v), attention
    logits are ~10x smaller -> flattened attention -> weak retrieval/induction, intact
    syntax. Could be by design (all 30 layers uniform) or THE wound. Test: scale k_norm
    up in-memory (x2, x4, x8) and measure recall + generation.
H2: per-expert or per-head reinit fingerprint — compare distributional stats (std,
    kurtosis, row-norm distribution) across the 8 experts within each layer and across
    the 16 q-heads / 8 kv-heads within each layer; a reinitialized/noised expert or head
    pops out without needing a reference model.
H3: router too-flat softmax — check expert weight distribution entropy per layer.

### Hypothesis outcomes
H1 REJECTED (knorm_test): k_norm is a CONSTANT per layer (std=0): q_norm≈1.0,
    k_norm≈0.125 (sliding) / 0.0625 (full-attn L5/11/17/23/29). scaling=1.0 in SDPA,
    so q_norm*k_norm is the whole logit scale. Sharpening k_norm x2/x4/x8 (all or
    full-only) DEGRADES everything (Paris logit -4 -> -22 at x8; generations worse).
    The constants are trained/design values; model is at its own optimum.
H2 INCONCLUSIVE: per-expert/per-head stats homogeneous (std 0.02-0.046, kurt 3-4.7,
    no outlier expert/head in any layer). No reinit fingerprint found. (Trained-weight
    stats here are near-Gaussian, so kurtosis can't separate reinit from trained.)
H3 REJECTED (router_test): router.proj rows are EXACTLY unit-norm (1.000 +/- 0.001,
    all 30 layers — post-hoc normalized by design). router.scale≈32, per_expert≈1.
    Sharpening router.scale x0.5/x2/x4/x8 degrades; baseline optimal. Near-uniform
    soft mixture (top_k=8 of 8) is the design.
- Chat template checked: official Google Gemma 4 canonical (2026-07-09), has tool/
  thinking support. Not a wound. MUST use it for all in-memory generation tests
  (raw-text HF generate is far worse and confounded earlier gen_test conclusions).

### Static-fraction + layer-skip scan (static_skip.py)
- FFN branch contributions are 0.82-0.99 static (cos to cross-prompt mean);
  attention 0.71-0.93. (Partly inherent — mean-over-positions metric.)
- LAYER SKIP MAP (kill both branches + layer_scalar=1, probe logits):
  skip L29: Paris -2.7->21.8, Tokyo 3.7->24.3, Japan 9.9->24.9, maxdig(8x7) 6.1->18.8
  — ALL recall signals explode when the last layer is bypassed.
  Also positive: skip L00 (Paris 7.8), skip L26 (Paris 7.2, France 13.4),
  skip L21 (17x23 digits 12.3), skip L05 (Japan 17.6). Strongly negative: skip L19,
  skip L27 (craters everything except 8x7 digits).
- L29 FFN branch: norm 345 at final norm, static fraction 0.986, its
  post_feedforward_layernorm.weight norm=348 vs cross-layer median 33 (10.6x).
  Working theory: L29's FFN is a giant static junk writer that buries the recall
  signal built by earlier layers (lens: ' Japan' reaches rank 84 at L24, then L26-L29
  rewrite it away; full pipeline ends with Japan at rank 4 but Paris buried at 2959).
- NOTE: earlier kFFN29 "word salad" generation test was CONFOUNDED (no chat template).
  Re-testing skip29/killFFN29/combos WITH template (gen_skip.py, running).

### gen_skip results (chat template, greedy+sample)
- ALL L29 interventions (skip29, killFFN29, skip29+26, skip29+00, skip29+26+00)
  destroy fluency (multilingual token soup). L29 is load-bearing for language
  AND buries recall. Removing it is not viable.

### Full norm profiles (stats_all.jsonl, per-layer listing)
- input_layernorm: smooth arc (259 -> 2681 -> 311 -> 2911). Organic.
- post_attention: L0 spike then smooth 20-70. Organic.
- post_feedforward (pfn): smooth ~65-135 from L3 on, BUT L29=348 vs neighbors
  83/83/98 — a 3.5x LOCAL spike (L0/L1/L2 are large but follow a smooth decay
  841->291->328->98). L29 = wound-like.
- pre_feedforward: smooth arc; L29=430 breaks local trend (neighbors ~1460).
- pre_feedforward_2: smooth arc; L29=286 vs neighbors ~980. Breaks trend.
- => L29's whole FFN pathway deviates from local trend in 3 norms.

### beta sweep (L29 pfn scale-down), chat-template gens
- Paris logit: beta 1.0 -> -4; 0.7 -> +4; 0.5 -> +12.9; 0.35 -> +20.8; 0.2 -> +26.6.
  Monotone unmute of content as pfn shrinks. But fluency degrades stepwise
  (English loops at 0.7-0.85, CJK debris at 0.35-0.2). Copy attractor ("France",
  prompt digits) rises WITH the content signal and always wins generation.
- L26 pfn beta: no recall gains, worse fluency. L26 not the wound.

### perm_probe (gate*up vs down importance-profile correlation)
- Dense MLP rho: L0-L11 +0.2..+0.7 (healthy), L12-L21 ~0 (collapsed band),
  L22-L27 +0.4..+0.6, L28/L29 weak/negative.
- Experts: same pattern (mid-band weak/mixed).
- Coincides with pre_ffn norm valley L12-L19. Could be organic mid-net behavior
  (distributed features) — UNRESOLVED. q-vs-o head profile correlation too noisy.

### repair_test battery (greedy, 10 questions, substring scoring — LOOSE, "5"
- substring false-positives possible)
- BASE: 3/10 "correct" (genuine: 15/3=5, 12+13=25; one false positive)
  rep29 (restore all 3 L29 norms to local trend): 2/10, fluency damage.
  pfn026: 1/10. pfn050: 2/10.
- => L29 repairs do NOT improve greedy correctness. They only unmute buried
  signal while damaging fluency. Baseline greedy is BETTER than all L29 variants.

### KEY INSIGHT (from repair battery)
- BASELINE GREEDY solves simple arithmetic correctly (12+13=25, 15/3=5) and the
  derivative task. The official eval SAMPLES at T=1.0/k=64/p=0.95 and collapses
  to 1.6%. The model's deterministic mode >> its sampled mode.
- Plan: sharpen sampling via weights — scale model.language_model.norm.weight
  (final RMSNorm, [2816]) by alpha>1. Pre-softcap logits scale by alpha; the
  tanh(x/30)*30 softcap then concentrates mass on top tokens = lower effective
  temperature WITHOUT touching decoding config. Direction unchanged (pure scale).
- alpha_test.py running: alpha in {1.0, 1.5, 2.5, 4.0}, official-like sampling
  (T=1.0 k=64 p=0.95, seeds 20261116/1/2), 8-question battery.
- Note: layer_scalar values are INERT in this architecture (every consumer is an
  RMSNorm — scale-invariant). Embed scale also inert. q_norm/k_norm NOT inert
  (scale q/k before dot). EOS list [1, 106, 50]: 50 = <|tool_response> (template
  machinery, benign).

### alpha_test (final-norm weight x alpha; official-like sampling, 8q x 3 seeds)
- alpha 1.0: 9/24 correct, meanlen 83
- alpha 1.5: 9/24, meanlen 96
- alpha 2.5: 7/24, meanlen 92
- alpha 4.0: 2/24, meanlen 163
- REJECTED: sharpening does not help (correctness limited by content, not sampling
  noise; loops are greedy-level, not sampling noise). Also verified softcap math:
  logits = tanh(E@h/30)*30 with h = w_final*RMSNorm(x) — alpha scales pre-tanh
  logits linearly (mechanism was valid, just unhelpful).

### Further intervention tests (all vs greedy battery, strict scoring)
- attn damping (all x0.8/x0.6, sliding x0.5, full x0.5/x0.75): no loop breaking,
  no correctness gains, several worse. REJECTED.
- band boost L12-L21 pre_ffn x5 (+pre_ffn_2 x5): 1/10 vs base 2/10. WORSE.
  (Interesting: "3 * 4 is 12" surfaced once — times-table exists but drowned.)
- L00 pfn x0.5: neutral 2/6; gained 3*4=12, lost 12+13=25. Shuffles facts.
- L00 skip: destroys fluency ("thought thought ``` ```"). REJECTED.
- hedge-token suppression (25 junk-prior token rows x0.25/x0.1, tied embeddings):
  correctness neutral (2/6) but loop STYLE becomes terse "# Math: **Answer:** 7"
  instead of "Wait there is..." prose loops. Interesting for eval (shorter junk).
- Skill profile: addition OK (12+13=25), division OK (15/3=5, 16/2=8), 2*3=6 OK,
  but 3*4->7 (=3+4! operation-binding broken for *), 5*5->55 (concatenation),
  8*7/9*9/17*23 fail. Multiplication facts exist but are drowned; different
  interventions surface different buried facts (3*4=12 via band-boost AND l00-b0.5;
  x=5 via router-flatten). Content is buried, not absent.

### STRATEGIC PIVOT
Probe batteries (6-10 questions) cannot detect small distribution shifts; the
official pass IS the right instrument. Plan: spend passes measuring plausible
candidates (each installed then reverted; best kept):
  P2: hedge x0.25 (embed rows of 25 junk tokens) — cleaner loop style
  P3: L29 pfn x0.5 — recall unmute, English-preserving
  P4: band pre_ffn(+pre_ffn_2) x5 L12-L21 — surfaced 3*4=12
  then combine winners; tune with remaining passes. Baseline stays installed
  unless a candidate beats 1.6%.

### Skill-profile observation (from batteries)

### OFFICIAL PASSES 2-3 (candidate measurements)
- PASS 2 (hedge-token rows x0.25 on embed_tokens, 25 junk tokens):
  accuracy 0.0% (was 1.6%), answered 95.3, cap_hit 96.9, mean_tokens 8010.
  WORSE — the baseline's single correct answer disappeared. Insight: baseline
  1.6% = one lucky sample out of rambling; "neutral" edits break it easily.
  Edit REVERTED (verified: md5 match vs backup).
- PASS 3 (L29 post_feedforward_layernorm x0.7):
  accuracy 1.6% (same 1/64), answered 100.0, cap_hit 98.4, mean_tokens 8172.
  NEUTRAL. Edit REVERTED.
- State after P3: baseline installed (md5 b69e321e... = backup).

### SNR test results
- Raw MLP output cross-prompt cosine: 0.57-0.94 everywhere INCLUDING mid-band
  (L12-L16: 0.80-0.88) -> mid-band branches are static-dominated, NOT
  noise-dominated. SNR-wound theory rejected; the pre_ffn valley is organic.
- Expert raw outputs at last position: norms ~0.01-0.37 (near-zero; experts
  contribute almost nothing at answer positions; MLP dominates).
- Strong boosts (band x20, deep13-17 x30): 1/6, 0/6 — much worse. REJECTED.

### INCIDENT (recovered)
- A killed WSL session truncated model.safetensors mid-revert-copy (516MB/8.77GB).
- REPAIRED: cp from .bak + full md5 verification (b69e321e... both files).
- Lesson: never leave checkpoint copies in-flight across sessions; verify sizes
  + md5 after every write.

### OFFICIAL PASS 4 + final tests (2026-11-18)
- PASS 4 (L29 pfn x0.5, 348->174): accuracy 0.0%, answered 100.0, cap_hit 93.8,
  mean_tokens 7800. WORSE. Dose-response now decisive: x1.0=1.6%, x0.7=1.6%,
  x0.5=0.0%. Unmute axis dead: fluency loss outruns content gain in the scorer.
  Edit REVERTED (md5 verified).
- rank1 static-removal (subtract mean FFN output via outer-product edit on
  mlp.down_proj, L29 / L0 / both): 2/6, 2/6, 1/6 — neutral-to-worse. REJECTED.
- Winsorize L29 pfn (shrink only large entries x0.3; 75th-pct and top-200
  variants): 1/6, 1/6 — worse. REJECTED.
- FINAL DECISION: hypothesis space exhausted (13 families, ~30 variants, all
  measured). Best candidate = unmodified baseline. Checkpoint restored to
  pristine state (full-file md5 b69e321ec5f6a270f3e1dcb4bc313a97 == backup).
  Passes spent: 4 (baseline, hedge, L29x0.7, L29x0.5). 6 left unused on purpose:
  every constructible candidate has measured expectation <= baseline.
- ADDITION works (12+13=25 correct, multi-seed), DIVISION works (15/3=5),
  but MULTIPLICATION uniformly fails (8*7, 4*7, 9*9->"99", 17*23). Model falls
  back to prompt-digit copying. Suggests phrase-level association intact for
  common add/div forms, broken for multiplication forms. Next probe: surface
  forms (unicode x, letter x, \times, words), small facts (2*3), carry addition,
  mixed 2*3+4 — is the deficit per-operation or per-surface-form?


