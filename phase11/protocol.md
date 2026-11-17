# Pre-registered protocol — Phase 11: prune for the bytes, program the repair

Committed BEFORE any download, pruning, or run. The intersection no published work holds:
gradient-free (weight-space) repair of an expert-pruned MoE.

## The gap it closes

Two viral threads, Nov 2026, never combined: the Gemma-4 "lobotomy" (r/LocalLLaMA, Nov 4:
128→8 experts/layer, −85% weights, 2.75 GB Q4, 91.7% single-task agreement — healed with
*gradient* LoRA) and weight-space programming (the original post, Nov 2; wsp-lab phases 1–10:
blind editors repair broken mechanisms, never add abilities). Nearest published neighbors
repair other damage classes: EoRA/DSnoT (dense unstructured pruning), MoRA (gradient router
training). The stunt author was pointed at WSP in his own thread and declined to look. Nobody
has measured how expert-pruning damage decomposes into weight-edit-repairable vs not.

## Design

- **Model:** `google/gemma-4-26B-A4B-it` (ungated; verified: 30 layers × 128 experts,
  moe_intermediate 704, hidden 2816, `Gemma4ForConditionalGeneration`; multimodal towers
  retained but unused by this eval). ~51.6 GB bf16 download; pruned checkpoint target ≈ 4–6 GB.
- **Prune (harness-side, deterministic, pre-registered):** keep 8 of 128 experts per layer,
  selected by **ablation impact** — mean |Δhidden-state| with each expert's contribution
  zeroed, measured over a fixed 64-problem calibration set, forward passes only — NOT
  activation frequency (the stunt author's reported failure mode). Shared/always-on expert
  kept. Delete the other 120 experts' tensors, slice router rows, renormalize over survivors
  (standard REAP/REAM bookkeeping). Export as a standard HF checkpoint, config updated.
- **Editor:** GLM-5.3-Flash, step-driven solo (phase-10 format), blind. Standing instruction
  verbatim: *"Improve this model's MATH-500 accuracy. You may edit weights directly. No
  gradient training."* — zero mention of pruning or MoE. Budget: 10 practice passes
  (64 problems, eval_pass.py counter-enforced, hypotheses pre-logged) + unlimited offline
  probing (standing symmetric rule since phase 4).
- **Arms:**
  - **A0** full model reference (evaluated at fp8 — 52 GB bf16 exceeds 32 GB VRAM; disclosed)
  - **A1** wound: pruned, unrepaired
  - **A2** pruned + WSP repair (the candidate)
  - **A2r** pruned + matched-norm random-edit control
  - **A3** dense-class reference (~4B): published numbers, not run locally
- **Decoding:** Gemma-4 model-card settings, identical for every arm. Final = full MATH-500
  (500 problems, 32K cap, 1 sample, phase-3c rescore extraction), phase-3c harness (vLLM
  0.33.0 in WSL2, arch support verified 2026-11-16).

## Amendment #1 (2026-11-16, before any arm completed at 32K)

Generation cap 32768 → **8192 for every arm identically**. Reason: the A0 fp8 weights
(26 GB) plus the KV cache of a single 34K-token sequence (3.97 GiB, measured) exceed
the RTX 5090's 32 GB; vLLM 0.33's cpu_offload path crashes on Gemma4 (meta-tensor
bug, `uva.py:114`). Healthy Gemma-4 math solutions terminate in ~1-3K tokens, so 8K
preserves the wound-vs-healthy distinction; the wound loops at any cap. Applied
before the first complete A0/A1 baseline pair; A1's 32K pilot (4.7%) is superseded.

## Amendment #2 (2026-11-17, ~04:00, before the editor session)

**Editor substitution.** The pre-registered editor (GLM-5.3-Flash via external API) is
unavailable: the Z.ai account returns `1113 insufficient balance / no resource package`
on every model. No recharge is possible unattended, and the user's OpenRouter key will
not be used (spends money without instruction). Substituted editor: a **context-sterile
subagent** on the harness session model, per the user's standing mandate ("why not let
you try and do it? if one of you is not enough, spawn subagents"). Blinding is
preserved structurally: the subagent receives a fresh context containing ONLY the
standing instruction (verbatim), the sandbox model path, the evaluator mechanics, the
budget rules, and the forbidden-techniques list. Zero information about the wound's
construction, pruning, MoE, experts, calibration, or the lobotomy literature passes
into its prompt. The subagent may read the model's own files (config/weights — same
information any editor in phases 4/10 had). Contamination disclosure: the harness
parent (this agent) constructed the wound and knows the diagnosis; the mitigation is
context separation, verified by the prompt being committed alongside this amendment
before the session runs. All other mechanics unchanged: 10-pass budget (counter
enforced), pre-logged hypotheses, unlimited offline probing, forbidden gradients/
LoRA/distillation/prompt changes, identical evaluator.

## Amendment #3 (2026-11-17, user-directed, before the rerun session)

**A2 rerun with the originally-registered editor.** Post-session discovery: the editor
models (GLM-5.3-Flash, mimo) are configured as host-routed providers in the user's
environment — the 4 AM "insufficient balance" wall was the legacy raw-API path
(`GLM_API_KEY` + api.z.ai), not a real shortage (user: "why not use workflow and
activate the keys and models I have inserted into you already?" — correct). The
pre-registered editor is therefore runnable as a context-sterile subagent pinned to
`GLM-5.3-Flash`. Scope: **A2 arm only.** A0/A1 finals stand (weight-identical
checkpoints, unchanged settings). The first A2 result (session-model editor, NULL,
5/10 passes) is retained and reported as **A2-session**; the rerun reports as
**A2-flash** with its own ledger (`editor_flash/`), fresh sterile sandbox, identical
blind prompt, identical 10-pass budget and rules. Outcome either way strengthens the
phase: A2-flash SUCCESS overturns the NULL with the registered editor; A2-flash NULL
confirms it cross-editor (the phase-4 tradition, now on the prune wound).

## Pre-registered endpoints

- **Primary** (final MATH-500 accuracy):
  - **SUCCESS:** A2 − A1 ≥ +5.0 pts AND A2 − A2r ≥ +2.5 pts
  - **PARTIAL:** A2 − A1 in [+2.5, +5.0) with control beaten
  - **NULL:** otherwise
- **Damage-split readout** (the scientific payload, reported regardless of verdict):
  on-task recovery (MATH-500) vs off-task recovery (MMLU subset, ~200 items) vs
  agreement-with-A0 (fixed 200-problem set). **Prediction P1 (falsifiable):** on-task
  recovery ≥ 3× off-task recovery — editors fix misdirected computation, not deleted
  knowledge. **P2:** the editor discovers the pruning blind (diagnosis ledger records it)
  within its first 3 passes.
- **Secondary:** passes spent; edit anatomy (router rows / embeddings / MLP columns / other);
  tok/s + GB table for A0/A1/A2 (the field's missing pruned-vs-dense speed/size numbers).

## Forbidden

Gradient steps, LoRA, distillation, any training loop; prompt or decoding changes relative to
the model card; perplexity as a lone metric (Jha 2607.16721 finding #3).

## Standing disclosures

Protocol author (this agent) researched the lobotomy literature on 2026-11-16; the editor is
an external API model blind to this repository and receives no pruning information. Harness
author ≠ editor; the wound is mechanical (no authored bug to know in advance). A0 at fp8 is a
quantization disclosure, applied only to the reference arm. Open item: confirm the top-k key
name in the downloaded config at build time (expected `num_experts_per_tok: 8`).
