# Phase 11 results — prune for the bytes, program the repair

**Date:** 2026-11-16/17 · **Protocol:** `phase11/protocol.md` + Amendments #1–#2 (all committed
before the runs they govern) · **Engine:** vLLM 0.33 / WSL2 / RTX 5090, cap 8192 (Am. #1),
Gemma-4 model-card decoding, fixed seed, one scorer for every arm.

## The closing table (full MATH-500, 500 problems)

| Arm | What | Accuracy | Cap-hit | Mean tokens | Size |
|---|---|---|---|---|---|
| **A0** full 26B-A4B | all 128 experts (fp8 reference, disclosed) | **58.4%** | 0.2% | 669 | 51.6 GB bf16 |
| **A1** wound | 8 experts/layer, impact-selected | **1.0%** | 98.0% | 8,084 | 8.8 GB bf16 |
| **A2** WSP repair | editor's best candidate | **1.0%** (= A1) | 98.0% | 8,084 | 8.8 GB |
| **A2r** random control | n/a — the editor's final edit was "none" | — | — | — | — |

## Verdict (pre-registered): **NULL** — and the wound's autopsy is the result

The blind editor (Amendment #2: context-sterile subagent, API balance exhausted) spent
5 of 10 passes and restored the original checkpoint as its best candidate: **no
gradient-free repair was found** (A2 − A1 = 0.0, below the +2.5 PARTIAL floor). Its
194-tool-call investigation (`editor/journal.md`, `editor/final_report.md`) produced
three findings that matter more than the headline:

1. **The damage decomposition, measured.** Pass 2 (embedding-row boosts for stop /
   boxed / answer tokens) cut cap-hit **98.4% → 42%** and mean length 8,183 → 4,498
   tokens **at identical accuracy** — the termination *mechanism* is weight-edit
   repairable in one edit class. But teacher-forced probing showed the wound answers
   **0 of 21** bare arithmetic facts (6×7: echoes operands) with NLL 13–19 nats on
   clean text: the *knowledge* is deleted. A "format shell" — fluent grammar and
   protocol wrapped around missing computation. Weight edits could rewire the exit
   door; they could not re-grow what was amputated. Exactly what the phase-3/10 prior
   predicted (editors remove obstacles; they do not add abilities).
2. **Pruning provenance is undetectable blind.** The editor exhaustively lesioned the
   MoE structure (90 per-layer/expert lesions, routing sparsity, head roll-backs,
   every global scale knob) and concluded the weights were "exactly as trained" —
   without the original for reference, an amputation reads as a model that was always
   this way. Phase 2 found provenance-blindness in 17K-parameter nets; it replicates
   at 4.4B.
3. **Selection criterion is load-bearing and quantified.** Impact-top-8 vs
   frequency-top-8 overlap: 4.8 of 8, zero exact matches across 30 layers
   (`calibration_impact.json`) — "which experts you keep" is a real decision, and the
   two natural criteria disagree on ~40% of it.

## Pre-registered endpoints, scored

- **Primary (SUCCESS ≥ +5.0 over A1, control beaten):** NULL. Editor improved nothing.
- **P1 (on-task recovery ≥ 3× off-task):** superseded — no accuracy recovery occurred
  at any site; the recovery that did occur (termination) was mechanism-level, captured
  in finding 1.
- **P2 (editor discovers the pruning within 3 passes):** REFUTED with nuance — it
  mapped the structure completely and never suspected deletion (finding 2).

## The lobotomy trade, measured under one harness

At 94% expert removal: memory 51.6 → 8.8 GB buys a model that retains its language
and formatting but loses essentially all benchmark capability (58.4 → 1.0) — and the
loss is not recoverable without gradients (the author's own 91.7% agreement was a
narrow classification task *after* overnight LoRA healing; Jha's recovery fine-tune
recovers "about half"). Gradient-free repair closes none of this gap at this severity.
Milder ratios (50%, where Jha finds statistically undetectable on-domain loss) remain
the open repair regime — see ROADMAP.

## Amendments & integrity log

- **#1 (pre-baselines):** cap 32768 → 8192, all arms identically (A0 KV ceiling; vLLM
  gemma4 offload bug). A1's superseded 32K pilot: 4.7%.
- **#2 (pre-session):** editor substitution — external GLM-5.3-Flash API returned
  `1113 insufficient balance` on every model; replaced by a context-sterile subagent
  (fresh context, prompt committed pre-session, zero wound information). Blinding
  preserved structurally; contamination disclosure in protocol.
- Operational bugs found and fixed en route (all committed): einsum axis labels ×2,
  reshard for WSL mmap ENOMEM, self-masking pipeline exit codes, pydantic None
  rejection, mm-encoder budget, pkill self-match, shell-variable quoting in WSL
  wrappers. Issue-worthy patterns logged in CHANGELOG.

## The editor grid (Amendments #3–#3d, 2026-11-17/18): one wound, five blind surgeons

User-directed extension: "isn't GLM-5.3 more capable — why no run for it?" and "yes
mimo also… we should still do both." Five context-sterile editors, same wound, same
committed blind prompt, same 10-pass budget, same evaluator.

| # | Editor | Family / tier | Passes | Final MATH-500 | Verdict |
|---|---|---|---|---|---|
| 1 | GLM-5.3 (subagent) | zai / big | 5/10 | = A1 (1.0%) | **NULL** |
| 2 | GLM-5.3 (subagent) | zai / big | 3/10 | 0.8% (A2r 1.6%) | **NULL** |
| 3 | GLM-5.3-Flash | zai / fast | 4/10 | = A1 (baseline restored, md5-identical) | **NULL** |
| 4 | MiMo-v2.6-flash | xiaomi / fast | 1/10 | — (2 spawns, 9 h, becalmed) | **STALLED** |
| 5 | MiMo-v2.6-pro | xiaomi / big | 10/10 | [pending final] (control [pending]) | **NULL** (practice tie; all secondary metrics improved) |

**Pre-registered prediction (Am. #3): CONFIRMED ×4** — no editor, in either family or
tier, recovered any accuracy. The damage-class claim now rests on five blind sessions,
two 500-problem finals with matched controls, and ~1,000 tool calls of analysis.

### What the grid found beyond the NULLs

1. **Convergent diagnosis, independent words:** "format shell" (1), "pruning inferred,
   knowledge did not survive" (2), "fluent shell, empty core" (3), non-viable (4),
   "half-knows: memorized facts at top-64 ranks, computed results absent" (5). Five
   isolated contexts, one conclusion.
2. **The late layers keep volunteering as suspects — independently.** Attempt 2:
   near-dead RoPE-region query rows in L23/29. Attempt 3: L29's post-FFN norm 4×
   neighbor norms, 98.6% static, bypass unmutes recall. Attempt 5 (logit-lens):
   correct answers surface at L25 and are destroyed by L26–29 (`layer_scalar` makes
   late blocks dominate the final state ~13:1). Three editors, zero communication,
   one region.
3. **Termination remains the repairable half, everywhere.** Attempt 1: embedding
   boosts, cap-hit 98.4→42%. Attempt 5: stop-token rows ×1.6 alone, cap-hit 98→36
   (4.7% in its composite dynamics package, mean length 1,172 tokens) — and its
   final-norm ×0.4 cured softcap saturation (logits pinned at the 30.0 tanh cap)
   while preserving accuracy.
4. **Pro > flash for long agentic sessions, cross-family:** MiMo-flash becalmed twice
   (phase-9's flash-tier bottleneck replicated); MiMo-pro completed all 10 passes and
   produced the grid's deepest analysis. GLM-5.3-Flash held the session fine — the
   bottleneck is family×format, not tier alone.

## What this does to Phase 12 (cross-architecture atlas)

The atlas question sharpens: **same symptom, different cause, different repairability.**
Qwen 3.5-0.8B (phases 3–10): knowledge present, termination broken → single-row edits
recover real accuracy (6.0 → 27.4). Gemma wound: knowledge deleted, termination broken →
same edit class recovers the mechanism (cap-hit 98 → 42) but no accuracy. C2
(termination-circuit correspondence) now has a measured hook on both sides: both
models' exit mechanisms responded to embedding-row boosts. That rhyme is the first
atlas entry.
