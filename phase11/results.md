# Phase 11 results — prune for the bytes, program the repair

**Date:** 2026-11-16/17 · **Protocol:** `phase11/protocol.md` + Amendments #1–#2 (all committed
before the runs they govern) · **Engine:** vLLM 0.33 / WSL2 / RTX 5090, cap 8192 (Am. #1),
Gemma-4 model-card decoding, fixed seed, one scorer for every arm.

**Corrected 2026-11-24 (ISSUES #19, #22):** the full model's score was re-scored with the
corrected answer checker, the wound is described as it was built, and the evidence below is
restated. A withdrawn reading is kept, in one line, where it stood.

## The closing table (full MATH-500, 500 problems)

| Arm | What | Accuracy | Cap-hit | Mean tokens | Size |
|---|---|---|---|---|---|
| **A0** full 26B-A4B | all 128 experts (fp8 reference, disclosed) | **76.4%** (382 of 500; 58.4% as first reported) | 0.2% | 669 | 51.6 GB bf16 |
| **A1** wound | 8 experts kept per layer, impact-selected; expert selection disabled | **1.2%** (6 of 500) | 98.0% | 8,084 | 8.8 GB bf16 |
| **A2** WSP repair | attempt 1's best candidate: the wound restored, unedited | **1.2%** (= A1) | 98.0% | 8,084 | 8.8 GB |
| **A2r** random control | none completed: attempt 1's final edit was "none" | — | — | — | — |

## Verdict (pre-registered): **NULL** — and the wound's autopsy is the result

The blind editor (Amendment #2: context-sterile subagent, API balance exhausted; attempt 1)
spent 5 of 10 passes and restored the original checkpoint as its best candidate: **no
gradient-free repair was found** (A2 − A1 = 0.0, below the +2.5 PARTIAL floor). Its
194-tool-call investigation (`editor_a2_attempt1/journal.md`,
`editor_a2_attempt1/final_report.md`) produced three findings. They were restated on
2026-11-24 (ISSUES #22); each keeps its earlier reading in italics.

1. **Most capability is lost; what remains is not expressed in free-running generation.**
   What weight edits reach is termination. On a 64-problem practice pass (pass 2:
   embedding-row boosts for stop / boxed / answer tokens) they cut cap-hits from 63 of 64 to
   27 of 64 (98.4% → 42.2%) and mean length from 8,183 to 4,498 tokens, with accuracy
   indistinguishable: 1 correct of 64 before and 2 after (a 95% interval for 1 of 64 is 0.3%
   to 8.3%). The remainder is present but not expressed: with one weight edit (the final
   norm × 0.4, attempt 5) the wound produced ten correct answers of 500 against its six,
   including four complete correct derivations (problems m13, m191, m345, m469). That
   difference is not statistically distinguishable, and a second build of the same edit
   scored 7 of 500. No gradient-free edit recovered a statistically demonstrable amount of
   accuracy; the largest observed gain is attempt 5's 10 of 500 against the wound's 6.
   *Withdrawn 2026-11-24: the reading that the knowledge did not survive pruning, leaving
   a shell of grammar and protocol around missing computation.* It rested on two numbers
   that do not hold. The arithmetic-facts test (`editor_a2_attempt1/arith_bench.py`) counted
   a fact as right only if the model's single top-scoring token contained the whole answer,
   and this tokenizer
   emits one token per digit, so 20 of the 21 answers (all with two or more digits) could
   never be scored; only gcd(12,18) = 6 was scorable. The same editor's own trace shows the
   wound predicting the digits of "391 - 11 = 380" at ranks 1, 1 and 2. The negative
   log-likelihood on clean text came from a script that scores raw text with no chat
   template, was never run on the full model, and two other editors report the same "token
   salad" in that regime.
2. **Pruning provenance: no editor identified it from the weights alone.** Attempt 1
   exhaustively lesioned the MoE structure (90 per-layer/expert lesions, routing sparsity,
   head roll-backs, every global scale knob) and found nothing it could call damage: without
   the original for reference, an amputation reads as a model that was always this way. Phase
   2 found provenance-blindness in 17K-parameter nets; this is one replication at 4.4B.
   Attempt 2 did infer the pruning, but it learned that the model was pruned from file names
   outside its sandbox ("prune_summary.json", "smoke_wound.json"), so that diagnosis did not
   come from the weights alone. Attempts 3 and 5 never raised it. *Until 2026-11-24 this read
   "pruning provenance is undetectable blind".*
3. **Which experts are kept is a near tie, and the criterion was never tested.** Experts
   were selected on 64 problems (4,982 tokens of question text), in int8. Impact-top-8 and
   frequency-top-8 overlap in 4.8 of 8, with zero exact matches across 30 layers
   (`calibration_impact.json`), so the two criteria disagree on about 40% of the choice. But
   the ranking's stability is unmeasured, the cut is a near tie (the 8th-to-9th impact ratio
   is below 1.05 in 13 of 30 layers; minimum 1.006), and 59 of the 240 kept experts are
   outside the top 16 by routing frequency. The registered off-task measurement (an MMLU
   subset, and agreement with the full model) was never run. So "which experts you keep is a real
   decision" is not established here. *Until 2026-11-24 this read "selection criterion is
   load-bearing and quantified".*

## Pre-registered endpoints, scored

- **Primary (SUCCESS ≥ +5.0 over A1, control beaten):** NULL. The largest observed gain over
  the wound is attempt 5's 10 of 500 against the wound's 6, which is not statistically
  distinguishable, and no matched control was completed.
- **P1 (on-task recovery ≥ 3× off-task):** not scorable — the registered off-task measurement
  (an MMLU subset and agreement with the full model) was never run. (First written as
  "superseded — no accuracy recovery occurred at any site".)
- **P2 (editor discovers the pruning within 3 passes):** attempt 1 never discovered it.
  Attempt 2 did infer the pruning, but from file names outside its sandbox (finding 2), so
  it is not a test of discovery from the weights. Recorded on 2026-11-17 as "REFUTED with
  nuance".

## The lobotomy trade, measured under one harness

At 94% expert removal: memory 51.6 → 8.8 GB buys a model that keeps its language and
formatting but loses most benchmark capability (76.4 → 1.2 on MATH-500). The wound is two
injuries, not one. `prune.py` keeps 8 experts per layer and leaves the active-expert count
at 8, so expert selection is switched off and every survivor runs on every token (the
script's own docstring calls it "the lobotomy regime; the router is effectively bypassed").
The wound therefore removes conditional computation as well as experts, uses the same 4.39
billion parameters per token as the full model (it saves memory, 51.6 to 8.8 GB, not
computation), and its collapse cannot be attributed to deletion alone. No final separates the
two. One practice pass with the same 8 experts and selection restored (top-k 1) was run on
2026-11-24 and scored 0 of 64, the same as the wound as built rerun in the same session (see
the last section).

No gradient-free edit recovered a statistically demonstrable amount of accuracy (above).
Whether gradient training recovers more was not tested here (the author's own 91.7%
agreement was a narrow classification task *after* overnight LoRA healing; Jha's recovery
fine-tune recovers "about half"). Milder ratios (50%, where Jha finds statistically
undetectable on-domain loss) remain the open repair regime — see ROADMAP. *Until 2026-11-24
this paragraph said the loss "is not recoverable without gradients" and that gradient-free
repair "closes none of this gap".*

## Amendments & integrity log

- **#1 (pre-baselines):** cap 32768 → 8192, all arms identically (A0 KV ceiling; vLLM
  gemma4 offload bug). A1's superseded 32K pilot: 4.7% (as recorded with the old checker).
- **#2 (pre-session):** editor substitution — external GLM-5.3-Flash API returned
  `1113 insufficient balance` on every model; replaced by a context-sterile subagent
  (fresh context, task instructions committed before the session, zero wound information).
  Blinding preserved structurally; contamination disclosure in protocol. Not in every later
  attempt: attempt 2 learned of the pruning from file names outside its sandbox.
- Operational bugs found and fixed en route (all committed): einsum axis labels ×2,
  reshard for WSL mmap ENOMEM, self-masking pipeline exit codes, pydantic None
  rejection, mm-encoder budget, pkill self-match, shell-variable quoting in WSL
  wrappers. Issue-worthy patterns logged in CHANGELOG.
- **Re-scoring (2026-11-24, ISSUES #19):** the evaluator had used a private copy of the
  defective answer checker. Re-scored with the corrected one (`phase11/rescore.py`; the
  checker was corrected twice that day): full model 76.4% on the final (382 of 500; 58.4% as
  first reported) and 73.4% on its practice pass (47 of 64; 53.1% as first reported); the
  wound 1.2% on the final (6 of 500; 1.0% as first reported) and attempt 1's second practice
  pass 3.1% (2 of 64; 1.6% as first reported). The other wound arms are unchanged (0.8, 2.0,
  1.6, 1.4; the wound's practice pass 1.6).
- **Files (2026-11-24, ISSUES #21):** attempt 1's practice files are the pinned copies
  under `editor_a2_attempt1/`; the top-level `ed_pass*.json` were overwritten by later
  attempts.

## The editor grid (Amendments #3–#3d, 2026-11-17/18): one wound, five blind surgeons

User-directed extension: "isn't GLM-5.3 more capable — why no run for it?" and "yes
mimo also… we should still do both." Five context-sterile editors, same wound, same
committed task instructions, same 10-pass budget, same evaluator.

| # | Editor | Family / tier | Passes | Final MATH-500 | Verdict |
|---|---|---|---|---|---|
| 1 | GLM-5.3 (subagent) | zai / big | 5/10 | = A1 (1.2%; final edit "none") | **NULL** |
| 2 | GLM-5.3 (subagent) | zai / big | 3/10 | 0.8% (4 of 500); control arm 1.6% (8 of 500), not matched | **NULL** |
| 3 | GLM-5.3-Flash | zai / fast | 4/10 | = A1 (baseline restored, md5-identical) | **NULL** |
| 4 | MiMo-v2.6-flash | xiaomi / fast | 1/10 | — (2 spawns, 9 h, becalmed) | **STALLED** |
| 5 | MiMo-v2.6-pro | xiaomi / big | 10/10 | 2.0% (10 of 500) against the wound's 1.2% (6 of 500); cap-hit 79.4% (397 of 500) against 98.0% (490 of 500); mean length 7,067 against 8,084 tokens | **NULL** |

**Pre-registered prediction (Am. #3): confirmed in the four attempts that produced a final;
one stalled.** No editor, in either family or tier, recovered a statistically demonstrable
amount of accuracy (the largest observed gain is attempt 5's 10 of 500 against the wound's
6). **No matched control was completed** for any final: attempt 1's final edit was "none";
attempt 2's control changed 374 rows in layer 29 against the editor's 7, with a change of
size 42.2 against the editor's 4.3; and attempt 5's first control build picked the edited
tensor itself (its filter matched nothing, leaving a pool of one). The rebuilt control (layer
21 `pre_feedforward_layernorm_2` × 0.4) was stopped at about 55 of 500 problems by the GPU
collision of ISSUES #18 and has no result file. `a2r5_final.json` (1.4%, 7 of 500) is a
second build of the editor's own edit, not a control. (Until 2026-11-24 this read
"CONFIRMED ×4" resting on "two 500-problem finals with matched controls"; ISSUES #22.)

### What the grid found beyond the NULLs

1. **A shared first reading, in each editor's own words:** "format shell" (1; withdrawn
   2026-11-24, finding 1), "pruning inferred, knowledge did not survive" (2), "fluent shell,
   empty core" (3), non-viable (4), "half-knows: memorized facts at top-64 ranks, computed
   results absent" (5). These are the editors' diagnoses of the wound, not findings, and
   (2) came from file names outside its sandbox. Five single sessions, one after another:
   "five isolated contexts, one conclusion" (the first version of this item) overstates them.
2. **Three editors pointed at the late layers: wound-only observations with no full-model
   reference; not evidence of pruning damage.** Attempt 2: near-dead RoPE-region query rows
   in L23/29. Attempt 3: L29's post-FFN norm 4× neighbor norms, 98.6% static, bypass unmutes
   recall. Attempt 5 (logit-lens): correct answers surface at L25 and are destroyed by
   L26–29 (`layer_scalar` makes late blocks dominate the final state ~13:1). The layer-29
   norm weight, the per-layer residual scale (13:1) and the final-norm gain are tensors
   `prune.py` copies unchanged from the full model, and none was measured on the full model
   for comparison: they describe the untouched model.
3. **Termination is the part weight edits reach, everywhere; the cap-hit reductions are
   real.** On 64-problem practice passes: attempt 1, pass 2 (embedding boosts) cut cap-hits
   from 63 of 64 to 27 of 64; attempt 5, pass 8 (stop-token rows ×1.6 alone) from 63 to 23;
   pass 10 (final norm ×0.4 and stop-token rows together) from 63 to 3, mean length 1,172
   tokens. In the finals: attempt 5's cap-hit is 79.4% (397 of 500) against the wound's
   98.0% (490 of 500). Attempt 5's report also says its final-norm ×0.4 "cured softcap
   saturation (logits pinned at the 30.0 tanh cap) while preserving accuracy"; its own
   journal says "no logit saturation". That line, too, is a wound-only observation with no
   full-model reference. (Until 2026-11-24 this item gave the attempt-5 figures as
   "cap-hit 98→36" and repeated the softcap line as a finding.)
4. **Pro > flash for long agentic sessions, cross-family:** MiMo-flash becalmed twice
   (phase-9's flash-tier bottleneck replicated); MiMo-pro completed all 10 passes and
   produced the grid's deepest analysis. GLM-5.3-Flash held the session fine — the
   bottleneck is family×format, not tier alone.

## Selection restored, same eight experts (2026-11-24)

**Switching selection back on brings back no accuracy: 0 of 64 practice problems right, the
same as the wound as built.** The repair thesis of phase 11 stands unchanged (verdict NULL).

**Question.** The wound keeps 8 of each layer's 128 experts and still activates 8 per token,
so every kept expert fires for every token: experts were removed and selection was switched
off. Does switching selection back on, with the same eight experts, restore any accuracy? The
audit proposed this as the cheapest way to separate the two injuries.

**Setup.** A copy of the wounded model with `top_k_experts = 1` (each token picks its one
best kept expert), weights unchanged; one 64-problem practice pass (cap 8,192 tokens), and,
in the same session, a rerun of the wound as built as the control. Files:
`phase11/a1_wound_topk1_practice.json` (+ `_texts.json`) and
`phase11/a1_wound_topk8_practice_rerun.json` (+ `_texts.json`).

**Result** (all scored with the final checker):

- Selection restored: **0 of 64 right**, 59 of 64 cut at the cap (92.2%), mean 7,630 tokens.
- Control, the wound as built: **0 of 64 right**, 64 of 64 cut at the cap (100%), mean 8,192
  tokens.
- The original practice pass on the wound as built (2026-11-16): 1 of 64, 63 of 64 cut
  (98.4%).
- The full model on the same 64 problems: 47 of 64 (73.4%), 1 of 64 cut (1.6%).

**Reading.** The comparison is 0 of 64 either way, and with 64 problems the pass cannot tell
apart differences of fewer than about three right answers. Switching selection back on does
shorten generation a little: 5 of 64 generations end before the cap instead of none.

**What it does and does not show.** With only eight experts left, removing the experts is by
itself enough to destroy performance; whether the switched-off selection costs anything on
top cannot be seen at this floor.

## What this does to Phase 12 (cross-architecture atlas)

The atlas question sharpens: **same symptom, different cause, different repairability.**
Qwen 3.5-0.8B (phases 3–10): knowledge present, termination broken → single-row edits
recover real accuracy (stock 8.4 → 36.0 for the phase-10 edit; 6.0 → 27.4 as first
reported). Gemma wound: most capability lost and the rest not expressed in free-running
generation (finding 1),
termination broken → the same edit class recovers the termination mechanism on a practice
pass (cap-hit 63 of 64 to 27 of 64) but no statistically demonstrable accuracy. Both
models' exit mechanisms responded to embedding-row boosts, which is the hook phase 12 used
for its termination test (C2).

**Phase 12's verdict was withdrawn on 2026-11-24 (ISSUES #21):** none of its three tests
establishes correspondence between the two models (`phase12/results.md`), and phase 13 as
registered is not armed. The contrast above is measured; it is a starting point for an
atlas, not an atlas entry. (The first version of this section called that rhyme "the first
atlas entry".)
