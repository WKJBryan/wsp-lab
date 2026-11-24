# Roadmap — portable weight-space medicine

The north-star goal (set 2026-11-17, user-directed): determine whether weight-space
diagnoses are **portable across architectures** — whether the functional map of one
model (which numbers do what) numerically corresponds to another's, so that a
diagnosis proven on model A can be applied to model B without search.

Three falsifiable rungs. Every outcome — including negatives — is a result.

Standing after the corrections of 2026-11-24: phase 11's numbers are re-scored and its
evidence restated (ISSUES #22); phase 12's verdict is withdrawn (ISSUES #21); phase 13
is not armed.

## Phase 11 — wound repair (COMPLETE 2026-11-17, verdict NULL)

Blind gradient-free repair of the expert-pruned Gemma-4 wound. Restated 2026-11-24
(ISSUES #22).
- Final table (full MATH-500): A0 full 76.4% / 0.2% cap-hit · A1 wound (8 experts
  kept, expert selection disabled) 1.2% / 98.0% · A2 (attempt 1) = A1 — no repair
  found; the editor spent 5/10 passes and restored the original (Amendment #2:
  context-sterile subagent; API balance exhausted).
- The wound is two injuries at once: `prune.py` keeps 8 experts per layer and leaves
  the active-expert count at 8, so expert selection is switched off as well as experts
  removed. It uses the same 4.39 billion parameters per token as the full model and
  saves memory (51.6 to 8.8 GB), not computation. No final separates the two; one
  practice pass with selection restored is recorded below.
- Attempts 2–5 (2026-11-17/18) found no repair either: the best final is attempt 5's
  (final norm × 0.4) at 2.0% (10 of 500) against the wound's 1.2% (6 of 500) — not
  distinguishable. No matched control was completed.
- The payload is the restated damage split. Termination is moved by embedding-row
  boosts: on the final, attempt 5's edit cut cap-hits from 98.0% to 79.4%; on a
  64-problem practice pass the same edit class cut them from 98.4% to 42.2%, with
  accuracy 1 of 64 before and 2 of 64 after (1.6% and 3.1% under the final checker) —
  a difference 64 problems cannot tell apart. Most capability is lost, and some
  remains present but not expressed in free-running generation. One editor learned
  of the pruning from file names, so "provenance undetectable blind" is not
  established.
- Selection restored, same eight experts (2026-11-24): a copy of the wound with
  `top_k_experts = 1` (each token picks its one best kept expert; weights unchanged)
  ran one 64-problem practice pass (cap 8,192 tokens), and in the same session the
  wound as built was rerun as the control. Selection restored: 0 of 64 right, 59 of 64
  cut at the cap (92.2%), mean 7,630 tokens. Control, the wound as built: 0 of 64
  right, 64 of 64 cut (100%), mean 8,192 tokens; its original practice pass of
  2026-11-16 scored 1 of 64, with 63 of 64 cut (98.4%). The full model on the same 64
  problems: 47 of 64 (73.4%), 1 of 64 cut (1.6%). Switching selection back on brings
  back no accuracy on this set — 64 problems cannot tell apart differences of fewer
  than about three right answers — and shortens generation a little (5 of 64
  generations end before the cap instead of none). With only eight experts left,
  removing the experts is by itself enough to destroy performance; whether the
  switched-off selection costs anything on top cannot be seen at this floor. The
  repair thesis of phase 11 stands unchanged (verdict NULL).
- Implication for Phase 12: Qwen (knowledge present + termination broken → real
  accuracy recovery 8.4→36.2) vs Gemma wound (most capability lost → mechanism-only
  recovery so far) — same symptom, different cause, different repairability. Both exit
  mechanisms answered to embedding-row boosts, but that was already on record from
  phases 4–5 and 11 and is not evidence of correspondence (ISSUES #21).

## Phase 12 — cross-architecture atlas (run 2026-11-17; its "COMPLETE, verdict CORRESPONDENCE 2/3" result is WITHDRAWN — NOT ESTABLISHED, ISSUES #21)

Map the same functions in both models and test numerical correspondence.
- Qwen3.5-0.8B side: consolidate phase-7 atlas (loop direction → MLP columns) and
  phases-4/10 landmarks (termination row 248069 family).
- Gemma-4 MoE side: same probes on full + wound; expert impact table already
  measured (`phase11/calibration_impact.json`, 30×128).
- Correspondence tests (pre-registered):
  - C1: loop-direction similarity across models (cosine / CKA on shared probe set),
    significance by permutation test (≥95th percentile of shuffled null).
  - C2: termination-circuit signature match — statistical fingerprint (row norm,
    activation statistics, positional behavior) of Qwen's `</think` row vs the
    best-matching Gemma row(s); report rank of the true analog among all rows.
  - C3: cluster-structure comparison — do the 8 kept experts' impact profiles
    cluster like Qwen's MLP column clusters (adjusted Rand index vs matched-size
    null).
- Registered rule, as written: CORRESPONDENCE if ≥2 of C1–C3 pass their bars;
  NO-CORRESPONDENCE otherwise (both publishable). The run of 2026-11-17 printed
  "CORRESPONDENCE, 2 of 3"; that verdict is withdrawn (see below).
- **Verdict after correction (2026-11-24): NOT ESTABLISHED** (ISSUES #21). The
  registered rule, applied literally, prints "2 of 3", but both passes are
  uninformative (`phase12/results.md`):
  - C1 (r = 0.96): the shape is fixed by construction — each trajectory is z-scored
    by its own loop window. Two generic step curves correlate at 0.96 or more 7.8% of
    the time; after each curve's own step is removed the residuals correlate at −0.13;
    the "1,000 circular shifts" test has only 24 distinct shifts; the two text sets
    are not comparable (23 Qwen texts against 10 Gemma texts).
  - C2: the behavioral part passes as registered (Qwen cap-hit 90.6% → 40.6% with
    accuracy up 27.8 points; Gemma 98.4% → 42.2% with accuracy indistinguishable,
    1 of 64 before and 2 of 64 after), but a row boost cutting cap-hits in both
    models was already on record from phases 4–5 and 11. The statistical part passes
    by chance (70% of the time with three Gemma candidate rows), and low norm is a
    description of the exit row, not a way to find it.
  - C3: fails as run (KS D = 0.55, p = 0.0003), and its stated direction is wrong —
    Gemma is less concentrated, not more.
- Integrity (ISSUES #21): the three tests registered above were replaced 8.4 hours
  before the run without a note; the first probe's held-out AUC of 0.68 was below the
  0.75 validity bar, after which the text selection and onset detector were changed
  (final 0.826); the calculator had read files later editor attempts overwrote, and
  the wrong Qwen arm. It now reads pinned copies under `phase11/editor_a2_attempt1/`.
- Consequence: Phase 13 is not armed.

## Phase 13 — the transfer test (not armed; to be redesigned)

Apply Qwen's proven diagnosis to the Gemma wound with ZERO search (and vice versa
where an edit class exists).
- Edit classes eligible: termination-row scaling; loop-direction subtraction from
  analogous columns; router-row rescale (MoE-only).
- Verdict (pre-registered): TRANSFER-POSITIVE if the transferred edit beats the
  matched-norm random control by ≥ +2.5 pts on practice MATH-500 without any
  parameter search on the target model; else TRANSFER-NEGATIVE.
- TRANSFER-POSITIVE is the headline result: portable weight-space medicine.
- **Not armed (2026-11-24):** as registered, this test presupposes that phase 12 established a correspondence between the two models' maps, and no phase-12 test did, so the section is kept as registered text only and is to be redesigned (ISSUES #21).

## Honest framing

The realistic claim spectrum: strong paper (any outcome) → field-defining first
(transfer-positive). Framing in any writeup rides on the data, nothing more.
