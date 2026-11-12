# wsp-lab

A replication and stress-test of **weight-space programming** — the claim, from the MIT CSAIL post [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming) (Nov 2026), that an agentic LLM given a neural network's raw weights can diagnose and repair it through direct weight edits alone: no gradient updates, no fine-tuning, just arithmetic on the parameter tensors.

The original used one editor (OpenAI gpt-6-astra via Codex CLI) and released [full logs](https://github.com/RikiyaT/lm_understand-logs) — but no cross-editor replication exists, no random-edit control was run, and (per the logs) the headline Qwen repair was **hinted** in the initial prompt, with the no-hint condition never tested at full budget.

## What this repo does

1. **Different editor:** the weight editor here is an agentic LLM (ZCode / GLM-5.3) with a code sandbox — a cross-editor data point nobody has published.
2. **Organic failures only:** the broken networks are genuine training local optima, never artificially corrupted (the editor also wrote the trainer, so sabotage would invalidate the test — see ISSUES.md #2).
3. **Pre-registered protocol:** success thresholds, budgets, and controls are locked in [PROTOCOL.md](PROTOCOL.md) *before* the editor sees any broken network. Git history proves the ordering.
4. **Missing controls included:** matched-norm random-edit control, untouched baseline, retrain upper bound.
5. **All local, all free:** runs on one RTX 5090; no paid APIs.

## Phases

- [x] Phase 0 — repo, uv environment (torch cu128 for Blackwell), pre-registered protocol
- [x] Phase 1 — CartPole repair: **9.4 → 371.6** (SUCCESS per protocol; random-edit control 9.5; results: [phase1/results.md](phase1/results.md))
- [x] Phase 2 — blind weights-only ID: feature-ID **2/2 exact PASS**; task assignment **3/5 FAIL** (parity↔random swap, diagnosed; results: [phase2/results.md](phase2/results.md))
- [x] Phase 3 — blind no-hint Qwen3.5-0.8B on one RTX 5090: editor **NULL** (+1.6, noise band; loops found blind, repair failed: no-op→seizure dose-response); original's published edit cross-check **6.8% → 12.4%** (+5.6, PARTIAL — far below their 48.1% headline under their settings; results: [phase3/results.md](phase3/results.md))
- [x] Phase 3b — corrected decoding (user-initiated): **VOID locally** — the model non-terminates under every decoding our transformers-on-Windows stack can produce (8/8 pilot runs loop to 32,768 with repetition tails, penalty on and off); scorer replaced after math-verify broke on this platform; the original's edit transfers per concordant third-party vLLM replications (50.6/51.0/61.7%) — see [phase3b/results.md](phase3b/results.md) and ISSUES #8–9

Results land in `phase1/results.md`, `phase2/results.md`, ... as each phase completes.

## Reproduce

```bash
uv sync                  # creates .venv with torch cu128 + gymnasium (locked by uv.lock)
uv run python phase1/train.py 20      # trains 20 seed policies, writes checkpoints + summary
uv run python phase1/evaluate.py phase1/ckpt/seed0.pt 64   # greedy eval, held-out seeds
```

Hardware: anything torch-cu128 runs on; the phase-1 nets are 17K-parameter MLPs (CPU is fine, the 5090 is idle until phase 3).

## Repo conventions

- [CHANGELOG.md](CHANGELOG.md) — notable changes, bugs, fixes per phase
- [ISSUES.md](ISSUES.md) — open/closed issue log (migrates to GitHub Issues on publish)
- `phaseN/editor/ledger.md` — the editor's action log: every budgeted evaluation, dated
- Conventional commits (`feat:`, `fix:`, `docs:`, `chore:`)

## Context

- Original post: <https://rikiyat.github.io/blog/weight-space-programming>
- Original logs (CC BY 4.0): <https://github.com/RikiyaT/lm_understand-logs>
- Edited model from the original: <https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming>
