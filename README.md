# wsp-lab

A complete replication, stress-test, and extension of **weight-space programming** — the claim, from the MIT CSAIL post [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming) (Nov 2026), that an agentic LLM given a neural network's raw weights can diagnose and repair it through direct weight edits alone: no gradient updates, no fine-tuning, just arithmetic on parameter tensors.

**Everything here ran locally on one RTX 5090. No paid APIs. Every number is reproducible from this repo.**

The original used one editor (OpenAI gpt-6-astra, hinted, 50 evaluation passes) and released [full logs](https://github.com/RikiyaT/lm_understand-logs) — but no cross-editor replication existed, no random-edit control had been run, and (as we verified from their logs) the headline repair was **hinted** in the initial prompt, with the no-hint condition never tested at full budget. This repo closes all of those gaps.

## TL;DR — the final scoreboard

Target: Qwen3.5-0.8B, whose "thinking" degenerates into infinite `Wait, no.` loops (stock scores **6.0%** on MATH-500; 91% of generations never terminate). All arms: same 500 problems, same engine (vLLM), same scorer, blind editors unless noted.

| Surgeon | Hint? | Passes | MATH-500 | What they did |
|---|---|---|---|---|
| Stock model | — | — | **6.0%** | loops forever |
| Random noise (matched edit size) | — | — | **2.2%** | control: proves edits aren't luck |
| GLM-5.3 (this agent, phase 3a)* | no | 3 of 5 | 8.4% | right diagnosis, failed cure |
| GLM-5.3-Flash (phase 4) | no | 6 of 10 | **25.2%** | 3-row rescale (`</think>` ×1.55, comma ×0.5, im_end ×1.15) |
| **mimo-v2.6-flash / pro** (4b/4c) | no | 7 of 10 | **27.8%** | single row ×1.5 — best blind result |
| GLM-5.3-Flash (phase 10, full budget) | no | **6 of 50** | **27.4%** | same convergent edit; budget unused |
| Best committee (of 7 formats, phase 9) | no | 2–6 | 24.4% | never beat its loner |
| **Original's editor** (published edit) | **yes** | 50 | **39.4%** (our stack) / 48.1% (theirs) | 2-tensor down_proj direction surgery |

*\*phase 3a initially looked worse (+1.6) — our own harness bug (prohibited greedy decoding), caught, corrected, and documented: ISSUES #8–9.*

And at small scale (phase 1): a blind agent-editor repaired an organically broken CartPole policy **9.4 → 371.6** steps (of 500) by rewriting 258 of 17,410 weights; random edits of identical size scored 9.5.

## The experiments

| Phase | Question | Result |
|---|---|---|
| [1](phase1/results.md) | Can an agent repair a small broken network by weight edits alone? | ✅ **SUCCESS** — 9.4 → 371.6; random control 9.5; retrain ceiling 500 |
| [2](phase2/results.md) | Can it *read* weights blind (5 mystery nets)? | ➗ which inputs matter: **2/2 exact**; task ID 3/5 (fooled by memorized noise) |
| [3](phase3/results.md) | Blind no-hint surgery on the real model (873M params) | ❌ NULL — found the loops blind; cure failed (dose: no-op → seizure). Contains the correction notice for our greedy-decoding error |
| [3b](phase3b/results.md) | Corrected decoding, take 2 | VOID — model non-terminates under *every* decoding our first engine could produce; engine forensics |
| [3c](phase3c/results.md) | Fair re-run on vLLM/WSL2 | ✅ the original's edit **TRANSFERS locally: 6.0 → 39.4** (terminations 9% → 83%) |
| [4, 4b, 4c](phase4/results.md) | Can *other, smaller* editor models do it blind? | ✅ **three editors, two families, two tiers — all succeed, all converge on the same single row** (248069, the `</think` embedding) at ×1.5–1.55 |
| [5](phase5/results.md) | Can the original's edit be pushed further? (dose-response) | NULL — higher doses win practice, lose finals (1.0× 38.2 vs 1.75× 35.4): the loop direction also carries real reasoning |
| [7](phase7/ATLAS.md) | *Where* does the loop live? (activation atlas) | the loop direction is written **diffusely** by every layer's MLP — no single hotspot; the original's 64 chosen columns are not weight-aligned (activation-chosen) |
| [8](phase8/protocol.md) | The missing baselines (thinking-off stock, decode-time bias control…) | pre-registered — see protocol |
| [9](phase9/results.md) | Do *teams* (look/discuss/red-team/judge/execute) beat a lone editor? | ❌ **0 of 7 committees beat their loner**, in any format, family, or mixture — committees systematically under-dose |
| [10](phase10/results.md) | The field's first **no-hint full-budget** run (50 passes) | 27.4% using only **6 passes** — budget is not the constraint; knowledge is |
| [11](phase11/results.md) | **Prune × WSP**: gradient-free repair of an expert-pruned MoE (Gemma-4 26B-A4B, 128→8 experts) | ❌ NULL repair (58.4% → 1.0%; editor restored original) — but the damage **decomposition is measured**: termination repairable by one edit class (cap-hit 98→42%), deleted knowledge not (0/21 arithmetic); provenance undetectable blind |
| [12](phase12/results.md) | **Cross-architecture atlas** (user-designed): do Qwen-0.8B and Gemma-MoE write the same functions with matching number signatures? | ✅ **CORRESPONDENCE 2/3** — loop-onset dynamics **r = 0.96** (p<0.001); termination rows both low-norm + boost-responsive; writer concentration focal in both but distributions differ |

(Phases 5–8 are the work of a second agent session sharing this repo; phases 1–4, 9, 10 by the primary. Same harness, same rules, one timeline in git.)

## What we learned — results and thoughts

1. **The core claim is real and robust.** Five different editor models — two families, three tiers, and a committee of four — independently diagnosed the same failure and repaired it with no hints, no training, on a gaming GPU, for free. When independent surgeons converge on one knob out of 873 million (10 of 10 blind sessions found row 248069), that's the signature of a real, fundamental target.
2. **But it's repair, not enhancement.** The edits unstick a broken "stop thinking" switch; they don't add intelligence. A healthy stock model scores ~45–50%; the best blind repair reaches 27.8%. Nobody — including the original — has shown an ability *gain* from weight surgery here.
3. **The hint is worth ~12 points — and now we know what it buys.** Blind editors converge fast (6 passes suffice; more budget adds ~nothing) and *rationally stop at the surface*: rescaling the output row. The hinted original went deeper — subtracting a measured direction from 64 columns inside layers 15/19 — reaching 39–48%. The hint's value isn't finding the disease (everyone does); it's **certifying that a riskier, deeper surgical route exists.**
4. **Committees don't beat loners at this task — they under-dose.** Seven deliberation formats (same-family, cross-family, interleaved) produced the phase's most sophisticated *methods* (untied heads, escalation gates, a depth-gated additive mechanism) and its most cautious *doses*. Every committee-shipped edit was gentler than its loner's, and gentle doses die on hard problems. Red-teams were right about every risk they measured — and wrong about which risk mattered.
5. **Session format beats model tier.** mimo tiers could not hold a whole multi-role session (empty reports, leaked reasoning) yet produced the best red-team work of the project when given one focused seat. The harness around a model matters as much as the model.
6. **The loudest scientific killer was measurement, not models.** Our biggest wrong conclusions came from our own rig: a decoding mode the model card forbids, a scorer that silently failed on Windows, an `ast.parse` check that doesn't catch repeated keywords. (The last was caught by the *editor models*, twice.) When results look wrong, suspect the ruler first — the user's instinct on this was the project's most valuable instrument.
7. **Symptoms are self-hinting; silence is the frontier.** Blind discovery worked because loops shout in the outputs. The original's own weights-only sycophancy hunt failed 4× — because flattery leaves no visible trace. Understanding is only needed where watching isn't enough.

## Integrity practices

- **Pre-registration everywhere:** every phase's thresholds, budgets, and controls committed *before* the run (git history proves ordering); deviations logged in [ISSUES.md](ISSUES.md) with reasons.
- **Software-enforced budgets:** editors cannot exceed their evaluation passes or skip hypothesis logging — the counter file refuses.
- **Controls the original skipped:** matched-norm random edits (2.2% — it's the edit's *content*, not its size), untouched baselines, retrain ceilings.
- **Full audit trail:** every editor session's ledger, probe scripts, and generations are committed (generation texts gzipped for re-scoring without a GPU).
- **Failures stay in the record:** wrong turns, void runs, and our own bugs are documented, not buried — see [CHANGELOG.md](CHANGELOG.md) and [ISSUES.md](ISSUES.md) (#4, #8–#16).

## Reproduce

```bash
uv sync                  # torch cu128 (Blackwell) + gymnasium, locked
uv run python phase1/train.py 20      # small-network repair target
uv run python phase1/evaluate.py phase1/ckpt/seed0.pt 64
```

Phases 3+ additionally need WSL2 + vLLM + CUDA toolkit (see `phase3c/results.md` for the full recipe: vLLM 0.33, CUDA 12.8 toolkit, ninja; models ~1.6 GB each). Re-score any committed run without a GPU:

```bash
gunzip phase3c/p4_glmflash_texts.json.gz && uv run python phase3c/rescore.py p4_glmflash
```

## Repo map

`PROTOCOL.md` + `phaseN/protocol.md` — pre-registered rules · `phaseN/results.md` — verdicts with evidence · `phaseN/session_*/` — editor ledgers, probes, logs · `phase3c/` — the fair-evaluation harness (engine, scorer, rescorer) · [CHANGELOG.md](CHANGELOG.md) · [ISSUES.md](ISSUES.md) (16 issues, all closed) · MIT license.

## Context

- Original post: <https://rikiyat.github.io/blog/weight-space-programming>
- Original logs (CC BY 4.0): <https://github.com/RikiyaT/lm_understand-logs>
- Edited model from the original: <https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming>
- Independent replications of the original: openwsp 50.6%, Khaliq 51.0%, Noble 61.7%
