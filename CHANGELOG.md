# Changelog

Notable changes, bugs, and fixes. Format follows [Keep a Changelog](https://keepachangelog.com/); dates are YYYY-MM-DD.

## [0.3.0] — 2026-11-04

### Added
- Phase 2 complete (blind weights-only identification, sealed ground truth, guesses committed before reveal):
  - **P2 feature identification PASS** — both informative-feature subsets exact ({2,3} blobs, {0,1} interact).
  - **P1 assignment FAIL at 3/5** (bar ≥4): parity↔random swap; post-hoc diagnosis — memorized noise carries the "huge norms + rough function" signature I attributed to overtrained parity (random net memorized 86.5% of 4K random labels).
  - Verdicts and integrity ordering in `phase2/results.md`.

## [0.9.0] — 2026-11-13

### Added
- **Phase 4b/4c: three-editor convergence — all blind arms succeed.** mimo-v2.6-flash retry (step-driven harness): **27.8%** MATH-500 via single-row ×1.5 edit; mimo-v2.6-pro (user-requested arm): **27.8%** (mathematically identical edit, bf16-rounding-different bytes, identical score). With GLM-Flash's 25.2%, **three independent blind editors converged on the same weight row (248069, the </think> embedding) at near-identical doses** — the strongest evidence yet that blind weight-space repair is real and the target fundamental. Harness findings: session format (one-ask vs step-driven) gates mimo performance; typed structured-submit replaced by plain replies after two session deaths; baseline determinism proven byte-identical across all editors.

## [0.8.0] — 2026-11-13

### Added
- **Phase 4 complete — cross-editor blind surgery (main-agent session):** GLM-5.3-Flash (blind, 6 passes) repaired the loop pathology: **6.0% → 25.2% MATH-500** (pre-registered SUCCESS ≥16.0) via a byte-verified 3-row rescale (`</think>` ×1.55, comma ×0.5, `im_end` ×1.15); C2 matched-random control 2.2% — content, not magnitude. mimo-v2.6-flash: competent inspection (independently caught the harness bug; baseline matched GLM pass-1 exactly — determinism cross-check) but **no edit attempted** across three continuations; long-horizon agency is the cross-editor difference.
- ISSUES #11 closed in full: duplicate-keyword evaluator bug shipped twice, caught twice by editor models; author's ast.parse check was insufficient (py_compile decides — repeated kwargs pass ast.parse). Repo file compiler-verified.
- phase3b edit-size claim corrected per ISSUES #14, independently re-verified (64 cols/tensor, ≈130.2K values, rest bit-identical — the original's release matches their claim).

## [0.7.0] — 2026-11-13

### Added
- **Phase 5 complete: NULL on finals with a measured trade-off.** Practice dose curve peaked at 1.75× (50.0% GSM8K practice, stuck 53→25%) but finals (MATH-500 ×4) reversed it: dose 1.0 = 38.2% vs 1.75× = 35.4% — the loop direction also carries legitimate hard-problem reasoning; the original's dose is well-balanced. GSM8K-practice confound documented.
- Edit anatomy (free): exactly 131,072 values, rank-1 per layer, one direction at two depths (cos 0.849); ISSUES #14 corrects our earlier 56× miscount — the original's release matches their claim.
- Harness forensics #13/#15: 2,000-in-flight token soup (probe-cleared n=4/depth; concurrency suspect), engine memory races, vLLM/py3.12 int-str crash class, missing `__main__` guard crash class.
- Phase 7 atlas part 1: loop direction written diffusely (no layer hotspot); original's 64 columns NOT weight-aligned (activation-chosen); L23 strongest attention-side writer (matches openwsp's independent pick); token map suggestive.

## [0.6.0] — 2026-11-12

### Added
- **Phase 3c: the original's weight edit TRANSFERS on local hardware — 6.0% → 39.4% MATH-500** (pre-registered TRANSFERS bar +10; terminations 9% → 83%). vLLM 0.33 under WSL2, full 32K budget, both arms 500 problems in ~80 min total. Anchors: stock 6.0/90.6% cap ≈ Toledo 7.8/93.1 ✓.
- Engine diagnosis PROVEN (probe: 3/4 terminate on vLLM vs 0/8 on fallback kernels) — closes #9. Format-aware extraction closes #10 (prose answers; boxed-only scored 0.0/0.4 raw).
- WSL2 setup documented: CUDA toolkit 12.8 + ninja + build-essential required for flashinfer JIT.

## [0.5.0] — 2026-11-12

### Added
- Phase 3b (corrected-decoding cross-check): **VOID locally** — the model loops to the cap under every decoding this transformers-on-Windows stack produces (pilot: 8/8 runs non-terminating at 32,768 tokens with repetition tails, penalty on AND off). Scorer replaced (`phase3b/scorer.py`, 11/11 sanity) after math-verify proved broken on this platform. Verdict on the original's claim rests with concordant third-party vLLM replications (50.6 / 51.0 / 61.7%).
- Full forensics in `phase3b/results.md`; ISSUES #8 (greedy-decode error, user-initiated) and #9 (stack-level non-termination) documented with costs (~44 GPU-hours).

## [0.4.0] — 2026-11-05

### Added
- Phase 3 complete (blind no-hint Qwen3.5-0.8B, local RTX 5090, greedy, math250 finals):
  - **Editor NULL**: +1.6 pts (random-control band ±0); loop pathology discovered blind and cited; exit-token edit characterized to exhaustion (×1.1 no-op → ×3.0 token seizure); declared with 2 passes unspent.
  - **C3 cross-check: original's published edit transfers partially under our harness — 6.8% → 12.4% (+5.6, PARTIAL bar)**, far below their 7.9 → 48.1 headline (sampled decoding, 32K cap, hint, 50 passes).
  - New observation: 100% cap-hit under greedy decoding for every arm — thinking never terminates at 6K tokens in this regime.

## [0.3.0] — 2026-11-04

### Added
- Phase 1 complete: editor repaired an organically collapsed CartPole policy by weight-space edits alone — primary endpoint **371.6** vs baseline **9.4** (pre-registered SUCCESS ≥195 ✅; STRONG ≥475 not reached). Edit: 258/17,410 params (1.5%).
- Controls (the ones the original never ran): matched-norm random-edit best-of-20 = **9.5** (editor beats it by 362); retrain upper bound 500.0/22.9/500.0 (seed 101 organically collapsed too).
- Full editor action log with pre-registered budget accounting: `phase1/editor/ledger.md` (512/512 episodes, 8 batches, all pre-logged; 2 hypotheses refuted offline at zero cost).
- Final measurements harness: `phase1/controls.py`.

### Added
- Repo scaffold: uv-managed environment (`pyproject.toml` + `uv.lock`), git conventions, README.
- Pre-registered Phase 1 protocol (`PROTOCOL.md`) — thresholds, budgets, controls locked before any editing.
- Phase 1 harness: REINFORCE CartPole trainer (`phase1/train.py`), greedy evaluator with held-out seed ranges (`phase1/evaluate.py`), shared policy definition (`phase1/policy.py`).

### Fixed
- #1: system Python had torch 2.6.0+cpu (no CUDA, predates Blackwell sm_120 support) — the uv env pins torch ≥ 2.7 from the cu128 index instead.
- #4: `phase1/select.py` crashed torch with a hard C++ abort during import — the filename shadowed the stdlib `select` module; with `phase1/` first on `sys.path`, a library's internal `import select` re-entered our file mid-torch-init. Renamed to `pick_target.py` (identical content verified working). Rule going forward: no stdlib-colliding script names.
