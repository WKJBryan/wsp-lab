# Issues log

Tracked here until the repo is published, then migrated to GitHub Issues. Status: open / closed / closed-by-design.

| # | Title | Status | Resolution |
|---|-------|--------|------------|
| 1 | System torch 2.6.0+cpu cannot use the RTX 5090 (Blackwell sm_120 needs ≥ 2.7, cu128 wheels) | closed | uv env pins torch from the cu128 index (`pyproject.toml`); verified via `torch.cuda.is_available()` after `uv sync` |
| 2 | Validity: the editor agent also wrote the trainer, so any "corruption" it authored would leak the answer | closed-by-design | Organic failures only — there is no corruption step in the codebase; the editor receives only the checkpoint + evaluator (PROTOCOL.md, Editor rules 1) |
| 3 | Editor could overfit edits to specific eval episodes | closed-by-design | Disjoint seed ranges: budget evals 800000+, primary endpoint 900000–900255 untouched during editing, controls 910000+ (PROTOCOL.md) |
| 4 | `phase1/select.py` hard-crashed torch (C++ abort) during import, deterministically, while identical imports worked elsewhere | closed | Root cause: filename shadowed stdlib `select` module; re-entrant import during torch init. Renamed `pick_target.py`; identical content then ran clean. Never name scripts after stdlib modules. |
| 5 | Ambiguity: does simulating cart-pole dynamics in editor code count against the evaluation budget? | closed-by-design | Yes — integrating the env's dynamics with the policy in the loop is functionally an evaluation and is forbidden to the editor. Only harness `evaluate.py` may run env episodes. Logged in the ledger before editing began. |
| 6 | Phase-3 harness misalignment found on first stock run: template defaults thinking OFF; boxed-only extraction scored 0/128 while generations contained correct prose answers | closed | Fixed identically for all arms BEFORE any comparison: `enable_thinking=True` everywhere; extraction = boxed-then-last-number. Void run kept (`phase3/runs/stock_practice.json`); protocol addendum committed. |
| 7 | Practice cap 4096 cannot separate pathological looping from long thinking (100% cap-hits at 16.4% acc); ~25 min/pass makes 10 passes impractical | closed | Addendum 2, pre-session: uniform 6144 caps, 64-problem passes, batch 32, editor budget 10→5 (strict weakening; thresholds unchanged). Applied to all arms equally. |
