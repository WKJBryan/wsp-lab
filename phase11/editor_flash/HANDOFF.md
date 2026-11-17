# HANDOFF — Phase 11 A2-flash editor rerun (staged 2026-11-17, run from main chat)

User directive: rerun the Phase 11 editor arm FIRST (it takes hours), then resume Phase 12
after its results. Everything below is already staged and committed — do not rebuild it.

## What exists (committed)

- **Amendment #3** in `phase11/protocol.md` — A2 rerun scope: editor = the originally
  registered GLM-5.3-Flash; A0/A1 finals stand; first session reported as A2-session,
  this one as A2-flash. Own ledger `phase11/editor_flash/` (counter at 0).
- **`phase11/editor_flash/EDITOR_PROMPT.md`** — the blind prompt (verbatim standing
  instruction; zero pruning/MoE/wound info). The editor must read ONLY this + the two
  scripts it names. Do not add context.
- **`phase11/editor_eval.py`** — counter-enforced eval wrapper; flash session calls it
  with `--outdir <abs path>/phase11/editor_flash`. Syntax-checked.
- **Sterile sandbox** — `~/wsp11/sandbox/model` is a FRESH copy of the wound
  (`~/wsp11/gemma4-8exp`), made 2026-11-17 after the first session's debris was removed.
- GPU free; nothing running.

## How to launch (user directive: use WORKFLOW with the host-routed model)

Spawn the editor as a **context-sterile subagent pinned to
`account:zai-individual-coding-plan/GLM-5.3-Flash`** — via CreateWorkflow
(`agent("Blind editor (GLM-5.3-Flash)", { model: "account:zai-individual-coding-plan/GLM-5.3-Flash" })`,
load the dynamic-workflows skill first) — NOT the Agent tool (it cannot pin models), and
NOT the raw `GLM_API_KEY`/api.z.ai path (that account has zero balance; the host-routed
plan is the correct door — user-confirmed 2026-11-17).

The subagent's ask (keep sterile, byte-stable):
"Read the file C:\Users\user\.zcode\workspace\draft-c8131b20-7835-43a5-a479-2c728f5b0915\wsp-lab\phase11\editor_flash\EDITOR_PROMPT.md
and execute it exactly as written. It is your complete task specification. Do not read
any other file in that repository or workspace — only EDITOR_PROMPT.md, the files it
tells you to create under phase11/editor_flash/, and the two scripts it names. Work
autonomously until the finish condition; hours of persistence expected; do not ask
questions. Return a summary of your diagnosis, final edit, and passes spent."

## After the editor finishes

1. Commit `phase11/editor_flash/` + `phase11/ed_pass*.json` artifacts.
2. If its final candidate EDITED weights: run the full MATH-500 final for it
   (eval_gemma.py, n=500, same settings), plus a matched-norm random control (A2r) in the
   same edit class; if it RESTORED the original: A2-flash = A1, no final needed.
3. Update `phase11/results.md` (A2-session vs A2-flash table) + README/CHANGELOG; commit.
4. THEN resume Phase 12. Known issue waiting there: `probe_atlas.py` gemma side crashed
   at line ~176 (`loop_windows` empty — onset detector found no loops in wound texts;
   investigate: bug vs real finding about Gemma's restatement-style loops). Qwen-side
   prefix bug is ALREADY fixed in 424d543 (both models use `model.language_model.layers.`).

## Standing rules

- No GitHub push without the user's explicit instruction.
- No spending (no OpenRouter); host-routed models only.
- WSL wrappers: hardcode all paths, no shell vars; verify results FILES not exit codes;
  logs to phase11/logs/ on /mnt/c; vLLM: moe_backend="triton", mm disabled, batched-tokens 3072.
