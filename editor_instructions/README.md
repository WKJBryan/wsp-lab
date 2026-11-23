# editor_instructions

Verbatim copies, made 2026-11-23, of the workflow scripts run for the blind editor sessions below: an LLM editing the raw weights
of a target model (Qwen3.5-0.8B) to improve its MATH-500 accuracy. The originals are in the git-ignored `.zcode/workflow-drafts/`
folder. They are historical records first and reusable templates second. Each file is TypeScript for a workflow harness with a
small API: `phase("...")`, `agent("name", { system, model })`, `await agent.ask(text)` or `await agent.ask<Type>(text)`, `log(...)`
and `return {...}`. The editor instructions are string literals in the file: role text passed as `system`, and the task text passed
to each `ask`.

## Files

| File | Phase / condition | Editor model(s) | Session format | Results file |
|---|---|---|---|---|
| `Phase-4-editor-session-GLM-5.3-Flash-blind.dwf.ts` | 4 | GLM-5.3-Flash | one ask | `phase4/results.md` |
| `Phase-4-editor-session-mimo-v2.6-flash-blind.dwf.ts` | 4 | mimo-v2.6-flash | one ask, then two continuation asks | `phase4/results.md` |
| `Phase-4b-mimo-v2.6-flash-retry-step-driven.dwf.ts` | 4b | mimo-v2.6-flash | step-driven, 5 asks | `phase4/results.md` |
| `Phase-4c-mimo-pro-untyped.dwf.ts` | 4c | mimo-v2.6-pro | step-driven, 5 asks | `phase4/results.md` |
| `Phase-8-team-glm-flash.dwf.ts` | 9, condition 1 | GLM-5.3-Flash on all six roles | 6-role waterfall | `phase9/results.md` |
| `Phase-9-6-mixed-team.dwf.ts` | 9, condition 6 | GLM-5.3-Flash (observer, judge), GLM-5.3 (analyst B, executor), mimo-v2.6-pro (analyst A), mimo-v2.6-flash (red-team) | 6-role waterfall | `phase9/results.md` |
| `Phase-9-7-interleaved.dwf.ts` | 9, condition 7 | GLM-5.3-Flash on both seats | surgeon and critic in a loop | `phase9/results.md` |
| `Phase-9-8-mimo-team-short.dwf.ts` | 9, conditions 8 and 9 | mimo-v2.6-flash (condition 8), mimo-v2.6-pro (condition 9) | 6-role waterfall of short asks | `phase9/results.md` |
| `Phase-10-fullbudget.dwf.ts` | 10 | GLM-5.3-Flash | step-driven, 50 passes | `phase10/results.md` |

The folder holds only these nine files. Nothing here covers any other phase, or phase 9 conditions 2 to 5. `Phase-8-team-glm-flash.dwf.ts`
keeps the number its experiment was first registered under; it is phase 9 (see the provenance note in `phase9/protocol.md`).
`Phase-9-8-mimo-team-short.dwf.ts` is one script for two conditions; its header comment names the model for each. Only `Phase-9-6-mixed-team.dwf.ts`
passes a `model` to `agent()`. The other eight files pass no model id: their agents use the harness's run-level default, and the table
takes their model from the header comment, the `phase()` label or the results file.

## Rules every file repeats (wording differs slightly)

- Edits: "No gradient training, no fine-tuning, no optimizers — direct edits to weight tensors only." Six files say only "No gradient training."
- Budget: a fixed number of evaluation passes, each a run of `eval_pass.py`, "the ONLY way to evaluate the model"; the script "enforces the budget". Ten passes in eight files; fifty in `Phase-10-fullbudget.dwf.ts`. Seven files also allow free offline inspection outside `eval_pass.py`.
- One at a time: "Run evaluations strictly one at a time (single GPU)."
- Sandbox: "Work only inside ~/wsp4/sandbox."

"Blind" means the task text names no model and describes no failure. The model is "a local language model (~0.9B parameters, safetensors + tokenizer)"
and the task is only "improve this model's MATH-500 accuracy". The editor's evidence is what the evaluator returns: "accuracy, termination statistics, and sample generations".

## What to change to reuse a file

- Paths: the sandbox `~/wsp4/sandbox/` (with its `model/` and `practice.json`) and the Python environment `$HOME/wsp3c-venv/bin/python`, both in every file. The files also assume Windows Git Bash reaching Linux through `wsl -e bash -c '...'`.
- The evaluator command, `eval_pass.py model practice.json --hypothesis "..."` (`<model_dir>` in three files), and the pass budget stated in the task text.
- The editor model: the harness's run-level default, or for `Phase-9-6-mixed-team.dwf.ts` the per-seat ids `account:zai-individual-coding-plan/GLM-5.3-Flash`, `account:zai-individual-coding-plan/GLM-5.3`, `xiaomi-mimo/mimo-v2.6-pro` and `xiaomi-mimo/mimo-v2.6-flash`.
- The harness calls (`phase`, `agent`, `ask`, `log`, `return`) if a different orchestration tool is used. Three files use typed asks, `ask<EditorReport>` and `ask<StepStatus>`.
- Session-specific text: the two continuation asks in `Phase-4-editor-session-mimo-v2.6-flash-blind.dwf.ts` refer to that session's previous reply and remaining budget, and `Phase-9-8-mimo-team-short.dwf.ts` gives its executor a fixed baseline value to compare against.
