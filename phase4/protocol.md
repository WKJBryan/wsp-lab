# Pre-registered protocol — Phase 4: external editor models, blind, on the working stack

Committed BEFORE the editor session runs.

## Research question

The original's hinted editor (gpt-6-astra, 50 passes) repaired this model's loop pathology
(6.0 → 39.4 locally in phase 3c). Our agent-editor achieved blind DISCOVERY but not repair
(phase 3a NULL, on the broken stack). Can a **different, smaller external editor model** —
blind, no hints, on the WORKING vLLM stack — diagnose and repair the pathology in 10 passes?

Editor arms (user-directed 2026-11-12: use the plan-configured models, "spawn subagent to be
able to use it" — implemented as a dynamic-workflow subagent pinned to each host model):
- **`account:zai-individual-coding-plan/GLM-5.3-Flash`** (user's individual coding plan)
- **`xiaomi-mimo/mimo-v2.6-flash`** (user's Xiaomi MiMo plan; `-pro` as optional third arm)

## Editor interface (subagent with a real tool sandbox, per user instruction)

- The editor IS a subagent (dynamic workflow, `subagent_model` = the arm's model) with
  Bash/Read/Write tools. Its sandbox: `~/wsp4/sandbox/` in WSL, containing `model/`
  (a copy of the stock Qwen3.5-0.8B safetensors + tokenizer), `practice.json`,
  `eval_pass.py`, `ledger.md`. It is told to work ONLY inside the sandbox (soft boundary;
  the transcript is audited afterward for out-of-sandbox reads — disclosed limitation).
- Python: `$HOME/wsp3c-venv/bin/python` (torch + safetensors available).
- **Budget is enforced in software**: `eval_pass.py` keeps `eval_counter.txt`, refuses
  beyond 10 passes, and requires `--hypothesis`, appending it to `ledger.md` BEFORE running
  (pre-logging without an honor system). The editor cannot run the practice set any other
  way — GPU contention is avoided by instructing serial evaluation.
- Blind wording given to the editor (verbatim, no failure-mode information):

  > "You can run Python on a local language model's weight files (safetensors) in `model/`.
  > TASK: improve this model's MATH-500 accuracy. You may edit the weights directly.
  > No gradient training. You have 10 evaluation passes (64 practice problems each).
  > Reply with one Python block per message, or HYPOTHESIS + RUN EVAL to evaluate."

## Practice evaluation (harness-side)

64 fixed problems (same GSM8K-train ids as phase 3a's set, seed 31337, first 64), thinking on,
temp 0.6 / top-p 0.95 / top-k 20 / presence penalty 1.5 (vLLM native), cap 8192, 1 sample.
Returned to the editor: accuracy, answered%, cap-hit%, mean tokens, and 3 sample tails
(truncated to 300 chars). Final answers are NOT given for practice problems.

## Final endpoint (pre-registered)

Identical to phase 3c (same engine, decoding, cap 32,768, 500 MATH-500 problems, 1 sample,
`phase3c/rescore.py` extraction). Baseline = **stock 6.0%** carried over from phase 3c
(same protocol, same scorer — no rerun; a rerun would only add noise).

- **SUCCESS:** editor's final model ≥ 16.0% (stock + 10.0)
- **PARTIAL:** 11.0–16.0%
- **NULL:** < 11.0% (or no edit produced)

Control: C2 random-edit matched-Frobenius on the same tensors the editor touched
(1 trial, final protocol), if an edit exists.

## Integrity and safety notes

- Blindness: no hints anywhere in the loop; any loop-targeting edit must cite the editor's
  own observed practice evidence (checked at session review). Full transcript committed.
- The harness (this agent) executes editor-produced code locally under timeout; it is the
  user's chosen model and machine, disclosed here. The editor cannot run evaluations
  itself — only via the budgeted harness command.
- Editor cost: glm-4.5-flash is free tier; glm-5.3-flash would bill the user's account
  (their explicit choice to enable by recharging).
