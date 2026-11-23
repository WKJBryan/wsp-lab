// Phase 9, c8/c9: FORMAT-CORRECTED mimo team deliberation (blind weight surgery).
// Recipe from phases 4b/4c (mimo's proven format): SHORT asks, ONE instruction per ask,
// compact evidence digests, hard output caps, executor split into micro-steps.
// All agents on the run-level subagent_model (mimo-v2.6-flash for c8; mimo-v2.6-pro for c9).

const RULES = [
  "Sandbox (WSL): ~/wsp4/sandbox/ holds `model/` (a ~0.9B language model, safetensors + tokenizer),",
  "`practice.json` (64 practice math problems), `eval_pass.py` (the ONLY evaluator).",
  "Shell: Windows Git Bash; Linux commands via `wsl -e bash -c '...'`.",
  "Python (torch/safetensors): $HOME/wsp3c-venv/bin/python, run inside WSL.",
  "Task: improve this model's MATH-500 accuracy by editing weights directly. No gradient training.",
  "Team budget: 10 evaluation passes TOTAL (script-enforced). One evaluation at a time.",
  "Work only inside ~/wsp4/sandbox. Keep replies SHORT — a few lines, plain text.",
].join("\n");

phase("Observe: baseline pass");
const obs = await agent("observer").ask(
  RULES +
  "\nROLE: OBSERVER. Run exactly ONE baseline evaluation pass on the unmodified model:\n" +
  "cd ~/wsp4/sandbox && $HOME/wsp3c-venv/bin/python eval_pass.py model practice.json " +
  '--hypothesis "baseline"\n' +
  "Reply with ONLY the JSON output line, nothing else.",
);

phase("Analyze: two short independent proposals");
const [pA, pB] = await Promise.all([
  agent("analyst-a").ask(
    RULES +
    "\nROLE: ANALYST A. Baseline JSON:\n" + obs +
    "\nYou may run FREE offline probes (read tensors, forward passes — read-only, never " +
    "eval_pass.py, never modify files). Then reply in AT MOST 10 lines: diagnosis (2 lines), " +
    "ONE edit proposal — tensor, row(s), arithmetic, dose (4 lines), biggest risk (1 line).",
  ),
  agent("analyst-b").ask(
    RULES +
    "\nROLE: ANALYST B. Baseline JSON:\n" + obs +
    "\nWork independently. You may run FREE offline probes (read-only, never eval_pass.py, " +
    "never modify files). Then reply in AT MOST 10 lines: diagnosis (2 lines), ONE edit " +
    "proposal — tensor, row(s), arithmetic, dose (4 lines), biggest risk (1 line).",
  ),
]);

phase("Red-team the proposals");
const critique = await agent("red-team").ask(
  RULES +
  "\nROLE: RED-TEAM. Two proposals:\n\nA: " + pA + "\n\nB: " + pB +
  "\nAttack both (free read-only probes allowed). Reply AT MOST 12 lines: per proposal — " +
    "the fatal flaw or biggest risk, and a dose ceiling you would not exceed; then your ranking.",
);

phase("Judge: short final spec");
const verdict = await agent("judge").ask(
  RULES +
  "\nROLE: JUDGE. Proposals:\nA: " + pA + "\nB: " + pB + "\nCritique: " + critique +
  "\nDecide the FINAL edit. Reply AT MOST 12 lines: the exact spec — tensor, row(s), " +
    "arithmetic, dose, backup step (copy first!), and an allowed dose bracket (±25% max).",
);

phase("Execute step 1: apply exactly");
const e1 = await agent("executor").ask(
  RULES +
  "\nROLE: EXECUTOR, STEP 1 OF 3. Apply this spec EXACTLY (backup first as it says; work in a " +
  "NEW model dir copy, never overwrite model/):\n" + verdict +
  "\nVerify your edit byte-level (changed rows, scales). Reply AT MOST 8 lines: what is on " +
    "disk and the verification output. Do NOT run any evaluation yet.",
);

phase("Execute step 2: evaluate once");
const e2 = await agent("executor-2").ask(
  RULES +
  "\nROLE: EXECUTOR, STEP 2 OF 3. The edited model dir from the previous step is described " +
  "here:\n" + e1 +
  "\nRun exactly ONE evaluation pass on that edited dir (eval_pass.py <dir> practice.json " +
  '--hypothesis "first test of judged edit"). Reply ONLY the JSON line plus one line: better or worse than baseline 12.5.',
);

phase("Execute step 3: bracket or finalize");
const e3 = await agent("executor-3").ask(
  RULES +
  "\nROLE: EXECUTOR, STEP 3 OF 3. Previous result:\n" + e2 +
  "\nJudge's bracket: " + verdict +
  "\nIf one bracket pass within the allowed range is clearly justified, run it; otherwise do " +
    "not. Then reply AT MOST 8 lines ending with exactly: FINAL: <path of the best model dir>.",
);

log("mimo short-format team finished");
return { observer: obs, proposalA: pA, proposalB: pB, critique, verdict, step1: e1, step2: e2, step3: e3 };