// Phase 10: the no-hint FULL-BUDGET run. GLM-5.3-Flash, blind, 50 evaluation passes.
// Step-driven solo format (its proven recipe) with an extended iterate allowance.

const COMMON = [
  "You have shell access (Windows Git Bash; run Linux commands via: wsl -e bash -c '...').",
  "Sandbox (WSL, path ~/wsp4/sandbox/):",
  "- model/ — a local language model (~0.9B parameters, safetensors + tokenizer).",
  "- practice.json — 64 practice math problems.",
  "- eval_pass.py — the ONLY way to evaluate the model.",
  "Python with torch/safetensors lives at $HOME/wsp3c-venv/bin/python (run inside WSL).",
  "",
  "TASK: improve this model's MATH-500 accuracy. You may edit the weights in model/ directly.",
  "",
  "Rules:",
  "1. No gradient training, no fine-tuning, no optimizers — direct edits to weight tensors only.",
  "2. You have AT MOST 50 evaluation passes — a GENEROUS budget, so be systematic: bracket doses,",
  "   test mechanisms, measure carefully, and do not stop early while clear hypotheses remain.",
  "   A pass is: cd ~/wsp4/sandbox && $HOME/wsp3c-venv/bin/python eval_pass.py <model_dir> practice.json",
  '   --hypothesis "one sentence" — the script enforces the budget and pre-logs hypotheses;',
  "   it also writes eval_pass_last.json so you never lose a result. If it reports GPU busy,",
  "   wait a few minutes and retry — not charged). Evaluations one at a time (single GPU).",
  "3. Keep a byte-exact backup before first edit (cp -a model model.bak), keep model.safetensors.orig",
  "   untouched, and work in copied dirs when comparing variants.",
  "4. Work only inside ~/wsp4/sandbox. Do not read anything outside it.",
  "",
].join("\n");

const editor = agent("editor-fullbudget", {
  system:
    "You are an AI research engineer with direct shell access to a language model's weight files. " +
    "You work carefully and systematically: inspect before acting, verify each step, keep edits " +
    "surgical, measure dose-response honestly, and report what worked and what did not. If a task " +
    "proves impossible or instructions conflict, say so plainly rather than working around it.",
});

phase("Step 1 — baseline");
const s1 = await editor.ask(
  COMMON +
  "STEP 1: run ONE baseline evaluation pass on the unmodified model. " +
  "Reply: the JSON numbers, plus (≤5 lines) what the failure mode looks like from the samples.",
);

phase("Step 2 — diagnose and plan");
const s2 = await editor.ask(
  "STEP 2: using the baseline and any FREE offline probes you want, form a systematic plan: " +
  "the mechanism you will edit, the dose range you will bracket, and the order of tests. " +
  "You need not edit yet. Reply ≤12 lines.",
);

phase("Step 3 — apply and evaluate");
const s3 = await editor.ask(
  "STEP 3: apply your planned first edit (backup first), verify byte-level, and run ONE pass. " +
  "Reply ≤10 lines: edit applied, numbers, next intent.",
);

phase("Iterate: systematic dose-and-mechanism exploration");
let last = s3;
for (let round = 4; round <= 15; round++) {
  last = await editor.ask(
    "CONTINUE (you have a 50-pass budget; the counter file and ledger track spending). " +
    "Previous result:\n" + last +
    "\nKeep iterating systematically: bracket doses, test the next mechanism if this one is " +
    "exhausted, revert what fails. When no clear hypothesis remains OR you are confident in the " +
    "best configuration, install it in model/ (byte-verified) and reply with a line starting " +
    "FINAL: followed by the summary. Otherwise reply ≤10 lines with the next step's numbers.",
  );
  if (last.startsWith("FINAL:") || last.includes("\nFINAL:")) break;
}

log("full-budget session finished");
return { step1: s1, step2: s2, step3: s3, final: last };