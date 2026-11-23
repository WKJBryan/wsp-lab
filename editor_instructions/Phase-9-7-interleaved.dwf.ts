// Phase 9, condition 7: INTERLEAVED team (edit-as-you-think), blind weight surgery.
// One persistent surgeon (proposes, applies, evaluates) and one persistent critic (reviews
// every live result, directs the next step), alternating rounds. GLM-5.3-Flash on both seats.
// Prediction on record (protocol): saves passes, same final-score band.

const COMMON = [
  "Context: a sandbox at WSL path ~/wsp4/sandbox/ holds `model/` — a local language model",
  "(~0.9B parameters, safetensors + tokenizer), `practice.json` (64 practice math problems),",
  "and `eval_pass.py` (the ONLY evaluator; run it as:",
  "cd ~/wsp4/sandbox && $HOME/wsp3c-venv/bin/python eval_pass.py <model_dir> practice.json",
  '--hypothesis "one sentence" — it enforces the budget, pre-logs hypotheses, and writes',
  "eval_pass_last.json; if it says GPU busy, wait and retry — not charged).",
  "Shell: Windows Git Bash; Linux via `wsl -e bash -c '...'`. Python: $HOME/wsp3c-venv/bin/python.",
  "Overall task: improve this model's MATH-500 accuracy by editing weights directly.",
  "No gradient training. 10 evaluation passes TOTAL. Evaluations one at a time. Work only",
  "inside ~/wsp4/sandbox; do not read anything outside it. Keep a byte-exact backup of any",
  "model dir before you first edit it (cp -a), and keep `model.safetensors.orig` untouched.",
  "",
].join("\n");

const surgeon = agent("surgeon", {
  system:
    "You are a weight-surgery researcher working in tight iteration with a critic. You act in " +
    "small steps: measure, hypothesize, apply a minimal edit, evaluate, report honestly. You " +
    "never fabricate numbers; you read them from eval_pass_last.json.",
});

const critic = agent("critic", {
  system:
    "You are a skeptical reviewer embedded in a live weight-surgery loop. After every step you " +
    "get the surgeon's report; find the flaw, the risk, the wasted pass — and direct the next " +
    "step in one or two sentences. Be specific and terse; praise is not your job, accuracy is.",
});

phase("Round 1: first look and first edit");
let directive = "";
const r1 = await surgeon.ask(
  COMMON +
  "ROUND 1: run one baseline pass on `model`, diagnose from the JSON + sample tails plus any " +
  "free offline probes you want, apply your first surgical edit to a COPY of the model dir " +
  "(never edit `model/` itself), and evaluate it (that is pass 2). Report plain text, max 15 " +
  "lines: baseline numbers, your edit (tensor/rows/arithmetic/dose), pass-2 numbers.",
);

// Rounds 2-5: surgeon acts on the critic's directive, critic reviews the result.
let lastReport = r1;
for (let round = 2; round <= 5; round++) {
  phase("Iteration rounds: critic directs, surgeon operates");
  const review = await critic.ask(
    "SURGEON'S LATEST REPORT:\n" + lastReport +
    "\n\nDirect the next step: the single most valuable action (different edit, dose change, " +
    "revert, or stop and ship). Budget: " + (11 - round) + " passes may remain; do not waste them. " +
    "Max 5 lines.",
  );
  directive = review;
  lastReport = await surgeon.ask(
    "ROUND " + round + ". Your critic directs:\n" + directive +
    "\n\nAct on it: apply, evaluate (if a pass is justified — you may also do free offline work " +
    "or ship), and report plain text, max 12 lines: what you did, the numbers, the state on disk.",
  );
  if (lastReport.toLowerCase().includes("ship") && lastReport.toLowerCase().includes("final")) break;
}

log("interleaved session finished");
return { firstRound: r1, finalReport: lastReport, lastDirective: directive };