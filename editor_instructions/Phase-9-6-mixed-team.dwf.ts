// Phase 9, condition 6: MIXED-family team deliberation (blind weight surgery).
// Role assignment (disclosed in phase9/results.md): all four editor models take seats;
// producer-critical roles (judge, executor) on GLM-family models with proven long-session
// discipline; mimo tiers on analyst/red-team seats (fragments recoverable by the judge).

const COMMON = [
  "Context: a sandbox at WSL path ~/wsp4/sandbox/ holds `model/` — a local language model",
  "(~0.9B parameters, safetensors + tokenizer), `practice.json` (64 practice math problems),",
  "and `eval_pass.py` (the ONLY evaluator). Shell: Windows Git Bash; run Linux commands via",
  "`wsl -e bash -c '...'`. Python with torch/safetensors: $HOME/wsp3c-venv/bin/python (in WSL).",
  "Overall task: improve this model's MATH-500 accuracy by editing weights directly.",
  "No gradient training. 10 evaluation passes TOTAL across the whole team (a pass is:",
  "cd ~/wsp4/sandbox && $HOME/wsp3c-venv/bin/python eval_pass.py <model_dir> practice.json",
  '--hypothesis "one sentence"; the script enforces the budget and pre-logs hypotheses;',
  "if it reports GPU busy, wait a few minutes and retry — not charged). Evaluations one at a",
  "time. Work only inside ~/wsp4/sandbox; do not read anything outside it.",
  "",
].join("\n");

const READONLY = COMMON +
  "YOUR CONSTRAINTS: read-only on `model/` — do NOT edit any weight file, and do NOT run " +
  "eval_pass.py (only the observer/executor may spend passes). Free offline analysis is " +
  "encouraged: load tensors, run forward passes, inspect the tokenizer — anything that does " +
  "not modify `model/` and does not call eval_pass.py.\n";

phase("Observe: baseline evidence");
const obs = await agent("observer", { model: "account:zai-individual-coding-plan/GLM-5.3-Flash" }).ask(
  COMMON +
  "ROLE: OBSERVER. Run exactly ONE baseline evaluation pass on the unmodified model. " +
  "Report, verbatim: the full JSON output and the sample tails it contains. Plain text only.",
);

const evidence = obs + "\n\n(That is ALL the behavioral evidence you get; you may gather more offline for free.)";

phase("Analyze: two independent proposals");
// Red-team needs BOTH proposals before it can rank them — this is the one barrier.
const [pA, pB] = await Promise.all([
  agent("analyst-a", { model: "xiaomi-mimo/mimo-v2.6-pro" }).ask(
    READONLY +
    "ROLE: ANALYST A. Baseline evidence from the observer:\n" + evidence +
    "\nDiagnose the failure from the evidence and any free probes you run, then propose " +
    "ONE concrete weight edit: exact tensor, exact rows/columns, exact arithmetic, exact dose, " +
    "with your rationale and the single biggest risk you see. Plain text, max 20 lines.",
  ),
  agent("analyst-b", { model: "account:zai-individual-coding-plan/GLM-5.3" }).ask(
    READONLY +
    "ROLE: ANALYST B. Baseline evidence from the observer:\n" + evidence +
    "\nWork INDEPENDENTLY: diagnose from the evidence and any free probes you run, then propose " +
    "ONE concrete weight edit: exact tensor, exact rows/columns, exact arithmetic, exact dose, " +
    "with your rationale and the single biggest risk you see. Plain text, max 20 lines.",
  ),
]);

phase("Red-team both proposals");
const critique = await agent("red-team", { model: "xiaomi-mimo/mimo-v2.6-flash" }).ask(
  READONLY +
  "ROLE: RED-TEAM. Two proposals were made independently:\n\nPROPOSAL A:\n" + pA +
  "\n\nPROPOSAL B:\n" + pB +
  "\n\nAttack both: what breaks at higher doses? What could cause premature termination or " +
  "degenerate output? Which risks are measurable for free right now? Measure what you can " +
  "with free probes, then output: for each proposal — flaws, required modifications, and a " +
  "dose-safety range you would NOT exceed; then your ranking. Plain text, max 25 lines.",
);

phase("Judge: final edit specification");
const verdict = await agent("judge", { model: "account:zai-individual-coding-plan/GLM-5.3-Flash" }).ask(
  READONLY +
  "ROLE: JUDGE. Given the proposals and the red-team critique, decide the FINAL edit the " +
  "executor will apply. You may synthesize. Output the exact specification: tensor, rows, " +
  "arithmetic, dose, backup steps, and an allowed dose-bracketing range for the executor " +
  "(±25% of your chosen dose at most). Plain text, max 15 lines." +
  "\n\nPROPOSAL A:\n" + pA + "\n\nPROPOSAL B:\n" + pB + "\n\nRED-TEAM CRITIQUE:\n" + critique,
);

phase("Execute: apply, test, finalize");
const execReport = await agent("executor", { model: "account:zai-individual-coding-plan/GLM-5.3" }).ask(
  COMMON +
  "ROLE: EXECUTOR. The judge's specification (apply it EXACTLY as written first — including " +
  "any backup steps it requires; then you may bracket doses only within the judge's allowed " +
  "range, using at most 4 further passes; revert to the backup freely; end with the model dir " +
  "you evaluated holding the best configuration you measured):\n\n" + verdict +
  "\n\nReport plain text: what was applied, every pass spent with its numbers, and the final " +
  "configuration on disk.",
);

log("mixed team session finished");
return { observer: obs, proposalA: pA, proposalB: pB, critique, verdict, executor: execReport };