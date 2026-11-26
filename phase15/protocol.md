# Pre-registered protocol — Phase 15: the blind editor under the original authors' rules, with a stronger editor

> **Technical check (2026-11-26, 22:51; run before the session, and its outcome sets nothing).** Four
> practice problems on the unmodified model in the fresh sandbox, under the full protocol: exit status 0;
> one pass charged; all four generations ran to the cap; the texts and the summary were written; the
> summary's base fingerprint equals `phase15/sandbox_manifest.json` and its tools digest equals the
> digest of the manifest's `harness` object (`36d3430c…`, `b57047ce50bc9935`). The ledger and counter were
> then reset: the sandbox holds no pass, and the evidence is kept outside it. The phase-14 sandbox was
> moved aside whole before the fresh one was built (renamed, not deleted).

> **Note 2026-11-26, 22:58 (before the editor session; no rule, endpoint or prediction changed).** The
> editor type could not be used as registered. The harness fixes its list of helper types when a chat
> starts, and `glm54-editor` was defined afterwards (a copy of the Flash editor's definition with only the
> model changed), so it cannot be loaded into the chat that runs the session. The editor therefore runs as
> the harness's built-in general helper, which takes the chat's model, GLM-5.4. Before the session a probe
> of that helper (no tools used) showed what it is handed at its start: the user's general working rules,
> which say nothing about this project, the fault or mathematics, and a list of tool names; no project
> notes. To give it what the editor type would have given, its briefing starts with the four operating
> rules of the editor types, word for word, and the sentence that its tools are exactly those of the editor
> types (Read, Glob, Grep, Write, Edit and Bash; no web tools, no other helpers). The helper has more
> tools than that, so the sentence is enforced by instruction and audited afterwards, like the rules of
> the briefing. The first attempt to start the registered type was refused by the harness; nothing ran.

Phase 14 ran its experiment with the protocol's fallback editor, GLM-5.3-Flash, a small model, and that
editor did not reach the published edit (11.0% against 54.2% on MATH-500). It left a question phase 14
could not answer: is the gap a matter of the editor's strength, or of the hint and the bigger budget that
the original authors' editor had?

This phase changes one thing and keeps everything else: **the editor.** It is GLM-5.4, a larger model of
the same family, run blind under the same rules, with the same briefing, the same practice set and the
same budget as the phase-14 editor.

## What is inherited unchanged from phase 14

`phase14/protocol.md` (pre-registered 2026-11-24) governs everything not named here:

- the model: stock Qwen3.5-0.8B in a sandbox; the editor is not told its name;
- the rules, enforced in software by `phase14/save_edit.py` (a candidate without its receipt is never
  evaluated);
- the practice pool `phase14/practice.json` (128 problems from the MATH training split) and the
  measurement protocol (the original authors': the reference question, thinking enabled, temperature 0.6,
  top_p 0.95, top_k 20, presence penalty 1.5, at most 32,768 new tokens, one sample, a fixed seed);
- the evaluation tool `phase14/eval_pass.py` and the budget of 15 passes, a pass refunded on a technical
  failure;
- the briefing: `phase14/EDITOR_PROMPT.md`, byte for byte the file the phase-14 editor received (it names
  the sandbox paths `~/wsp14/...`, so the sandbox is rebuilt at the same place);
- the endpoints and the decision rule (restated below) and the audit after the session (tools, base model,
  practice file, the tensors changed, the editor's scripts).

Two things differ from phase 14 and are recorded here. The tools carry the answer checker's third rule of
2026-11-26 (ISSUES #30, #31; stamp `#d74fdebd340f`), so three of the seven tool files have other digests
than in phase 14; they are in `phase15/sandbox_manifest.json`. And every comparator is read under that
checker.

## Setup

- **Editor:** one context-sterile session on GLM-5.4, run through the harness's built-in general helper
  because the registered `glm54-editor` type cannot be loaded into the running chat (the note above). It
  is briefed with the editor types' four operating rules, their tool list, and the path of
  `phase14/EDITOR_PROMPT.md`; nothing else. One autonomous session. The host sends no message unless a
  turn ends before the finish condition; then it sends a neutral "continue" and nothing else, and each
  message is recorded.
- **Sandbox:** rebuilt with `phase14/make_sandbox.sh` at `~/wsp14/sandbox`, after the phase-14 sandbox
  has been moved aside whole (renamed, not deleted).
- **Technical check:** as in phase 14. Four practice problems on the unmodified model before the session;
  its ledger and counter are then reset, and its outcome sets nothing below.

## Endpoints (fixed now)

Finals on MATH-500 (the 500 scored problems; one sample; the seed of this repository's other finals), run
only after the checker has accepted the final weights. If the editor's final weights equal the base model,
this repository's stock final stands in for the candidate's and no final is run.

- **Primary — the repository's question** (bare question; `phase3c/rescore.py`), paired against stock
  (9.4%) and the published edit (54.2%).
- **Exploratory, new in this phase:** the same final paired against the phase-14 editor's final (11.0%;
  `phase3c/p14_final`) on the same 500 problems.
- **Secondary — the reference question** (last boxed answer; math-verify and the repository's matcher on
  the same boxed answers), paired against the reference arms `ref_stock` and `ref_orig`.
- **Verdict, primary protocol:** the words of phase 14.
  - REPAIR if the final is at least 16.0% and at least stock + 10.0 points (the standing bars).
  - Against the published edit (paired difference with a 95% interval): REACHES IT if within ±4.0 points;
    EXCEEDS if more than +4.0 with the interval excluding 0; FALLS SHORT if less than −4.0 with the
    interval excluding 0; otherwise INCONCLUSIVE.
- Also recorded: the tensors the final edit changed (from its receipt), passes spent, every candidate the
  checker refused and why, the editor's own diagnosis, and the session's length.

## Predictions (on record before the session)

| # | Prediction | Point estimate |
|---|---|---|
| P1 | The first pass, on the unmodified model, has at least half of its 128 generations cut at the cap. | 0.95 |
| P2 | The final candidate FALLS SHORT of the published edit under the primary protocol. | 0.80 |
| P3 | REPAIR is not met. | 0.65 |
| P4 | The editor ships an edit: the final weights differ from the base model in at least one tensor. | 0.90 |
| P5 | The final candidate scores above the phase-14 editor's final by a paired difference whose 95% interval excludes zero. | 0.40 |

The estimates are the author's guesses, made before any pass of this phase.

## What each outcome would change

- **P2 wrong** (the editor reaches or exceeds the published edit): a larger blind editor, held to the
  original rules and practising in distribution with 15 passes, finds a repair of the hinted frontier
  editor's level. The gap left by phase 14 was a matter of the editor's strength, not of the hint or of
  the budget.
- **P2 right and P3 wrong** (a repair that falls short of the published edit): a larger blind editor finds
  a real repair without the hint. The original's level needs more than rules and in-distribution
  practice; repair does not need the hint.
- **P2 and P3 right** (no repair): two blind editors of different strength fail alike under these rules
  and this budget. Editor strength alone does not explain the gap; what remains unseparated is the hint
  and the budget (15 passes of 128 problems against 50 of up to 512).
- **P5** compares the two editors directly. It is exploratory: one session each and one seed each, so it
  cannot separate the editors' strength from the luck of a session.

## Cost

About 1.6 GPU-hours fixed (a technical check and two finals) and about 12 minutes of graphics-card time
per practice pass the editor uses: 2 to 5 GPU-hours in all on an idle machine. The editor's own session
runs on the host's model plan; the phase-14 editor's session used about 86 million tokens in 100 minutes,
most of it re-reads of its own context.

Committed BEFORE the editor session and BEFORE any practice pass of this phase (2026-11-26). Do not edit
after the session begins; deviations go to ISSUES.md.
