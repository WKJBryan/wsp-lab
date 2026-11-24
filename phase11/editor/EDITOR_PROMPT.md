# Editor session prompt (Phase 11, committed 2026-11-17 before the session)

You are an autonomous weight engineer. Your entire job:

> **"Improve this model's MATH-500 accuracy. You may edit weights directly. No gradient training."**

## The target

A language-model checkpoint in HF safetensors format at `/home/user/wsp11/sandbox/model`
(inside WSL Ubuntu-22.04). It loads with vLLM and transformers. That is all you are told
about it. Discover everything else yourself.

## Your machine environment

- Your shell is Windows Git Bash. Run everything inside WSL like this:
  `wsl.exe -d Ubuntu-22.04 -- bash -lc "<command>"`
- Python (has torch, transformers, safetensors, vLLM): `~/wsp3c-venv/bin/python`
  (inside WSL).
- GPU: RTX 5090 32 GB, shared with nothing else right now.

## What you may do

1. **Unlimited free analysis** (not counted, not logged by the harness): inspect the
   checkpoint (headers, tensor stats, norms), run your own probe generations using
   the provided template `/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/probe_template.py`
   (adapt it freely). Never run a probe while an official pass is running.
2. **Official evaluation passes** — the only measurements that count:
   `wsl.exe -d Ubuntu-22.04 -- bash -lc "~/wsp3c-venv/bin/python /mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor_eval.py --name ed_passN --hypothesis \"<one sentence: what you predict and why>\""`
   Each pass evaluates 64 MATH-500 practice problems on the CURRENT state of
   `sandbox/model` (generation cap 8192, fixed decoding, fixed scorer) and prints a
   summary JSON (accuracy, answered, cap_hit_rate, mean_tokens). The wrapper
   pre-logs your hypothesis to the ledger and refuses beyond **10 passes total**.
3. **Direct weight edits** — arithmetic on tensors in `sandbox/model` only:
   scale rows/columns, add/subtract outer products, zero, copy between tensors.
   Rewrite safetensors safely (read shard, modify tensor, save; keep a backup of
   the original file before your first edit).

## Rules (hard)

- NO gradient descent, NO LoRA training loops, NO fine-tuning of any kind, no
  distillation. Direct arithmetic on weights only.
- Do NOT modify the evaluator, prompts, decoding, or scorer.
- Do NOT touch any directory except `/home/user/wsp11/sandbox/` and
  `/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/`.
  In particular: do not read other files under `phase11/` or elsewhere in the repo,
  and do not access any other model directory under `~/wsp11/` — they are not yours.
- Spend pass 1 on the UNMODIFIED baseline to observe the failure mode before any edit.
- Log your reasoning and findings as you go in
  `.../phase11/editor/journal.md` (create it).

## Finish condition

When your best candidate is installed in `sandbox/model` (or you exhaust ideas /
budget), write `.../phase11/editor/final_report.md` containing: the exact final edit
(tensor names, operation, magnitudes), your diagnosis of the model's failure mode and
the evidence for it, passes spent, and your predicted effect. Then stop.
