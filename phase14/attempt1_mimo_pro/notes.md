# Phase 14 weight-editing session notes

Model: Qwen3.5-style hybrid (24L x 1024; layers 3,7,11,15,19,23 full-attn 8Q/2KV hd256;
others Gated DeltaNet linear attn 16h x 128; SwiGLU 3584; vocab 248320; tied embeddings).
Protected (checker): embed_tokens.weight, lm_head.weight, language_model.norm.weight.

## Plan
1. Pass 1: unmodified model (baseline). Read generations -> diagnose failure mode.
2. Teacher-forced reads (CPU, free) to localize the failure in the network.
3. Targeted weight edits, save_edit.py -> eval pass. Keep only gains clearly > SE (~4.4 pts).
4. Copy good states to ~/wsp14/work/.

## Log
- 2026-11-25 session start. Sandbox fresh (0 passes). Launching pass 1 (base model).
- 2026-11-25 env fix: Git Bash MSYS path conversion was rewriting WSL paths (C:/Program Files/Git/...).
  All wsl.exe invocations now use MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'. Launcher: work/run_pass.sh.
  Earlier launch attempts died before eval_pass.py ran: no pass charged (passes.txt only now written).
- Weight survey (work/weight_survey.txt): all LM tensors dense, no zeros, norms in expected ranges.
  Row-outlier hunt (outlier_hunt.py): cold/hot rows are consistent across layers (dead-neuron pattern),
  nothing like a localized injected defect. So the failure is behavioral, not an obviously broken tensor.
- Pass 1 (base, n=128) running at 22:04. While waiting: CPU-only work.
