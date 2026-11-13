#!/usr/bin/env bash
# Phase-7 causal spot-checks: pre-registered practice passes, in the order given.
# Runs inside WSL. Harness = phase5/eval_pass5.py, unchanged.
# Arms are "name" or "name@pass_index" (default index 1). No arguments = the first batch of six.
# Waits for the GPU instead of fighting another process; one retry per arm on engine failure.
# Usage (from Windows): wsl.exe -e bash -lc 'bash /mnt/c/<repo>/phase7/causal/run_passes.sh [arm ...]'
set -u
W=$HOME/wsp7
[ $# -eq 0 ] && set -- stock brake_d100 brake_rand brake_d050 brake_d200 brake_sel
for spec in "$@"; do
  arm=${spec%@*}; idx=1; [ "$spec" != "$arm" ] && idx=${spec#*@}
  dir=$W/$arm
  [ "$arm" = stock ] && dir=$HOME/wsp3c/stock
  [ "$arm" = orig ] && dir=$HOME/wsp3c/c3_original_edit
  log=$W/pass_${arm}_p${idx}.log
  for try in 1 2; do
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt 27000 ]; do sleep 30; done
    echo "$(date -Is) start $arm pass_index $idx try $try" >> "$W/ledger.txt"
    "$HOME/wsp3c-venv/bin/python" "$HOME/wsp5/eval_pass5.py" "$dir" "$HOME/wsp5/practice.json" "$idx" > "$log" 2>&1
    if grep -q '^{"n"' "$log"; then echo "$(date -Is) done $arm" >> "$W/ledger.txt"; break; fi
    echo "$(date -Is) FAILED $arm try $try" >> "$W/ledger.txt"
  done
  sleep 20
done
