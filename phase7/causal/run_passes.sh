#!/usr/bin/env bash
# Phase-7 causal spot-checks: the six pre-registered practice passes, in the fixed order.
# Runs inside WSL. Harness = phase5/eval_pass5.py, unchanged; pass_index 1 for every arm.
# Waits for the GPU instead of fighting another process; one retry per arm on engine failure.
# Usage (from Windows): wsl.exe -e bash /mnt/c/<repo>/phase7/causal/run_passes.sh
set -u
W=$HOME/wsp7
: > "$W/ledger.txt"
for arm in stock brake_d100 brake_rand brake_d050 brake_d200 brake_sel; do
  dir=$W/$arm; [ "$arm" = stock ] && dir=$HOME/wsp3c/stock
  for try in 1 2; do
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt 27000 ]; do sleep 30; done
    echo "$(date -Is) start $arm try $try" >> "$W/ledger.txt"
    "$HOME/wsp3c-venv/bin/python" "$HOME/wsp5/eval_pass5.py" "$dir" "$HOME/wsp5/practice.json" 1 > "$W/pass_$arm.log" 2>&1
    if grep -q '^{"n"' "$W/pass_$arm.log"; then echo "$(date -Is) done $arm" >> "$W/ledger.txt"; break; fi
    echo "$(date -Is) FAILED $arm try $try" >> "$W/ledger.txt"
  done
  sleep 20
done
grep -h '^{"n"' "$W"/pass_*.log | wc -l
