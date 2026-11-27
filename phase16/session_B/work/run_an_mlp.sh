#!/bin/bash
# usage: run_an_mlp.sh <layers...>
cd ~/wsp14/work
for L in "$@"; do
  PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 ~/wsp3c-venv/bin/python an_mlp.py r1/mlp 1 --layer "$L" --top 8 2>&1 | grep -v falling | cut -c1-520
  echo
done
