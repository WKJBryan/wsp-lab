#!/usr/bin/env bash
# Evaluate the complete grafted-neuron response for several thresholds.
cd ~/wsp14/work
for th in 425 445 460; do
  CUDA_VISIBLE_DEVICES='' ~/wsp3c-venv/bin/python scripts/graft_eval.py --theta "$th" --s 20 --c 5.5 --flip 3.5 2>&1 | grep -v -i -E 'warn|^$' | cut -c1-700
  echo
done
