#!/bin/bash
cd ~/wsp14/work
for pair in "2 8" "3 9" "2 3" "2 5" "3 4"; do
  set -- $pair
  echo "== passes $1 vs $2"
  CUDA_VISIBLE_DEVICES= nice ~/wsp3c-venv/bin/python diverge.py "$1" "$2"
done
