#!/bin/bash
# usage: build_E_variants.sh   (threshold, delta, gamma, b1, name) lines below
cd ~/wsp14/work
while read -r theta delta gamma b1 name; do
  [ -z "$theta" ] && continue
  PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 ~/wsp3c-venv/bin/python fit_E.py --theta "$theta" --delta "$delta" --gamma "$gamma" --b1 "$b1" --out "$name" 2>&1 | grep -E 'boost at|^saved|z >= ' | head -3
done <<'EOF'
25 2 1 5 E2
26 2 1 5 E3
23.5 2 1 4 E4
EOF
