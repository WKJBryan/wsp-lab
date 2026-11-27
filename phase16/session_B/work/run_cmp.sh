#!/bin/bash
# paired comparisons for the robustness draws
cd ~/wsp14/work
for pair in "9 10" "2 11" "8 11" "11 10" "2 3" "3 9" "3 10"; do
  set -- $pair
  echo "== pass $1 vs pass $2"
  CUDA_VISIBLE_DEVICES= ~/wsp3c-venv/bin/python cmp_pass.py "$1" "$2" 2>&1 | sed -n '3,5p' | cut -c1-170
done
for n in 10 11; do
  CUDA_VISIBLE_DEVICES= ~/wsp3c-venv/bin/python - <<EOF
import json
d = json.load(open('/home/user/wsp14/sandbox/harness/state/eval_pass_$n.json'))
print('pass $n:', {k: d[k] for k in ('accuracy_boxed', 'accuracy_rescore', 'closed', 'cap_hit_rate', 'mean_tokens', 'wall_s')}, d['by_level'])
EOF
done
CUDA_VISIBLE_DEVICES= ~/wsp3c-venv/bin/python close_table.py 9 logs/close_table_p9.txt; tail -1 logs/close_table_p9.txt | cut -c1-100
grep -c . logs/close_table_p9.txt
