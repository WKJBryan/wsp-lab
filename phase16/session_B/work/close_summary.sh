#!/bin/bash
# close outcomes (clean / leaky / right) for every measured attempt
cd ~/wsp14/work
for n in 1 2 3 4 5 8 9 10 11 12; do
  ~/wsp3c-venv/bin/python close_table.py $n logs/close_table_$n.txt
done
~/wsp3c-venv/bin/python - <<'EOF'
import json, sys
sys.path.insert(0, '/home/user/wsp14/work')
import reads as R
print("attempt | closed | finished cleanly (right) | closed but ran to the cap (right) | never closed (right by a box inside the thinking) | right (boxed rule) | right (second extractor)")
for n in (1, 2, 3, 4, 5, 8, 9, 10, 11, 12):
    rows = R.gen_rows(n)
    s = json.load(open(f'/home/user/wsp14/sandbox/harness/state/eval_pass_{n}.json'))
    cl = [r for r in rows.values() if r['closed']]
    clean = [r for r in cl if r['finish'] == 'stop']
    leak = [r for r in cl if r['finish'] != 'stop']
    nc = [r for r in rows.values() if not r['closed']]
    print(f"{n:7d} | {len(cl):6d} | {len(clean):3d} ({sum(r['ok_boxed'] for r in clean):2d}) | {len(leak):3d} ({sum(r['ok_boxed'] for r in leak):2d}) | "
          f"{len(nc):3d} ({sum(r['ok_boxed'] for r in nc):2d}) | {s['accuracy_boxed']:5.1f} | {s['accuracy_rescore']:5.1f}")
EOF
