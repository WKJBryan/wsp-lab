#!/bin/bash
# Put the chosen weights back into sandbox/model and have the checker accept them.
set -u
W=~/wsp14/work
NAME=${1:-R1E2}
export NAME
cp $W/cands/$NAME/model.safetensors-00001-of-00001.safetensors ~/wsp14/sandbox/model/
~/wsp3c-venv/bin/python ~/wsp14/sandbox/harness/save_edit.py ~/wsp14/sandbox/model --note "$(cat $W/cands/$NAME.note)" 2>&1 | grep -v 'Warning\|info.append\|Consider'
echo "--- fingerprint check (sandbox/model vs the candidate copy)"
~/wsp3c-venv/bin/python $W/fpcheck.py
echo "--- receipt on file"
~/wsp3c-venv/bin/python - <<'EOF'
import os, sys, json, glob
sys.path.insert(0, '/home/user/wsp14/sandbox/harness')
from save_edit import fingerprint
fp = fingerprint('/home/user/wsp14/sandbox/model')
r = f'/home/user/sandbox/harness/state/receipts/{fp}.json'
r = f'/home/user/wsp14/sandbox/harness/state/receipts/{fp}.json'
print('fingerprint', fp)
print('receipt exists:', os.path.exists(r))
if os.path.exists(r):
    d = json.load(open(r))
    print('receipt time', d['time'], '| changed tensors', len(d['changed_tensors']))
EOF
ls -la ~/wsp14/sandbox/model/ | head -30
cat ~/wsp14/sandbox/harness/state/passes.txt
