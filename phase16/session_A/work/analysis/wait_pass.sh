#!/bin/bash
# Wait until the summary of pass $1 exists (or until $2 seconds have passed), then print it.
n=${1:-1}
max=${2:-580}
f=/home/user/wsp14/sandbox/harness/state/eval_pass_${n}.json
t=0
while [ ! -f "$f" ] && [ $t -lt $max ]; do
  sleep 5
  t=$((t + 5))
done
date
ls -la /home/user/wsp14/sandbox/harness/state/
cat "$f" 2>/dev/null
echo
cat /home/user/wsp14/sandbox/harness/state/ledger.md
