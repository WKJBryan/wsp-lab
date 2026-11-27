#!/bin/bash
# usage: launch_pass.sh <candidate name> <hypothesis file> [n]
# 1. runs the edit checker on the candidate directory (issues the receipt),
# 2. installs the candidate's weight file into sandbox/model and checks that both fingerprints agree,
# 3. runs one forward pass on sandbox/model with the hypothesis.
# Everything is logged under ~/wsp14/work/logs/.
set -u
NAME=$1
HYP=$2
N=${3:-128}
W=~/wsp14/work
C=$W/cands/$NAME
PY=~/wsp3c-venv/bin/python
if [ ! -s "$HYP" ]; then echo "hypothesis file missing or empty: $HYP (nothing launched)"; exit 1; fi
if [ ! -s "$W/cands/$NAME.note" ]; then echo "note file missing or empty: $W/cands/$NAME.note (nothing launched)"; exit 1; fi
export NAME
cd $W
$PY ~/wsp14/sandbox/harness/save_edit.py $C --note "$(cat $W/cands/$NAME.note)" > $W/logs/check_$NAME.log 2>&1
if ! grep -q ACCEPTED $W/logs/check_$NAME.log; then
  echo "CHECKER REFUSED $NAME"; cat $W/logs/check_$NAME.log; exit 1
fi
cp $C/model.safetensors-00001-of-00001.safetensors ~/wsp14/sandbox/model/
$PY $W/fpcheck.py
if [ $? -ne 0 ]; then echo "FINGERPRINT MISMATCH: not running the pass"; exit 1; fi
echo "other GPU processes (should be none):"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
$PY ~/wsp14/sandbox/harness/eval_pass.py ~/wsp14/sandbox/model --hypothesis "$(cat $HYP)" --n $N > $W/logs/pass_$NAME.log 2>&1
echo "EXIT=$?" >> $W/logs/pass_$NAME.log
grep -v '^Processed\|^Capturing\|^Rendering\|WARNING\|INFO\|it/s\]' $W/logs/pass_$NAME.log | tail -5 | cut -c1-3000
