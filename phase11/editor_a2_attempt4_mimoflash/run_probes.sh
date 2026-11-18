#!/bin/bash
# Free probe runner: passes each question arg to probe.py
cd /mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor
PY=~/wsp3c-venv/bin/python
for q in "$@"; do
  echo "=== Q: $q"
  "$PY" probe.py "$q" --max-tokens 80 --greedy 2>/dev/null | sed -n '/finish:/,/--- end ---/p'
done
