#!/bin/bash
# Run battery3 on a list of variant dirs: runbat3.sh tag1 tag2 ...
ED=/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor
PY=~/wsp3c-venv/bin/python
for v in "$@"; do
  "$PY" "$ED/battery3.py" "/home/user/wsp11/sandbox/m_$v" 2048 "$v" 2>/dev/null | tail -1
  tail -1 "$ED/battery3_$v.txt"
done
