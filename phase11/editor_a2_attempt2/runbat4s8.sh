#!/bin/bash
# Run battery4s8 (seed 8) on variants: runbat4s8.sh tag1 tag2 ...
ED=/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor
PY=~/wsp3c-venv/bin/python
for v in "$@"; do
  dir="/home/user/wsp11/sandbox/m_$v"
  [ "$v" = "base" ] && dir="/home/user/wsp11/sandbox/model"
  "$PY" "$ED/battery4s8.py" "$dir" 2048 "s8_$v" 2>/dev/null | tail -1
done
