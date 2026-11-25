#!/bin/bash
# usage: wrap.sh <script_basename> [args...]  (run .py from scripts dir on CPU)
SCRIPT="$1"; shift
CUDA_VISIBLE_DEVICES='' /home/user/wsp3c-venv/bin/python "/home/user/wsp14/work/scripts/${SCRIPT}.py" "$@"
