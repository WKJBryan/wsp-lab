#!/bin/bash
# usage: run_save.sh <outname> <script_basename> [args...]
# runs scripts/<basename>.py on CPU, saves stdout+stderr to work/<outname>.txt
OUT="$1"; SCRIPT="$2"; shift 2
CUDA_VISIBLE_DEVICES='' /home/user/wsp3c-venv/bin/python "/home/user/wsp14/work/scripts/${SCRIPT}.py" "$@" > "/home/user/wsp14/work/${OUT}.txt" 2>&1
echo "EXIT=$? saved /home/user/wsp14/work/${OUT}.txt"
