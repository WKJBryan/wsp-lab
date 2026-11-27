#!/bin/bash
# usage: kill_jobs.sh <pattern>   (kills my python jobs whose command line contains the pattern)
for p in $(pgrep -f "$1"); do
  case "$p" in $$) ;; *) kill "$p" 2>/dev/null ;; esac
done
sleep 3
pgrep -af "$1" | grep -v pgrep | head -5
nvidia-smi --query-gpu=memory.used --format=csv,noheader
