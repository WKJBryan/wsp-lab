#!/bin/bash
# Wait until file $1 contains pattern $2 (grep -F), or until $3 seconds have passed. Then print the tail.
f=$1
pat=${2-EXIT=}
max=${3:-560}
t=0
while ! grep -qF -- "$pat" "$f" 2>/dev/null && [ $t -lt $max ]; do
  sleep 5
  t=$((t + 5))
done
date
echo "waited ${t}s; finished: $(grep -cF -- "$pat" "$f" 2>/dev/null)"
tail -n ${4:-4} "$f"
