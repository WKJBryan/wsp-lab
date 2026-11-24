#!/bin/bash
cd /mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor
PY=~/wsp3c-venv/bin/python
"$PY" probe.py \
  "The capital of France is Paris. The capital of Germany is" \
  "Problem: What is 17 * 23? Solution:" \
  "Q: What is the capital of France?
A:" \
  "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends with four every day. She sells the remainder at the farmers market daily for \$2 per fresh duck egg. How much in dollars does she make every day at the farmers market?" \
  --max-tokens 100 --greedy > probe_batch3.txt 2>/dev/null
echo DONE
