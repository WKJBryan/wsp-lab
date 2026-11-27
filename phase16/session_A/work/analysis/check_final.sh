#!/bin/bash
# Verify that the working model differs from the pristine base in exactly one file (the weight shard).
B=/home/user/wsp14/sandbox/base_model
M=/home/user/wsp14/sandbox/model
same=0; diff=0
while IFS= read -r f; do
  rel="${f#$B/}"
  if [ -f "$M/$rel" ]; then
    if cmp -s "$f" "$M/$rel"; then same=$((same+1)); else diff=$((diff+1)); echo "DIFFERENT: $rel"; fi
  else
    echo "MISSING in model: $rel"
  fi
done < <(find "$B" -type f | sort)
extra=$(find "$M" -type f | while IFS= read -r f; do rel="${f#$M/}"; [ -f "$B/$rel" ] || echo "$rel"; done)
echo "files identical to the base: $same; different: $diff; extra files in model: ${extra:-none}"
sha256sum "$M/model.safetensors-00001-of-00001.safetensors" "$B/model.safetensors-00001-of-00001.safetensors" | cut -c1-16,65-
cat /home/user/wsp14/sandbox/harness/state/passes.txt
