"""Phase-7 addendum 10, step 0: how many of a run's cap-hit answers are repeating loops?

Loop = the detector of build_sets.py (copied: importing that file rebuilds the sets): 12-gram
repeats above 50% in four consecutive 250-token bins.

Usage: uv run python phase7/count_loops.py p3c_c3_original
"""

import gzip
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
N, BIN = 12, 250
tok = AutoTokenizer.from_pretrained(ROOT / "phase3" / "models" / "stock")


def onset_of(ids):
    seen, flags = set(), []
    for i in range(len(ids) - N):
        g = tuple(ids[i:i + N])
        flags.append(g in seen)
        seen.add(g)
    bins = [sum(flags[i:i + BIN]) / BIN for i in range(0, len(flags) - BIN + 1, BIN)]
    for k in range(len(bins) - 3):
        if all(x > 0.5 for x in bins[k:k + 4]):
            return k * BIN
    return None


assert onset_of(list(range(3000))) is None and onset_of(list(range(500)) + [1, 2, 3] * 1000) is not None
run = sys.argv[1]
texts = json.loads(gzip.open(ROOT / "phase3c" / f"{run}_texts.json.gz", "rt", encoding="utf-8").read())
rows = json.loads((ROOT / "phase3c" / f"{run}.json").read_text(encoding="utf-8"))["rows"]
cap = [r["id"] for r in rows if r["finish"] == "length"]
onsets = {i: onset_of(tok(texts[i], add_special_tokens=False).input_ids) for i in cap}
loops = [i for i in cap if onsets[i] is not None]
on = sorted(onsets[i] for i in loops)
print(f"{run}: {len(rows)} answers, {len(cap)} cap-hit ({len(cap) / len(rows):.1%}); {len(loops)} of those are loops ({len(loops) / max(1, len(cap)):.1%})")
if on:
    print(f"loop onset (tokens into the answer): median {on[len(on) // 2]}, quartiles {on[len(on) // 4]}-{on[3 * len(on) // 4]}")
