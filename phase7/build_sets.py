"""Phase-7 atlas, part 2 — fix the STUCK / CLEAN lists BEFORE any activation probing.

Protocol rule: "stuck set and clean set are fixed lists committed before probing."
CPU only (tokenizer + saved phase-3c generations). Output: phase7/sets.json.

Loop onset := first 250-token bin where >50% of 12-token windows already occurred earlier
in the same generation, sustained for 4 bins (1,000 tokens).

Groups (token positions index the generated text, 0-based):
  stuck  stock's own cap-hit, never-closed generations with onset in [2000, 8000] (80, seeded)
           pre  = 32 positions in [200, onset-500)      same problem, before the loop
           loop = 32 positions in [onset+500, onset+1500)  deep in sustained repetition
  clean  every stock generation that closed thinking and terminated; 32 positions in thinking
  deep   the original-edit model's clean generations with >= 9,000 thinking tokens and no
         onset (60, seeded); 32 positions in [2500, 9000) — a depth-matched, non-looping
         control for `loop`. Caveat: text written by the edited model, read by stock.
  half   0 = fit, 1 = held-out (alternating) — directions are fit on 0 and scored on 1.

Usage: uv run python phase7/build_sets.py
"""

import gzip
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
P3C = HERE.parent / "phase3c"
N, BIN, SEED, NPOS = 12, 250, 7, 32

tok = AutoTokenizer.from_pretrained(HERE.parent / "phase3" / "models" / "stock")
rng = random.Random(SEED)


def load(run):
    texts = json.loads(gzip.open(P3C / f"{run}_texts.json.gz", "rt", encoding="utf-8").read())
    rows = {r["id"]: r for r in json.loads((P3C / f"{run}.json").read_text(encoding="utf-8"))["rows"]}
    return texts, rows


def ids_of(text):
    return tok(text, add_special_tokens=False).input_ids


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


def pick(lo, hi):
    return sorted(rng.sample(range(lo, hi), NPOS))


items = []

texts, rows = load("p3c_stock")
cand = []
for pid in sorted(texts, key=lambda s: int(s[1:])):
    t = texts[pid]
    if rows[pid]["finish"] == "length" and "</think>" not in t:
        on = onset_of(ids_of(t))
        if on is not None and 2000 <= on <= 8000:
            cand.append((pid, on))
    elif rows[pid]["finish"] != "length" and "</think>" in t:
        think = len(ids_of(t.split("</think>")[0]))
        if think >= 400:
            items.append(dict(id=pid, run="p3c_stock", group="clean", think_len=think, n_ctx=think,
                              positions=dict(clean=pick(200, think))))
for pid, on in sorted(rng.sample(cand, 80), key=lambda x: int(x[0][1:])):
    items.append(dict(id=pid, run="p3c_stock", group="stuck", onset=on, n_ctx=on + 1500,
                      positions=dict(pre=pick(200, on - 500), loop=pick(on + 500, on + 1500))))
n_stuck_eligible = len(cand)

texts, rows = load("p3c_c3_original")
cand = []
for pid in sorted(texts, key=lambda s: int(s[1:])):
    t = texts[pid]
    if rows[pid]["finish"] != "length" and "</think>" in t:
        ids = ids_of(t.split("</think>")[0])
        if len(ids) >= 9000 and onset_of(ids) is None:
            cand.append(pid)
for pid in sorted(rng.sample(cand, min(60, len(cand))), key=lambda s: int(s[1:])):
    pos = pick(2500, 9000)
    items.append(dict(id=pid, run="p3c_c3_original", group="deep", n_ctx=pos[-1] + 1, positions=dict(deep=pos)))
n_deep_eligible = len(cand)

for g in ("stuck", "clean", "deep"):
    for k, it in enumerate(x for x in items if x["group"] == g):
        it["half"] = k % 2

out = dict(
    definition=dict(ngram=N, bin=BIN, seed=SEED, positions_per_region=NPOS,
                    stuck_eligible=n_stuck_eligible, deep_eligible=n_deep_eligible),
    counts={g: sum(x["group"] == g for x in items) for g in ("stuck", "clean", "deep")},
    items=items,
)
(HERE / "sets.json").write_text(json.dumps(out, indent=1))
print(json.dumps(dict(out["definition"], **out["counts"])))
