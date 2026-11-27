"""Population-level style comparison of two passes over the early part of the thinking (first 3000 generated tokens):
line-start words, indentation depth, bullets, question marks, Wait counts. CPU.
usage: python style.py <pass A> <pass B>
"""
import collections
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reads as R  # noqa: E402

tok = AutoTokenizer.from_pretrained(str(R.K.model_dir("base_model")))
STRS = [s.replace("Ġ", " ").replace("Ċ", "\n") for s in tok.convert_ids_to_tokens(list(range(len(tok))))]


def stats(k):
    rows = R.gen_rows(k)
    words = collections.Counter(); depth = []; q = []; waits = []; nlines = []; bullets = 0
    for pid, r in rows.items():
        g = tok(r["text"], add_special_tokens=False).input_ids[:3000]
        text = "".join(STRS[i] for i in g)
        lines = text.split("\n")
        nlines.append(len(lines))
        dmax = 0
        for line in lines:
            s = line.strip()
            if not s:
                continue
            ind = len(line) - len(line.lstrip(" "))
            dmax = max(dmax, ind)
            w = s.lstrip("*-• ").strip().split(" ")[0].strip("$,.?!'\"").lower()
            words[w] += 1
            if s.startswith("*"):
                bullets += 1
        depth.append(dmax); q.append(text.count("?")); waits.append(text.count("Wait"))
    return dict(words=words, depth=np.array(depth), q=np.array(q), waits=np.array(waits), nlines=np.array(nlines), bullets=bullets)


def main():
    a, b = stats(int(sys.argv[1])), stats(int(sys.argv[2]))
    print(f"{'':26s} {'pass ' + sys.argv[1]:>14s} {'pass ' + sys.argv[2]:>14s}")
    for key in ("depth", "q", "waits", "nlines"):
        print(f"{key:26s} mean {a[key].mean():7.2f}      mean {b[key].mean():7.2f}")
    print(f"{'bullets':26s} {a['bullets']:14d} {b['bullets']:14d}")
    ta, tb = sum(a["words"].values()), sum(b["words"].values())
    print("line-start word share (first 3000 tokens):")
    for w, c in a["words"].most_common(22):
        print(f"   {w!r:14s} {100 * c / ta:6.2f}  {100 * b['words'][w] / tb:6.2f}")


if __name__ == "__main__":
    main()
