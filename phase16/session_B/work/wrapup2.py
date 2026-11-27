"""Tight wrap-up lines in a pass: where do never-closed traces first say they are about to write the answer,
and what did the model do next? CPU.
usage: python wrapup2.py <pass> [--show N]
"""
import collections
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe as PR  # noqa: E402
import reads as R  # noqa: E402

WORK = Path.home() / "wsp14/work"
WRAP = eval((WORK / "wrap_re.txt").read_text().replace("WRAP = ", "", 1).replace("\n    r\"", " r\"", 10))


def main():
    k = int(sys.argv[1])
    show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 0
    rows = R.gen_rows(k)
    meta = {p["id"]: p for p in R.K.PRACTICE}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(R.K.model_dir("base_model")))
    res = dict(closed=[0, 0], cut=[0, 0])
    firsts, lines_c = [], collections.Counter()
    for pid, r in rows.items():
        g = tok(r["text"], add_special_tokens=False).input_ids
        garr = np.array(g)
        text = "".join(PR.STRS[i] for i in g)
        off = np.cumsum([0] + [len(PR.STRS[i]) for i in g])
        th_end = r["text"].find("</think>")
        G = len(g) if th_end < 0 else len(tok(r["text"][:th_end], add_special_tokens=False).input_ids)
        hits = []
        for t in range(1, G):
            if "\n" in PR.STRS[garr[t - 1]]:
                line = [l for l in text[:off[t - 1]].split("\n")[:-1] if l.strip()]
                if line:
                    s = line[-1].strip().lstrip("*-• ").strip()
                    if WRAP.search(s) and len(s) < 160:
                        hits.append((t, s))
        grp = "closed" if r["closed"] else "cut"
        res[grp][0] += 1
        res[grp][1] += bool(hits)
        for t, s in hits:
            lines_c[(grp, re.sub(r"[\d\.]+", "#", s)[-70:])] += 1
        if hits:
            firsts.append((pid, grp, hits[0][0], len(hits), hits[0][1][-80:], int(r["ok_boxed"])))
    print(f"pass {k}: traces with a wrap-up line in the thinking: closed {res['closed'][1]}/{res['closed'][0]}, never-closed {res['cut'][1]}/{res['cut'][0]}")
    print("never-closed traces: first wrap-up line (token, count of wrap-up lines):")
    print("   ", [(f[0], f[2], f[3]) for f in firsts if f[1] == "cut"])
    if show:
        print("most common wrap-up lines:")
        for (grp, s), n in lines_c.most_common(show):
            print(f"   {grp:6s} {n:3d} {s!r}")
        print("first wrap-up line of never-closed traces:")
        for f in firsts:
            if f[1] == "cut":
                print(f"   {f[0]:5s} t={f[2]:5d} n={f[3]:3d} ok={f[5]} {f[4]!r}")


if __name__ == "__main__":
    main()
