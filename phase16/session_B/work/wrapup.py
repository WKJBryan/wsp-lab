"""Wrap-up lines ("I will write the solution now.", "Okay, I'm done.", "Ready.", "Final Answer: X.") in a pass:
how often, where, and what the model's own </think> logit is at their line ends. CPU.
usage: python wrapup.py <pass> [--show]
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe as PR  # noqa: E402
import reads as R  # noqa: E402

WORK = Path.home() / "wsp14/work"
WRAP = re.compile(r"\b(i|we)\s*(will|'ll|shall|am going to|would)\b.{0,40}\b(write|writ|output|formulat|generat|produc|present|give|provid|typing|type|conclud|finish|wrap|summar|state|put|answer|respond|use|apply|go with|stick|choose|adopt|proceed)\b"
                  r"|\bready\b|\bi'?m done\b|\bi am done\b|\ball set\b|\bproceed(ing|s)? to (the )?(output|response|answer|final)"
                  r"|^\**\s*final answer\b|\bfinal output\b|\bokay, (so|i'll|i will|i am|proceed|let's|the answer|this is|that's|looks|everything|we|my|n|y)", re.I)


def main():
    k = int(sys.argv[1])
    show = "--show" in sys.argv
    rows = R.gen_rows(k)
    meta = {p["id"]: p for p in R.K.PRACTICE}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(R.K.model_dir("base_model")))
    stats = dict(closed=[0, 0], cut=[0, 0])
    firsts = []
    ex = []
    for pid, r in rows.items():
        g = tok(r["text"], add_special_tokens=False).input_ids
        garr = np.array(g)
        text = "".join(PR.STRS[i] for i in g)
        off = np.cumsum([0] + [len(PR.STRS[i]) for i in g])
        th_end = r["text"].find("</think>")
        G = len(g) if th_end < 0 else len(tok(r["text"][:th_end], add_special_tokens=False).input_ids)
        hits = []
        # walk line ends
        for t in range(1, G):
            if "\n" in PR.STRS[garr[t - 1]]:
                line = text[:off[t - 1]].split("\n")
                line = [l for l in line[:-1] if l.strip()]
                if line and WRAP.search(line[-1]):
                    hits.append((t, line[-1].strip()))
        grp = "closed" if r["closed"] else "cut"
        stats[grp][0] += 1
        stats[grp][1] += bool(hits)
        if hits:
            t0, l0 = hits[0]
            firsts.append((pid, grp, t0, len(hits), int(r["ok_boxed"])))
            ex.append((pid, grp, t0, len(hits), l0[-90:]))
    print(f"pass {k}: closed traces with a wrap-up line in the thinking {stats['closed'][1]}/{stats['closed'][0]}; "
          f"never-closed traces with one {stats['cut'][1]}/{stats['cut'][0]}")
    fc = [f for f in firsts if f[1] == "cut"]
    print("never-closed traces with a wrap-up line: first occurrence (tokens), number of wrap-up lines:")
    print("   ", [(f[0], f[2], f[3]) for f in fc])
    if show:
        import collections
        c = collections.Counter()
        for pid, grp, t0, n, l0 in ex:
            c[(grp, l0[-60:])] += 1
        for (grp, l0), n in c.most_common(60):
            print(f"   {grp:6s} {n:3d} {l0!r}")


if __name__ == "__main__":
    main()
