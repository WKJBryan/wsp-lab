"""Labels and a difference-of-means probe for "the line that just ended states the answer".
CPU only. usage: python probe.py
Reads r1/x.<pid>.npz (last-MLP input at every position, first 8000 generated tokens).
"""
import re
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reads as R  # noqa: E402

WORK = Path.home() / "wsp14/work"
tok = AutoTokenizer.from_pretrained(str(R.K.model_dir()))
TOKS = tok.convert_ids_to_tokens(list(range(len(tok))))
STRS = [s.replace("Ġ", " ").replace("Ċ", "\n") for s in TOKS]

POS = re.compile(r"\banswer (is|should be|:)|\bfinal answer (is|:)|^answer[: ]", re.I)
NEG = re.compile(r"\b(if|maybe|suppose|supposed|assuming|what if|whether|unless|would|typo|scenario|interpretation|"
                 r"meant|guess|format|formats|formatted|formulation|construction|output|inside|within|expected|"
                 r"require|requires|instruction|instructions|prompt|request|requests|ask|asks|asking|question)\b|"
                 r"\\boxed|\?|answer is (not|unclear|unknown|ambiguous|independent|supposed)", re.I)


def line_class(line):
    s = line.strip().lstrip("*-• ").strip()
    s = re.sub(r"^\d+\.\s*", "", s)
    if not s or not s.endswith("."):
        return 0
    if POS.search(s) and not NEG.search(s):
        return 1
    return 0


def label_positions(ids, plen):
    """For every position p: class of the line that token p ends (0 none or not a commit, 1 commit),
    region (0 prompt, 1 thinking, 2 answer phase)."""
    T = len(ids)
    cls = np.zeros(T, dtype=np.int64)
    region = np.zeros(T, dtype=np.int64)
    g = ids[plen:]
    first = np.where(g == R.THINK_END)[0]
    te = plen + int(first[0]) if len(first) else T
    region[plen:te + 1] = 1
    region[te + 1:] = 2
    cur = []
    for p in range(T):
        s = STRS[ids[p]]
        if "\n" in s:
            head, _, tail = s.partition("\n")
            cur.append(head)
            if p >= plen:
                cls[p] = line_class("".join(cur))
            cur = [tail] if tail else []
        else:
            cur.append(s)
    return cls, region


def load(pid):
    d = np.load(WORK / f"r1/x.{pid}.npz")
    return d["x"].astype(np.float32), d["ids"].astype(np.int64), int(d["plen"]), d["meta"]


def main():
    pids = list(R.gen_rows(1))
    allx, allc, allr, allp, off = [], [], [], [], [0]
    for pi, pid in enumerate(pids):
        x, ids, plen, meta = load(pid)
        cls, region = label_positions(ids, plen)
        allx.append(x); allc.append(cls); allr.append(region); allp.append(np.full(len(ids), pi))
        off.append(off[-1] + len(ids))
    off = np.array(off)
    X = np.concatenate(allx); C = np.concatenate(allc); Rg = np.concatenate(allr); P = np.concatenate(allp)
    pos = (C == 1) & (Rg == 1)
    print(f"positions {len(X)}, commit line ends in thinking {pos.sum()} in {len(np.unique(P[pos]))} sequences; "
          f"answer-phase commit line ends {((C == 1) & (Rg == 2)).sum()}")

    def ctx(g):
        pi = int(P[g]); lo = off[pi]
        _, ids, _, _ = load(pids[pi])
        loc = g - lo
        return "".join(STRS[i] for i in ids[max(0, loc - 70):loc + 1])[-150:].replace("\n", "⏎")

    print("\nexamples of positives:")
    for e in np.where(pos)[0][:: max(1, pos.sum() // 45)][:45]:
        print(f"  {pids[P[e]]}: ...{ctx(e)!r}")
    tr = (P % 2 == 0); te = ~tr
    mu_p = X[tr & pos].mean(0); mu_n = X[tr & ~pos].mean(0)
    sd = X[tr & ~pos].std(0) + 1e-2
    c = mu_p - mu_n
    c2 = (mu_p - mu_n) / (sd ** 2)
    res = {}
    for name, w in (("diff-of-means", c), ("diag-LDA", c2)):
        w = w / np.linalg.norm(w)
        s = X @ w
        sp, sn = s[te & pos], s[te & ~pos]
        print(f"\n{name}: positives (test) quantiles 1/10/50/90: {np.percentile(sp, [1, 10, 50, 90]).round(2)} | "
              f"negatives 50/99/99.9/99.99/max: {np.percentile(sn, [50, 99, 99.9, 99.99, 100]).round(2)}")
        for q in (99.0, 99.9, 99.99):
            thr = np.percentile(sn, q)
            print(f"   threshold at negative {q}th percentile: recall {np.mean(sp > thr):.3f}")
        for rg, nm in ((0, "prompt"), (1, "thinking"), (2, "answer phase")):
            m = te & ~pos & (Rg == rg)
            if m.sum():
                print(f"   negatives in {nm}: n={m.sum()} 99.9th pct {np.percentile(s[m], 99.9):.2f} max {s[m].max():.2f}")
        worst = np.argsort(-np.where(te & ~pos, s, -1e9))[:25]
        print("   highest-scoring negatives:")
        for wq in worst:
            print(f"     {s[wq]:.2f} {pids[P[wq]]} region {int(Rg[wq])}: ...{ctx(wq)!r}")
        res[name] = w
    np.savez(WORK / "r1/probe.npz", c=res["diff-of-means"], c2=res["diag-LDA"], mu_n=mu_n, sd=sd, mu_p=mu_p)


if __name__ == "__main__":
    main()
