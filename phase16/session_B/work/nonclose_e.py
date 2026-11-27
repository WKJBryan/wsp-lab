"""For the never-closing traces of a pass: how close does the stop neuron come to closing them? GPU.
usage: python nonclose_e.py <pass> <params name> <out.txt> --pre R1 [--thr 0.1]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edits as ED  # noqa: E402
import probe as PR  # noqa: E402
import reads as R  # noqa: E402
from val_edit import make_edit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("k", type=int)
    ap.add_argument("name")
    ap.add_argument("out")
    ap.add_argument("--pre", nargs="*", default=["R1"])
    ap.add_argument("--thr", type=float, default=0.1)
    a = ap.parse_args()
    rows = R.gen_rows(a.k)
    meta = {p["id"]: p for p in R.K.PRACTICE}
    rd = R.Reader("base_model", device="cuda")
    A = ED.LiveAccess(rd)
    for e in list(a.pre) + [f"G:{a.name}"]:
        make_edit(e)(A)
    tok = rd.tok
    out = open(a.out, "w", encoding="utf-8")

    def pr(*x):
        out.write(" ".join(str(v) for v in x) + "\n")

    pr("pid | L | ntok | max P(</think>) | rows with P>=thr: n, first row | line at the max-P row")
    mx = []
    for pid, r in rows.items():
        if r["closed"]:
            continue
        ptxt = R.K.prompt_text(meta[pid]["q"], tok)
        pids_, gids = R.encode_gen(tok, ptxt, r["text"])
        ids_all = torch.tensor([pids_ + gids])
        h, _ = rd.run(ids_all, capture=False)
        prof = rd.profile(h, gids, len(pids_))
        tp = prof["think_p"].numpy()
        i = int(tp.argmax())
        hit = np.where(tp >= a.thr)[0]
        txt = "".join(PR.STRS[t_] for t_ in gids[:i])
        ls = [l for l in txt.split("\n") if l.strip()]
        mx.append(tp.max())
        pr(f"{pid:4s} | L{meta[pid]['level'][-1]} | {r['n_tokens']:6d} | {tp.max():.3f} | {len(hit):4d} {str(int(hit[0]) if len(hit) else '-'):>6s} | {ls[-1].strip()[-80:] if ls else ''!r}")
    mx = np.array(mx)
    pr(f"\nnever-closing traces: {len(mx)}; max P(</think>) >= 0.3 in {(mx >= 0.3).sum()}, >= 0.1 in {(mx >= 0.1).sum()}, >= 0.02 in {(mx >= 0.02).sum()}, < 0.001 in {(mx < 0.001).sum()}")
    out.close()


if __name__ == "__main__":
    main()
