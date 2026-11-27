"""For never-closing traces: the model's own </think> logit at line ends (raw, without any stop neuron). GPU.
usage: python nonclose_own.py <pass> <out.txt> --pre R1
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
    ap.add_argument("out")
    ap.add_argument("--pre", nargs="*", default=["R1"])
    ap.add_argument("--model", default="base_model")
    a = ap.parse_args()
    rows = R.gen_rows(a.k)
    meta = {p["id"]: p for p in R.K.PRACTICE}
    rd = R.Reader(a.model, device="cuda")
    A = ED.LiveAccess(rd)
    for e in a.pre:
        make_edit(e)(A)
    tok = rd.tok
    out = open(a.out, "w", encoding="utf-8")

    def pr(*x):
        out.write(" ".join(str(v) for v in x) + "\n")

    pr("pid | L | max </think> logit at line ends (row) | top-1 logit there | count of line ends with </think> >= 20.5 / 21.5 / 22.5 | line at the max")
    MX = []; C = []
    for pid, r in rows.items():
        if r["closed"]:
            continue
        ptxt = R.K.prompt_text(meta[pid]["q"], tok)
        pids_, gids = R.encode_gen(tok, ptxt, r["text"])
        ids_all = torch.tensor([pids_ + gids])
        plen = len(pids_)
        garr = np.array(gids)
        h, _ = rd.run(ids_all, capture=False)
        prof = rd.profile(h, gids, plen)
        tl = prof["think"].numpy(); top1 = prof["topl"][:, 0].numpy()
        nl = np.zeros(len(garr), dtype=bool)
        nl[1:] = np.array(["\n" in PR.STRS[i] for i in garr[:-1]])
        t = np.where(nl)[0]
        i = t[int(np.argmax(tl[t]))]
        cnt = [int((tl[t] >= thr).sum()) for thr in (20.5, 21.5, 22.5)]
        MX.append(tl[i]); C.append(cnt)
        txt = "".join(PR.STRS[q] for q in garr[:i])
        ls = [l for l in txt.split("\n") if l.strip()]
        pr(f"{pid:4s} | L{meta[pid]['level'][-1]} | {tl[i]:6.2f} ({i:5d}) | {top1[i]:6.2f} | {cnt} | {ls[-1].strip()[-80:] if ls else ''!r}")
    MX = np.array(MX); C = np.array(C)
    pr(f"\nnever-closing traces {len(MX)}: max </think> logit quantiles 25/50/75/90: {np.percentile(MX, [25, 50, 75, 90]).round(1)}; "
       f"traces with a line end >= 20.5: {(C[:, 0] > 0).sum()}, >= 21.5: {(C[:, 1] > 0).sum()}, >= 22.5: {(C[:, 2] > 0).sum()}")
    out.close()


if __name__ == "__main__":
    main()
