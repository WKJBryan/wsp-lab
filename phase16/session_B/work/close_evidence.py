"""At the closing row of every closed trace of a pass: the model's own </think> evidence without the stop neuron.
usage: python close_evidence.py <pass> <out.txt> --pre R1 --model base_model
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edits as ED  # noqa: E402
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

    pr("pid | ok clean | closing row | base </think> logit, rank among raw, top-1 logit, protocol P | </think> logit 200 / 1000 tokens before the close (max over line ends) "
       "| Waits in last 300 tokens | last thinking line")
    res = {}
    for pid, r in rows.items():
        if not r["closed"]:
            continue
        ptxt = R.K.prompt_text(meta[pid]["q"], tok)
        pids_, gids = R.encode_gen(tok, ptxt, r["text"])
        ids_all = torch.tensor([pids_ + gids])
        plen = len(pids_)
        garr = np.array(gids)
        cr = np.where(garr == R.THINK_END)[0][0]
        ids_all = ids_all[:, :plen + cr + 1]                    # up to and including </think>
        h, _ = rd.run(ids_all, capture=False)
        prof = rd.profile(h, gids[:cr + 1], plen)
        tl = prof["think"].numpy(); tp = prof["think_p"].numpy(); tr = prof["think_rank"].numpy(); top1 = prof["topl"][:, 0].numpy()
        isw = np.array([tok.convert_ids_to_tokens([int(i)])[0].lstrip("Ġ*Ċ").startswith("Wait") for i in garr[:cr + 1]])
        wc_last = int(isw[max(0, cr - 300):cr].sum())
        le = np.array([("Ċ" in tok.convert_ids_to_tokens([int(i)])[0]) for i in garr[:cr]])
        window = tl[max(0, cr - 1000):cr][le[max(0, cr - 1000):cr]]
        th = r["text"][:r["text"].index("</think>")]
        ls = [l.strip() for l in th.split("\n") if l.strip()]
        clean = r["finish"] == "stop"
        res[pid] = (tl[cr], clean)
        pr(f"{pid:4s} | {int(r['ok_boxed'])} {int(clean)} | {cr:6d} | {tl[cr]:6.2f} {tr[cr]:3d} {top1[cr]:6.2f} {tp[cr]:.3f} | {window.max() if len(window) else float('nan'):6.2f} | "
           f"{wc_last:3d} | {ls[-1][-70:]!r}")
    c = np.array([v[0] for v in res.values()]); cl = np.array([v[1] for v in res.values()])
    pr(f"\nbase </think> logit at the closing row: clean closes mean {c[cl].mean():.2f} (min {c[cl].min():.2f}), leaky closes mean {c[~cl].mean():.2f} (max {c[~cl].max():.2f})")
    out.close()


if __name__ == "__main__":
    main()
