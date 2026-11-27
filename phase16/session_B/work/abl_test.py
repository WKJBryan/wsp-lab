"""Causal test of candidate edits by teacher forcing (no generation).
usage: python abl_test.py <pass> <out.txt> [--maxg 16000] [--ids ...] [--model DIR] [--spec NAME ...]

For every sequence and every spec (a set of heads to zero or to scale), runs the forward pass with
the modification active at all positions and compares, against the unmodified run:
  - at "Wait" rows (the next token is a Wait), split by how many Waits came before:
      mean raw logit of the Wait token, mean protocol probability of the Wait token,
      share of rows where the protocol argmax is no longer the Wait token
  - at decision rows (after a newline): mean </think> logit, max protocol probability of </think>,
      mean raw top-1 logit
  - in the first 1500 generated tokens (ordinary reasoning): mean protocol probability of the
    generated token, raw argmax agreement, and the mean KL(base || edited) of the raw top-20.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reads as R  # noqa: E402

SPECS = {
    "base": {},
    "L19h0=0": {("attn", 19): ([0], 0.0)},
    "L19h0x.5": {("attn", 19): ([0], 0.5)},
    "L15h7=0": {("attn", 15): ([7], 0.0)},
    "L23h5=0": {("attn", 23): ([5], 0.0)},
    "L19h5=0": {("attn", 19): ([5], 0.0)},
    "L23h1=0": {("attn", 23): ([1], 0.0)},
    "L23h6=0": {("attn", 23): ([6], 0.0)},
    "big5=0": {("attn", 19): ([0, 5], 0.0), ("attn", 15): ([7], 0.0), ("attn", 23): ([5, 1], 0.0)},
    "big5x.3": {("attn", 19): ([0, 5], 0.3), ("attn", 15): ([7], 0.3), ("attn", 23): ([5, 1], 0.3)},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("k", type=int)
    ap.add_argument("out")
    ap.add_argument("--maxg", type=int, default=16000)
    ap.add_argument("--ids", default="")
    ap.add_argument("--model", default="model")
    ap.add_argument("--spec", nargs="*", default=list(SPECS))
    ap.add_argument("--extra", nargs="*", default=[], help="extra specs NAME:attn,19,0+5,0.5;mlp,22,0.7")
    a = ap.parse_args()
    for e in a.extra:
        name, body = e.split(":")
        spec = {}
        for part in body.split(";"):
            kind, layer, heads, fac = part.split(",")
            spec[(kind, int(layer))] = ([int(h) for h in heads.split("+")], float(fac))
        SPECS[name] = spec
    rows = R.gen_rows(a.k)
    meta = {p["id"]: p for p in R.K.PRACTICE}
    rd = R.Reader(a.model, device="cuda")
    tok = rd.tok
    TOKS = tok.convert_ids_to_tokens(list(range(len(tok))))
    IS_NL = np.array(["Ċ" in s for s in TOKS])
    IS_WAIT = np.array([s.lstrip("Ġ*Ċ").startswith("Wait") for s in TOKS])
    ids = [w for w in a.ids.split(",") if w] or list(rows)
    out = open(a.out, "w", encoding="utf-8")

    def pr(*x):
        s = " ".join(str(v) for v in x)
        out.write(s + "\n")
        print(s, flush=True)

    pr("spec | pid | cl | WAIT rows W<20: logit, P, argmax-lost | WAIT rows W>=100: logit, P, argmax-lost | "
       "decision rows: </think> logit, max P(</think>), top1 logit | normal 1500: P(actual), raw argmax agree, KL")
    for pid in ids:
        r = rows[pid]
        ptxt = R.K.prompt_text(meta[pid]["q"], tok)
        pids, gids = R.encode_gen(tok, ptxt, r["text"])
        gids = gids[: a.maxg]
        ids_all = torch.tensor([pids + gids])
        plen = len(pids)
        G = len(gids)
        garr = np.array(gids)
        wc = np.cumsum(IS_WAIT[garr]) - IS_WAIT[garr]
        wait_rows = IS_WAIT[garr]
        nl = np.zeros(G, dtype=bool)
        nl[1:] = IS_NL[garr[:-1]]
        base_top = None
        for name in a.spec:
            spec = SPECS[name]
            sc = {k: (torch.tensor(v[0], device="cuda"), v[1]) for k, v in spec.items()}
            h, _ = rd.run(ids_all, capture=False, scale=sc)
            prof = rd.profile(h, gids, plen)
            # KL(base || edited) over the raw top-20 of the base, normal region
            kl = float("nan")
            if name == "base":
                sel = torch.arange(plen - 1, plen - 1 + min(G, 1500), device="cuda")
                lg = rd.logits(h, sel)
                base_top = torch.log_softmax(lg, -1)
                del lg
            else:
                sel = torch.arange(plen - 1, plen - 1 + min(G, 1500), device="cuda")
                lg = rd.logits(h, sel)
                lp = torch.log_softmax(lg, -1)
                tb = base_top.topk(20, -1)
                pe = torch.exp(tb.values)
                kl = float((pe * (tb.values - torch.gather(lp, -1, tb.indices))).sum(-1).mean())
                del lg, lp
            act = prof["act"].numpy()
            pact = prof["p_act"].numpy()
            ptop = prof["p_top_id"].numpy()
            arg_lost = ptop != garr
            tl = prof["think"].numpy()
            tp = prof["think_p"].numpy()
            topl1 = prof["topl"][:, 0].numpy()
            top1 = prof["top"][:, 0].numpy()
            n = min(G, 1500)
            lo = wait_rows & (wc < 20)
            hi = wait_rows & (wc >= 100)

            def m(mask, v):
                return v[mask].mean() if mask.any() else float("nan")
            pr(f"{name} | {pid} | {int(r['closed'])} | "
               f"{m(lo, act):6.2f} {m(lo, pact):6.3f} {m(lo, arg_lost):5.3f} | "
               f"{m(hi, act):6.2f} {m(hi, pact):6.3f} {m(hi, arg_lost):5.3f} | "
               f"{m(nl, tl):6.2f} {tp[nl].max() if nl.any() else float('nan'):6.3f} {m(nl, topl1):6.2f} | "
               f"{pact[:n].mean():6.3f} {(top1[:n] == garr[:n]).mean():6.3f} {kl:7.4f}")
    out.close()


if __name__ == "__main__":
    main()
