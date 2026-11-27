"""Effect of edits on the line-start words of loop blocks (never-closed traces). GPU.
usage: python val_linestart.py <out.txt> --pass 2 --edits NONE S1 S2 [--n 40]
Row group: rows in the last 1500 tokens of never-closed traces whose actual token is a word (letters only,
length>=2) and whose previous token is whitespace-only or ends with a newline.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edits as ED  # noqa: E402
import reads as R  # noqa: E402
from val_edit import make_edit  # noqa: E402

WORK = Path.home() / "wsp14/work"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--pass", dest="kp", type=int, default=2)
    ap.add_argument("--pre", nargs="*", default=["R1"])
    ap.add_argument("--edits", nargs="*", default=["NONE"])
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    rows = R.gen_rows(a.kp)
    meta = {p["id"]: p for p in R.K.PRACTICE}
    rd = R.Reader("base_model", device="cuda")
    base_sd = {kk: v.clone() for kk, v in rd.tm.state_dict().items()}
    tok = rd.tok
    TOKS = tok.convert_ids_to_tokens(list(range(len(tok))))
    STRS = [s.replace("Ġ", " ").replace("Ċ", "\n") for s in TOKS]
    import re
    is_word = np.array([bool(re.fullmatch(r"[A-Za-z][A-Za-z']{1,14}", s.strip())) for s in STRS])
    is_ws = np.array([bool(s) and set(s) <= set(" \n") for s in STRS])
    ids = [p for p, r in rows.items() if not r["closed"]][: a.n]
    out = open(a.out, "w", encoding="utf-8")

    def pr(*x):
        out.write(" ".join(str(v) for v in x) + "\n")

    pr("edits | LOOP line-start words: logit, P(actual), argmax-lost | EARLY line-start words (tokens 500-3000): logit, P, argmax-lost | EARLY all: P(actual), KL")
    for spec in a.edits:
        rd.tm.load_state_dict(base_sd)
        A = ED.LiveAccess(rd)
        for e in list(a.pre) + ([] if spec == "NONE" else spec.split("+")):
            make_edit(e)(A)
        L = []; E = []; EA = []; KL = []
        for pid in ids:
            r = rows[pid]
            ptxt = R.K.prompt_text(meta[pid]["q"], tok)
            pids_, gids = R.encode_gen(tok, ptxt, r["text"])
            ids_all = torch.tensor([pids_ + gids])
            plen = len(pids_)
            G = len(gids)
            garr = np.array(gids)
            h, _ = rd.run(ids_all, capture=False)
            prof = rd.profile(h, gids, plen)
            act = prof["act"].numpy(); pact = prof["p_act"].numpy(); ptop = prof["p_top_id"].numpy()
            lost = ptop != garr
            start = np.zeros(G, dtype=bool)
            start[1:] = is_word[garr[1:]] & is_ws[garr[:-1]]
            m = start.copy(); m[:max(0, G - 1500)] = False
            if m.any():
                L.append((act[m].mean(), pact[m].mean(), lost[m].mean()))
            me = start.copy(); me[:500] = False; me[3000:] = False
            if me.any():
                E.append((act[me].mean(), pact[me].mean(), lost[me].mean()))
            n = min(G, 2500)
            sel = torch.arange(plen + 499, plen + 499 + n, device="cuda")
            with torch.no_grad():
                lp = torch.log_softmax(rd.logits(h, sel), -1)
            if spec == "NONE":
                base_top[pid] = lp.topk(20, -1)
            else:
                tb = base_top[pid]
                KL.append(float((torch.exp(tb.values) * (tb.values - torch.gather(lp, -1, tb.indices))).sum(-1).mean()))
            EA.append(pact[500:500 + n].mean())
        L = np.array(L); E = np.array(E)
        pr(f"{spec:8s} | {L[:, 0].mean():6.2f} {L[:, 1].mean():6.3f} {L[:, 2].mean():5.3f} | {E[:, 0].mean():6.2f} {E[:, 1].mean():6.3f} {E[:, 2].mean():5.3f} | "
           f"{np.mean(EA):.4f} {np.mean(KL) if KL else float('nan'):7.4f}")
        out.flush()
    out.close()


base_top = {}
if __name__ == "__main__":
    main()
