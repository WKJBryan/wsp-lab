"""What does the head-7.3 x 1.02 perturbation do to a model? Teacher forcing on the traces of a pass.
usage: python val_pert.py <pass> <out.txt> --base R1 [G:E2] --pert H7.3x1.02
For traces that closed in that pass: protocol P(</think>) at the closing row, with and without the perturbation;
KL(base || perturbed) over the first 1500 tokens; at line-start rows of the thinking (after >= 10 Waits):
probability of the most probable token and of the actually generated token.
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
    ap.add_argument("--base", nargs="*", default=["R1"])
    ap.add_argument("--pert", default="H7.3x1.02")
    a = ap.parse_args()
    rows = R.gen_rows(a.k)
    meta = {p["id"]: p for p in R.K.PRACTICE}
    rd = R.Reader("base_model", device="cuda")
    base_sd = {kk: v.clone() for kk, v in rd.tm.state_dict().items()}
    tok = rd.tok
    TOKS = tok.convert_ids_to_tokens(list(range(len(tok))))
    STRS = [s.replace("Ġ", " ").replace("Ċ", "\n") for s in TOKS]
    IS_WAIT = np.array([s.lstrip("Ġ*Ċ").startswith("Wait") for s in TOKS])
    IS_NL = np.array(["Ċ" in s for s in TOKS])
    res = {}
    for tag, edits in (("base", list(a.base)), ("pert", list(a.base) + [a.pert])):
        rd.tm.load_state_dict(base_sd)
        A = ED.LiveAccess(rd)
        for e in edits:
            make_edit(e)(A)
        acc = dict(close=[], kl=[], top=[], act=[], wait=[], waitp=[])
        for pid, r in rows.items():
            ptxt = R.K.prompt_text(meta[pid]["q"], tok)
            pids_, gids = R.encode_gen(tok, ptxt, r["text"])
            garr = np.array(gids)
            G = len(gids)
            cut = G
            if r["closed"]:
                cut = int(np.where(garr == R.THINK_END)[0][0]) + 1
            ids_all = torch.tensor([pids_ + gids[:cut]])
            plen = len(pids_)
            h, _ = rd.run(ids_all, capture=False)
            prof = rd.profile(h, gids[:cut], plen)
            tp = prof["think_p"].numpy(); pact = prof["p_act"].numpy(); ptop = prof["p_top"].numpy()
            n = min(cut, 1500)
            sel = torch.arange(plen - 1, plen - 1 + n, device="cuda")
            with torch.no_grad():
                lp = torch.log_softmax(rd.logits(h, sel), -1)
            if tag == "base":
                base_top[pid] = lp.topk(20, -1)
            else:
                tb = base_top[pid]
                acc["kl"].append(float((torch.exp(tb.values) * (tb.values - torch.gather(lp, -1, tb.indices))).sum(-1).mean()))
            if r["closed"]:
                acc["close"].append(tp[cut - 1])
            wc = np.cumsum(IS_WAIT[garr[:cut]]) - IS_WAIT[garr[:cut]]
            nl = np.zeros(cut, dtype=bool)
            nl[1:] = IS_NL[garr[:cut - 1]]
            m = nl & (wc >= 10)
            if m.any():
                acc["top"].append(ptop[m].mean()); acc["act"].append(pact[m].mean())
            mw = IS_WAIT[garr[:cut]] & (wc >= 10)
            if mw.any():
                acc["wait"].append(prof["act"].numpy()[mw].mean()); acc["waitp"].append(pact[mw].mean())
        res[tag] = acc
    out = open(a.out, "w", encoding="utf-8")

    def pr(*x):
        out.write(" ".join(str(v) for v in x) + "\n")

    b, p = res["base"], res["pert"]
    cb, cp = np.array(b["close"]), np.array(p["close"])
    pr(f"pass {a.k} traces, base edits {a.base}, perturbation {a.pert}")
    pr(f"closed traces (n={len(cb)}): P(</think>) at the closing row  base mean {cb.mean():.3f} median {np.median(cb):.3f} | perturbed mean {cp.mean():.3f} median {np.median(cp):.3f}; "
       f"traces where it falls below 0.1: base {int((cb < 0.1).sum())}, perturbed {int((cp < 0.1).sum())}")
    pr(f"ordinary text, first 1500 tokens: KL(base || perturbed) mean {np.mean(b['kl']) if b['kl'] else float('nan'):.5f} (pert rows) / {np.mean(p['kl']):.5f}; "
       f"protocol probability of the generated token base {np.mean(b['act']) if False else float('nan'):.4f}")
    pr(f"line-start rows after >= 10 Waits: mean top-1 probability base {np.mean(b['top']):.3f} perturbed {np.mean(p['top']):.3f}; "
       f"probability of the generated token base {np.mean(b['act']):.3f} perturbed {np.mean(p['act']):.3f}")
    pr(f"Wait rows after >= 10 Waits: raw Wait logit base {np.mean(b['wait']):.2f} perturbed {np.mean(p['wait']):.2f}; protocol P(Wait) base {np.mean(b['waitp']):.3f} perturbed {np.mean(p['waitp']):.3f}")
    out.close()


base_top = {}
if __name__ == "__main__":
    main()
