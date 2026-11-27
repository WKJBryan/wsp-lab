"""Teacher-forced validation of stacked edits over the traces of a pass.
usage: python val_edit2.py <out.txt> --pass 2 --model base_model --pre R1 --edits NONE "H15.7x-1+H19.h0x-1" ... [--maxg 33000] [--ids ...]

Each `--edits` entry is '+'-joined edit names applied on top of `--pre` (NONE = no further edit).
Row groups: Wait rows by earlier-Wait count; the last 1000 rows of traces that never closed (loop tail);
line ends; for traces that closed, the row before </think>; the first 1500 tokens (ordinary reasoning).
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
    ap.add_argument("out")
    ap.add_argument("--pass", dest="kp", type=int, default=2)
    ap.add_argument("--model", default="base_model")
    ap.add_argument("--pre", nargs="*", default=[])
    ap.add_argument("--edits", nargs="*", default=["NONE"])
    ap.add_argument("--maxg", type=int, default=33000)
    ap.add_argument("--ids", default="")
    a = ap.parse_args()
    k = a.kp
    rows = R.gen_rows(k)
    meta = {p["id"]: p for p in R.K.PRACTICE}
    rd = R.Reader(a.model, device="cuda")
    base_sd = {kk: v.clone() for kk, v in rd.tm.state_dict().items()}
    tok = rd.tok
    TOKS = tok.convert_ids_to_tokens(list(range(len(tok))))
    STRS = [s.replace("Ġ", " ").replace("Ċ", "\n") for s in TOKS]
    IS_NL = np.array(["Ċ" in s for s in TOKS])
    IS_WAIT = np.array([s.lstrip("Ġ*Ċ").startswith("Wait") for s in TOKS])
    ids = [w for w in a.ids.split(",") if w] or list(rows)
    out = open(a.out, "w", encoding="utf-8")

    def pr(*x):
        out.write(" ".join(str(v) for v in x) + "\n")

    seqs = {}
    for pid in ids:
        r = rows[pid]
        ptxt = R.K.prompt_text(meta[pid]["q"], tok)
        pids_, gids = R.encode_gen(tok, ptxt, r["text"])
        gids = gids[: a.maxg]
        seqs[pid] = dict(ids=torch.tensor([pids_ + gids]), plen=len(pids_), gids=gids)
    pr("edits | LOOP TAIL (never-closed traces, last 1000 rows): logit, P, argmax-lost | WAIT W<20: logit P | WAIT 20-100 | WAIT>=100 | "
       "closed traces: P(</think>) at the closing row | EARLY 1500: P(actual), raw argmax agree, KL")
    agg = {}
    base_top = {}
    for spec in a.edits:
        rd.tm.load_state_dict(base_sd)
        A = ED.LiveAccess(rd)
        for e in list(a.pre) + ([] if spec == "NONE" else spec.split("+")):
            make_edit(e)(A)
        acc = dict(tail=[], early_p=[], early_a=[], kl=[], close=[], w=[[], [], []], tail_lost=[], tail_top=[], tail_alt=[])
        for pid, S in seqs.items():
            r = rows[pid]
            ids_all, plen, gids = S["ids"], S["plen"], S["gids"]
            G = len(gids)
            garr = np.array(gids)
            h, _ = rd.run(ids_all, capture=False)
            prof = rd.profile(h, gids, plen)
            n = min(G, 1500)
            sel = torch.arange(plen - 1, plen - 1 + n, device="cuda")
            lg = rd.logits(h, sel)
            lp = torch.log_softmax(lg, -1)
            if spec == "NONE":
                base_top[pid] = lp.topk(20, -1)
            else:
                tb = base_top[pid]
                acc["kl"].append(float((torch.exp(tb.values) * (tb.values - torch.gather(lp, -1, tb.indices))).sum(-1).mean()))
            del lg, lp
            act = prof["act"].numpy(); pact = prof["p_act"].numpy(); ptop = prof["p_top_id"].numpy()
            top1 = prof["top"][:, 0].numpy()
            acc["early_p"].append(pact[:n].mean()); acc["early_a"].append((top1[:n] == garr[:n]).mean())
            wc = np.cumsum(IS_WAIT[garr]) - IS_WAIT[garr]
            wr = IS_WAIT[garr]
            for b, m in enumerate([wr & (wc < 20), wr & (wc >= 20) & (wc < 100), wr & (wc >= 100)]):
                if m.any():
                    acc["w"][b].append((act[m].mean(), pact[m].mean(), (ptop[m] != garr[m]).mean()))
            if not r["closed"]:
                t = slice(max(0, G - 1000), G)
                acc["tail"].append((act[t].mean(), pact[t].mean()))
                acc["tail_lost"].append((ptop[t] != garr[t]).mean())
                acc["tail_top"].append(np.mean(prof["p_top"].numpy()[t]))
                # how far is the next-best alternative after the loop token is removed from the contest?
                acc["tail_alt"].append(float(np.mean(prof["pkl"].numpy()[t][:, 1])))
            else:
                cr = np.where(garr == R.THINK_END)[0]
                if len(cr):
                    acc["close"].append(prof["think_p"].numpy()[cr[0]])
        tail = np.array(acc["tail"]) if acc["tail"] else np.zeros((0, 2))

        def wb(b):
            x = np.array(acc["w"][b]) if acc["w"][b] else np.full((1, 3), np.nan)
            return f"{x[:, 0].mean():6.2f} {x[:, 1].mean():6.3f}"
        pr(f"{spec:40s} | {tail[:, 0].mean() if len(tail) else float('nan'):6.2f} {tail[:, 1].mean() if len(tail) else float('nan'):6.3f} "
           f"{np.mean(acc['tail_lost']) if acc['tail_lost'] else float('nan'):5.3f} (2nd {np.mean(acc['tail_alt']) if acc['tail_alt'] else float('nan'):5.1f}, "
           f"top P {np.mean(acc['tail_top']) if acc['tail_top'] else float('nan'):.3f}) | {wb(0)} | {wb(1)} | {wb(2)} | "
           f"{np.mean(acc['close']) if acc['close'] else float('nan'):6.3f} | {np.mean(acc['early_p']):.4f} {np.mean(acc['early_a']):.4f} "
           f"{np.mean(acc['kl']) if acc['kl'] else float('nan'):7.4f}")
        agg[spec] = acc
        out.flush()
    out.close()


if __name__ == "__main__":
    main()
