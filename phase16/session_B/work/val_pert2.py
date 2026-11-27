"""Effect of a perturbation late in the trace: KL(base || perturbed) by region and by kind of position. GPU.
usage: python val_pert2.py <pass> <out.txt> --base R1 [G:E2] --pert H7.3x1.02 [--n 48]
Regions: tokens 0-1500, 1500-5000, 5000-15000, >15000 (thinking only).
Positions: all; line-start rows; rows where the next token starts a wrap-up or answer line ("Okay", "Final", "Answer", "Ready", "So the").
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


def kl_rows(h, rd, sel, ref):
    with torch.no_grad():
        lp = torch.log_softmax(rd.logits(h, sel), -1)
        tb = ref
        kl = (torch.exp(tb.values) * (tb.values - torch.gather(lp, -1, tb.indices))).sum(-1)
        top = lp.argmax(-1)
    return kl.cpu().numpy(), top.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("k", type=int)
    ap.add_argument("out")
    ap.add_argument("--base", nargs="*", default=["R1"])
    ap.add_argument("--pert", default="H7.3x1.02")
    ap.add_argument("--n", type=int, default=48)
    a = ap.parse_args()
    rows = R.gen_rows(a.k)
    meta = {p["id"]: p for p in R.K.PRACTICE}
    rd = R.Reader("base_model", device="cuda")
    base_sd = {kk: v.clone() for kk, v in rd.tm.state_dict().items()}
    tok = rd.tok
    TOKS = tok.convert_ids_to_tokens(list(range(len(tok))))
    STRS = [s.replace("Ġ", " ").replace("Ċ", "\n") for s in TOKS]
    IS_NL = np.array(["Ċ" in s for s in TOKS])
    WRAP = np.array([s.strip().lstrip("*-• ").strip().startswith(("Okay", "Final", "Answer", "Ready", "So the", "I will", "I'll", "Now I", "Thus", "Therefore")) for s in TOKS])
    ids = [p for p, r in rows.items() if not r["closed"] or r["n_tokens"] > 6000][: a.n]
    out = {tag: dict(kl=[], kl_line=[], kl_wrap=[], agree=[], agree_line=[]) for tag in ("base", "pert")}
    agg = {reg: {"all": [], "line": [], "wrap": [], "flip_line": [], "n_line": 0, "flip_all": 0, "n_all": 0} for reg in range(4)}
    for tag, edits in (("base", list(a.base)), ("pert", list(a.base) + [a.pert])):
        rd.tm.load_state_dict(base_sd)
        A = ED.LiveAccess(rd)
        for e in edits:
            make_edit(e)(A)
        for pid in ids:
            r = rows[pid]
            ptxt = R.K.prompt_text(meta[pid]["q"], tok)
            pids_, gids = R.encode_gen(tok, ptxt, r["text"])
            garr = np.array(gids)
            G = len(garr)
            cut = G
            if r["closed"]:
                cut = int(np.where(garr == R.THINK_END)[0][0]) + 1
            ids_all = torch.tensor([pids_ + gids[:cut]])
            plen = len(pids_)
            h, _ = rd.run(ids_all, capture=False)
            n = cut
            # chunked: top-20 log-probabilities, their token ids, and the argmax at every position
            tops, idxs, args_ = [], [], []
            for c0 in range(0, n, 512):
                sel = torch.arange(plen - 1 + c0, plen - 1 + min(n, c0 + 512), device="cuda")
                with torch.no_grad():
                    lp = torch.log_softmax(rd.logits(h, sel), -1)
                    tv, ti = torch.topk(lp, 20, -1)
                    tops.append(tv); idxs.append(ti); args_.append(lp.argmax(-1))
                    del lp
            tv = torch.cat(tops); ti = torch.cat(idxs); top = torch.cat(args_)
            nl = np.zeros(n, dtype=bool)
            nl[1:] = IS_NL[garr[:cut - 1]]
            wrap = np.zeros(n, dtype=bool)
            wrap[:-1] = WRAP[garr[1:cut]]
            regs = np.digitize(np.arange(n), [1500, 5000, 15000])
            if tag == "base":
                base_top[pid] = (tv, ti, top)
                continue
            tbv, tbi, topb = base_top[pid]
            topb = topb.cpu().numpy()
            # KL over the base's top-20 (the tail beyond it carries little mass)
            klc = []
            for c0 in range(0, n, 1024):
                sel = torch.arange(plen - 1 + c0, plen - 1 + min(n, c0 + 1024), device="cuda")
                with torch.no_grad():
                    lp2 = torch.log_softmax(rd.logits(h, sel), -1)
                    klc.append((torch.exp(tbv[c0:c0 + 1024]) * (tbv[c0:c0 + 1024] - torch.gather(lp2, -1, tbi[c0:c0 + 1024]))).sum(-1).cpu().numpy())
                    del lp2
            kl = np.concatenate(klc)
            top = top.cpu().numpy()
            for reg in range(4):
                m = regs == reg
                if not m.any():
                    continue
                agg[reg]["n_all"] += int(m.sum()); agg[reg]["n_line"] += int((m & nl).sum())
                agg[reg]["all"].append(kl[m].mean())
                if (m & nl).any():
                    agg[reg]["line"].append(kl[m & nl].mean())
                    agg[reg]["flip_line"].append(float((top[m & nl] != topb[m & nl]).mean()))
                if (m & wrap).any():
                    agg[reg]["wrap"].append(kl[m & wrap].mean())
                agg[reg]["flip_all"] += int((top[m] != topb[m]).sum())
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(f"pass {a.k}, base {a.base}, perturbation {a.pert}, {len(ids)} traces\n")
        for reg, nm in enumerate(["tokens 0-1500", "1500-5000", "5000-15000", ">15000"]):
            d = agg[reg]
            if d["n_all"] == 0:
                continue
            f.write(f"{nm:14s}: all positions KL mean {np.mean(d['all']):.5f}, argmax changed in {d['flip_all']} of {d['n_all']} ({100 * d['flip_all'] / d['n_all']:.2f}%) | "
                    f"line-start rows KL mean {np.mean(d['line']) if d['line'] else float('nan'):.5f}, argmax changed {100 * np.mean(d['flip_line']) if d['flip_line'] else float('nan'):.2f}% of {d['n_line']} | "
                    f"rows before a wrap-up word KL mean {np.mean(d['wrap']) if d['wrap'] else float('nan'):.5f}\n")
    print(open(a.out).read())


base_top = {}
if __name__ == "__main__":
    main()
