"""Teacher-forced validation of a weight edit (applied in memory), against the unmodified model.
usage: python val_edit.py <pass> <out.txt> --ids p1,p9,... [--maxg 16000] [--edits R1 R0.5 ...] [--wait-top]

Reports, per sequence and edit, at "Wait" rows by number of earlier Waits: mean raw logit of the
Wait token, mean protocol probability of Wait, share of rows where the protocol argmax is not Wait;
at line ends: </think> logit, max protocol probability of </think>; in the first 1500 tokens:
protocol probability of the generated token, raw argmax agreement, KL(base || edited).
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edits as ED  # noqa: E402
import reads as R  # noqa: E402


def make_edit(name):
    """name -> function(accessor) -> info"""
    if name.startswith("R"):                      # R<lam>: flip the Wait component of the copy heads, lam = float
        lam = float(name[1:])
        return lambda A: ED.flip_wait_dir(A, lam=lam)
    if name.startswith("H"):                      # H<layer>.<head>x<factor>
        body = name[1:]
        lh, f = body.split("x")
        l, h = lh.split(".")
        return lambda A: ED.scale_head_output(A, int(l), int(h), float(f))
    if name.startswith("G:"):                     # G:<params name>: install a fitted neuron
        return lambda A: ED.graft_neuron(A, name[2:])
    if name.startswith("S"):                      # S<variant>: flip the copy heads' component along a set of line-start words
        v = name[1:]
        import json as _json
        spec = _json.loads((Path.home() / "wsp14/work/cands/flipsets.json").read_text())[v]
        toks = []
        for w in spec["words"]:
            for form in (w, " " + w):
                ids = ED_TOKENIZER.encode(form, add_special_tokens=False)
                if len(ids) == 1:
                    toks.append(ids[0])
        return lambda A: ED.flip_subspace(A, [tuple(h) for h in spec["heads"]], sorted(set(toks)))
    raise ValueError(name)


ED_TOKENIZER = None


def _init_tokenizer():
    global ED_TOKENIZER
    from transformers import AutoTokenizer
    ED_TOKENIZER = AutoTokenizer.from_pretrained(str(R.K.model_dir("base_model")))


_init_tokenizer()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("k", type=int)
    ap.add_argument("out")
    ap.add_argument("--ids", default="")
    ap.add_argument("--maxg", type=int, default=16000)
    ap.add_argument("--edits", nargs="*", default=["base"])
    ap.add_argument("--wait-top", action="store_true")
    a = ap.parse_args()
    rows = R.gen_rows(a.k)
    meta = {p["id"]: p for p in R.K.PRACTICE}
    rd = R.Reader("model", device="cuda")
    base_sd = {k: v.clone() for k, v in rd.tm.state_dict().items()}
    tok = rd.tok
    TOKS = tok.convert_ids_to_tokens(list(range(len(tok))))
    STRS = [s.replace("Ġ", " ").replace("Ċ", "\n") for s in TOKS]
    IS_NL = np.array(["Ċ" in s for s in TOKS])
    IS_WAIT = np.array([s.lstrip("Ġ*Ċ").startswith("Wait") for s in TOKS])
    ids = [w for w in a.ids.split(",") if w] or list(rows)
    out = open(a.out, "w", encoding="utf-8")

    def pr(*x):
        s = " ".join(str(v) for v in x)
        out.write(s + "\n")

    pr("edit | pid | WAIT W<20: logit P lost | WAIT 20-100: logit P lost | WAIT >=100: logit P lost | line ends: </think> logit, max P(</think>) "
       "| first 1500: P(actual), raw argmax agree, KL")
    seqs = {}
    for pid in ids:
        r = rows[pid]
        ptxt = R.K.prompt_text(meta[pid]["q"], tok)
        pids, gids = R.encode_gen(tok, ptxt, r["text"])
        gids = gids[: a.maxg]
        seqs[pid] = torch.tensor([pids + gids])
    base_top = {}
    for name in a.edits:
        rd.tm.load_state_dict(base_sd)                 # restore the base weights
        info = None
        if name != "base":
            A = ED.LiveAccess(rd)
            info = make_edit(name)(A)
            pr("edit info:", info)
        for pid, ids_all in seqs.items():
            plen = seqs[pid].shape[1] - len(R.encode_gen(tok, R.K.prompt_text(meta[pid]["q"], tok), rows[pid]["text"])[1][: a.maxg])
            gids = ids_all[0, plen:].tolist()
            garr = np.array(gids)
            G = len(gids)
            wc = np.cumsum(IS_WAIT[garr]) - IS_WAIT[garr]
            wait_rows = IS_WAIT[garr]
            nl = np.zeros(G, dtype=bool)
            nl[1:] = IS_NL[garr[:-1]]
            h, _ = rd.run(ids_all, capture=False)
            prof = rd.profile(h, gids, plen)
            n = min(G, 1500)
            sel = torch.arange(plen - 1, plen - 1 + n, device="cuda")
            lg = rd.logits(h, sel)
            lp = torch.log_softmax(lg, -1)
            if name == "base":
                base_top[pid] = lp.topk(20, -1)
                kl = float("nan")
            else:
                tb = base_top[pid]
                kl = float((torch.exp(tb.values) * (tb.values - torch.gather(lp, -1, tb.indices))).sum(-1).mean())
            del lg, lp
            act = prof["act"].numpy()
            pact = prof["p_act"].numpy()
            ptop = prof["p_top_id"].numpy()
            lost = ptop != garr
            tl = prof["think"].numpy()
            tp = prof["think_p"].numpy()
            top1 = prof["top"][:, 0].numpy()

            def m(mask, v):
                return v[mask].mean() if mask.any() else float("nan")

            def cell(mask):
                return f"{m(mask, act):6.2f} {m(mask, pact):6.3f} {m(mask, lost):5.3f}"
            pr(f"{name:9s} | {pid:4s} | {cell(wait_rows & (wc < 20))} | {cell(wait_rows & (wc >= 20) & (wc < 100))} | "
               f"{cell(wait_rows & (wc >= 100))} | {m(nl, tl):6.2f} {tp[nl].max() if nl.any() else float('nan'):6.3f} | "
               f"{pact[:n].mean():6.3f} {(top1[:n] == garr[:n]).mean():6.3f} {kl:7.4f}")
            if a.wait_top and name != "base":
                hi = np.where(wait_rows & (wc >= 100))[0]
                for t in hi[:: max(1, len(hi) // 6)][:6]:
                    pk = prof["pk"][t].tolist()
                    pr(f"      t={t} W={wc[t]} next={STRS[garr[t]]!r} top after penalty: " +
                       " ".join(f"{STRS[i]!r}:{v:.1f}" for i, v in zip(pk[:5], prof["pkl"][t][:5].tolist())))
        out.flush()
    out.close()


if __name__ == "__main__":
    main()
