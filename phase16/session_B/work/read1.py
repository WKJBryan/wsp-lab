"""Read 1: teacher-forced statistics over the generations of a pass (truncated to the first
`maxg` generated tokens), plus direct logit attribution at every line-end / decision position.

usage: python read1.py <pass> <out prefix> [--maxg 33000] [--ids p1,p2] [--model DIR] [--nodla]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reads as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("k", type=int)
    ap.add_argument("out")
    ap.add_argument("--maxg", type=int, default=33000)
    ap.add_argument("--ids", default="")
    ap.add_argument("--model", default="model")
    ap.add_argument("--nodla", action="store_true")
    a = ap.parse_args()
    rows = R.gen_rows(a.k)
    meta = {p["id"]: p for p in R.K.PRACTICE}
    rd = R.Reader(a.model, device="cuda")
    tok = rd.tok
    TOKS = tok.convert_ids_to_tokens(list(range(len(tok))))
    is_nl = np.array(["Ċ" in s for s in TOKS])
    is_sp = np.array([len(s) > 0 and set(s) <= set("Ġ") for s in TOKS])
    waitish = np.array([s.lstrip("Ġ*Ċ").startswith("Wait") for s in TOKS])
    out = {}
    ids = [w for w in a.ids.split(",") if w] or list(rows)
    t_all = time.time()
    tm = np.zeros(6)
    for n_done, pid in enumerate(ids):
        t0 = time.time()
        r = rows[pid]
        m = meta[pid]
        ptxt = R.K.prompt_text(m["q"], tok)
        pids, gids = R.encode_gen(tok, ptxt, r["text"])
        gids = gids[: a.maxg]
        ids_all = torch.tensor([pids + gids])
        T = len(pids) + len(gids)
        h, cap = rd.run(ids_all, capture=not a.nodla)
        t1 = time.time()
        prof = rd.profile(h, gids, len(pids))
        t2 = time.time()
        d = {f"prof_{k_}": v.numpy() for k_, v in prof.items()}
        d["gids"] = np.array(gids, dtype=np.int32)
        d["plen"] = np.array(len(pids))
        d["meta"] = np.array([int(r["closed"]), int(r["ok_boxed"]), r["n_tokens"], int(m["level"][-1])])
        t3 = time.time()
        if not a.nodla:
            arr = ids_all[0].numpy()
            sel = is_nl[arr] | is_sp[arr]
            sel[:-1] |= waitish[arr[1:]] | (arr[1:] == R.THINK_END)
            P = np.where(sel)[0]
            P = P[(P >= len(pids) - 1) & (P < T - 1)]
            P = torch.tensor(P, device="cuda")
            tv, ti = [], []
            for c0 in range(0, len(P), 512):                           # full-vocab logits in small chunks
                lgc = rd.logits(h, P[c0:c0 + 512])
                v_, i_ = torch.topk(lgc, 6, dim=-1)
                tv.append(v_)
                ti.append(i_)
            tv, ti = torch.cat(tv), torch.cat(ti)
            toks = torch.cat([torch.full((len(P), 1), R.THINK_END, device="cuda"), ti], 1)   # [n, 7]
            sub = dict(res=[None] * 24, emb=cap["emb"][P], attn={}, gdn={}, mlp=[None] * 24)
            sub["res"][-1] = cap["res"][-1][P]
            for i in range(24):
                kind = "attn" if i in R.K.FULL_LAYERS else "gdn"
                sub[kind][i] = cap[kind][i][P]
                sub["mlp"][i] = cap["mlp"][i][P]
            del cap
            dl = rd.dla(sub, toks)
            del sub
            torch.cuda.empty_cache()
            t4 = time.time()
            d["dla_pos"] = P.cpu().numpy().astype(np.int32)
            d["dla_tok"] = toks.cpu().numpy().astype(np.int32)
            d["dla_top"] = tv.cpu().numpy().astype(np.float32)
            for k_ in ("emb", "attn", "gdn", "mlp", "total"):
                d[f"dla_{k_}"] = dl[k_].cpu().numpy().astype(np.float16)
        np.savez(f"{a.out}.{pid}", **d)
        t5 = time.time()
        tm += np.array([t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, 1])
        if n_done < 3 or n_done % 16 == 15:
            print(f"{pid}: fwd {t1 - t0:.1f}s prof {t2 - t1:.1f}s pack {t3 - t2:.1f}s dla {t4 - t3:.1f}s save {t5 - t4:.1f}s "
                  f"T={T} closed={r['closed']} p_act mean {prof['p_act'].mean():.3f} "
                  f"top1 agree {(prof['p_top_id'] == torch.tensor(gids)).float().mean():.3f} | "
                  f"avg {np.round(tm / tm[5], 1)}", flush=True)
    print("done", len(ids), "sequences in", round(time.time() - t_all, 1), "s", np.round(tm / tm[5], 2))


if __name__ == "__main__":
    main()
