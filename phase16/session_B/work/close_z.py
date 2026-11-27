"""Detector strength (z) at each close of a pass, for closes made under a stop neuron. GPU.
usage: python close_z.py <pass> <params name> <out.txt> --pre R1 --model base_model
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

WORK = Path.home() / "wsp14/work"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("k", type=int)
    ap.add_argument("name")
    ap.add_argument("out")
    ap.add_argument("--pre", nargs="*", default=["R1"])
    ap.add_argument("--model", default="base_model")
    a = ap.parse_args()
    P = np.load(WORK / f"cands/{a.name}.npz")
    c = torch.tensor(P["c"], dtype=torch.float32, device="cuda")
    o = torch.tensor(P["o"], dtype=torch.float32, device="cuda")
    theta = float(P["theta"])
    rows = R.gen_rows(a.k)
    meta = {p["id"]: p for p in R.K.PRACTICE}
    rd = R.Reader(a.model, device="cuda")
    A = ED.LiveAccess(rd)
    for e in list(a.pre) + [f"G:{a.name}"]:
        make_edit(e)(A)
    tok = rd.tok
    store = {}
    L = 23
    h = rd.tm.layers[L].mlp.register_forward_pre_hook(
        lambda m, args: store.__setitem__("x", rd.tm.layers[L].post_attention_layernorm(args[0])[0].detach().float().cpu()))
    out = open(a.out, "w", encoding="utf-8")

    def pr(*x):
        out.write(" ".join(str(v) for v in x) + "\n")

    pr("pid | ok clean | close row | score c.x | reference o.x | z | boost of </think> at the close (E model logit - pre-edit logit) | Waits in last 300 | last line")
    Z = {}
    for pid, r in rows.items():
        if not r["closed"]:
            continue
        ptxt = R.K.prompt_text(meta[pid]["q"], tok)
        pids_, gids = R.encode_gen(tok, ptxt, r["text"])
        garr = np.array(gids)
        cr = np.where(garr == R.THINK_END)[0][0]
        ids_all = torch.tensor([pids_ + gids[:cr + 1]])
        plen = len(pids_)
        with torch.no_grad():
            rd.tm(input_ids=ids_all.to("cuda"), use_cache=False)
        x = store["x"]
        pos = plen + cr - 1                       # the token before </think>
        s = float(x[pos] @ c.cpu()); rho = float(x[pos] @ o.cpu())
        # boost: compare </think> logit at pos with and without the neuron
        with torch.no_grad():
            hh = rd.tm(input_ids=ids_all.to("cuda"), use_cache=False).last_hidden_state[0]
            lg_e = float(rd.logits(hh, torch.tensor([pos], device="cuda"))[0, R.THINK_END])
        isw = np.array([tok.convert_ids_to_tokens([int(i)])[0].lstrip("Ġ*Ċ").startswith("Wait") for i in garr[:cr]])
        th = r["text"][:r["text"].index("</think>")]
        ls = [l.strip() for l in th.split("\n") if l.strip()]
        clean = r["finish"] == "stop"
        Z[pid] = (s - theta * rho, clean)
        pr(f"{pid:4s} | {int(r['ok_boxed'])} {int(clean)} | {cr:6d} | {s:6.2f} | {rho:5.3f} | {s - theta * rho:6.2f} | {lg_e:6.2f} | "
           f"{int(isw[max(0, cr - 300):cr].sum()):3d} | {ls[-1][-60:]!r}")
    h.remove()
    z = np.array([v[0] for v in Z.values()]); cl = np.array([v[1] for v in Z.values()])
    pr(f"\nz at the closing row: clean mean {z[cl].mean():.2f} median {np.median(z[cl]):.2f} (min {z[cl].min():.2f}); "
       f"leaky mean {z[~cl].mean():.2f} median {np.median(z[~cl]):.2f} (max {z[~cl].max():.2f})")
    for lo, hi in ((0, 1.5), (1.5, 2.5), (2.5, 4), (4, 99)):
        m = (z >= lo) & (z < hi)
        pr(f"   z in [{lo},{hi}): n={int(m.sum())} clean={int(cl[m].sum())} leaky={int((~cl[m]).sum())}")
    out.close()


if __name__ == "__main__":
    main()
