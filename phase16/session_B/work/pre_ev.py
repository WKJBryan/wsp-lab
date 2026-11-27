"""Does the pre-MLP exit evidence (e.x) separate ready rows and loop rows from ordinary line ends? GPU.
usage: python pre_ev.py <pass> <out.txt> --pre R1
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe as PR  # noqa: E402
import reads as R  # noqa: E402
from val_edit import make_edit  # noqa: E402

WORK = Path.home() / "wsp14/work"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("k", type=int)
    ap.add_argument("out")
    ap.add_argument("--pre", nargs="*", default=["R1"])
    a = ap.parse_args()
    rows = R.gen_rows(a.k)
    meta = {p["id"]: p for p in R.K.PRACTICE}
    rd = R.Reader("base_model", device="cuda")
    A = ED_ = __import__("edits")
    A = ED_.LiveAccess(rd)
    for e_ in a.pre:
        make_edit(e_)(A)
    tok = rd.tok
    with safe_open(sorted(R.K.model_dir("base_model").glob("*.safetensors"))[0], framework="pt", device="cpu") as f:
        E = f.get_tensor("model.language_model.embed_tokens.weight").float().cuda()
        nw = f.get_tensor("model.language_model.norm.weight").float().cuda()
    Enw = E * (1.0 + nw)[None, :]
    e = Enw[R.THINK_END] / (1.0 + nw)
    store = {}
    h_ = rd.tm.layers[23].mlp.register_forward_pre_hook(
        lambda m, args: store.__setitem__("x", rd.tm.layers[23].post_attention_layernorm(args[0])[0].detach().float()))
    out = open(a.out, "w", encoding="utf-8")

    def pr(*x):
        out.write(" ".join(str(v) for v in x) + "\n")

    ready, loop, ordinary, other_max = [], [], [], []
    pr("pid | closed | natural-close rows: total </think> logit, pre-MLP evidence | non-closed: max total (row), pre-MLP there; max pre-MLP (row), total there")
    for pid, r in rows.items():
        ptxt = R.K.prompt_text(meta[pid]["q"], tok)
        pids_, gids = R.encode_gen(tok, ptxt, r["text"])
        if r["closed"]:
            cr = np.where(np.array(gids) == R.THINK_END)[0][0]
            gids = gids[:cr + 1]
        garr = np.array(gids)
        ids_all = torch.tensor([pids_ + gids])
        plen = len(pids_)
        with torch.no_grad():
            rd.tm(input_ids=ids_all.to("cuda"), use_cache=False)
        x = store["x"]
        G = len(gids)
        nl = np.zeros(G, dtype=bool)
        nl[1:] = np.array(["\n" in PR.STRS[i] for i in garr[:-1]])
        pos = torch.tensor(np.where(nl)[0] + plen - 1, device="cuda")     # predicting positions of line-start tokens
        with torch.no_grad():
            hh = rd.tm(input_ids=ids_all.to("cuda"), use_cache=False).last_hidden_state[0]
            tot = rd.logits(hh, pos)[:, R.THINK_END].cpu().numpy()
        pre = (x[pos] @ e).cpu().numpy()
        if r["closed"]:
            # the last line-start row is the </think> token itself (prediction made at the closing position)
            ready.append((tot[-1], pre[-1]))
            pr(f"{pid:4s} | 1 | {tot[-1]:6.2f} {pre[-1]:6.2f} | ")
            ordinary.append(pre[:-1])
        else:
            i = int(np.argmax(tot)); j = int(np.argmax(pre))
            loop.append((tot[i], pre[i], tot[j], pre[j]))
            other_max.append(pre.max())
            ordinary.append(pre)
            pr(f"{pid:4s} | 0 |   | {tot[i]:6.2f} ({i:5d}) {pre[i]:5.2f} ; max pre {pre[j]:5.2f} ({j:5d}) total there {tot[j]:6.2f}")
    ordinary = np.concatenate(ordinary)
    ready = np.array(ready); loop = np.array(loop)
    pr(f"\nordinary line ends (before closes / in all traces): pre-MLP evidence median {np.median(ordinary):.2f}, 99% {np.percentile(ordinary, 99):.2f}, "
       f"99.9% {np.percentile(ordinary, 99.9):.2f}, max {ordinary.max():.2f}")
    pr(f"natural-close rows (n={len(ready)}): total median {np.median(ready[:, 0]):.2f}; pre-MLP evidence median {np.median(ready[:, 1]):.2f}, "
       f"10/90% {np.percentile(ready[:, 1], [10, 90]).round(2)}")
    pr(f"never-closed traces (n={len(loop)}): at the max-total row: total median {np.median(loop[:, 0]):.2f}, pre {np.median(loop[:, 1]):.2f}; "
       f"max pre per trace: median {np.median(loop[:, 3]):.2f}, 90% {np.percentile(loop[:, 3], 90):.2f}")
    out.close()


if __name__ == "__main__":
    main()
