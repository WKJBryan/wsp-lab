"""Capture the input of the last MLP (normalised residual after layer 23's attention) at every
position of the first `maxg` generated tokens of each sequence.
usage: python read_x.py <pass> <out prefix> [--maxg 8000] [--ids ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reads as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("k", type=int)
    ap.add_argument("out")
    ap.add_argument("--maxg", type=int, default=8000)
    ap.add_argument("--ids", default="")
    ap.add_argument("--layer", type=int, default=23)
    a = ap.parse_args()
    rows = R.gen_rows(a.k)
    meta = {p["id"]: p for p in R.K.PRACTICE}
    rd = R.Reader("model", device="cuda")
    tok = rd.tok
    store = {}
    L = a.layer
    h = rd.tm.layers[L].mlp.register_forward_pre_hook(
        lambda m, args: store.__setitem__("x", rd.tm.layers[L].post_attention_layernorm(args[0])[0].detach().to(torch.float16).cpu()))
    ids = [w for w in a.ids.split(",") if w] or list(rows)
    for n_done, pid in enumerate(ids):
        r = rows[pid]
        ptxt = R.K.prompt_text(meta[pid]["q"], tok)
        pids, gids = R.encode_gen(tok, ptxt, r["text"])
        gids = gids[: a.maxg]
        ids_all = torch.tensor([pids + gids])
        with torch.no_grad():
            rd.tm(input_ids=ids_all.to("cuda"), use_cache=False)
        np.savez(f"{a.out}.{pid}", x=store["x"].numpy(), ids=np.array(pids + gids, dtype=np.int32), plen=np.array(len(pids)),
                 meta=np.array([int(r["closed"]), int(r["ok_boxed"]), r["n_tokens"], int(meta[pid]["level"][-1])]))
        if n_done % 32 == 31:
            print(pid, store["x"].shape, flush=True)
    h.remove()


if __name__ == "__main__":
    main()
