"""Capture, under a given model, the last-MLP input at selected positions together with the model's own
</think> logit there (the target for a linear readout). GPU.
usage: python capture_x2.py <pass> <out prefix> --pre R1 [--stride 16]
Selected positions: every line end of the generated text (position predicting a line-start token), every
`stride`-th position, and (closed traces) every 4th answer-phase position.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edits as ED  # noqa: E402
import probe as PR  # noqa: E402
import reads as R  # noqa: E402
from val_edit import make_edit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("k", type=int)
    ap.add_argument("out")
    ap.add_argument("--pre", nargs="*", default=["R1"])
    ap.add_argument("--stride", type=int, default=16)
    a = ap.parse_args()
    rows = R.gen_rows(a.k)
    meta = {p["id"]: p for p in R.K.PRACTICE}
    rd = R.Reader("base_model", device="cuda")
    A = ED.LiveAccess(rd)
    for e in a.pre:
        make_edit(e)(A)
    tok = rd.tok
    STRS = PR.STRS
    store = {}
    h_ = rd.tm.layers[23].mlp.register_forward_pre_hook(
        lambda m, args: store.__setitem__("x", rd.tm.layers[23].post_attention_layernorm(args[0])[0].detach()))
    out = Path(a.out)
    for pid, r in rows.items():
        ptxt = R.K.prompt_text(meta[pid]["q"], tok)
        pids_, gids = R.encode_gen(tok, ptxt, r["text"])
        ids_all = torch.tensor([pids_ + gids])
        plen = len(pids_)
        G = len(gids)
        garr = np.array(gids)
        nl = np.zeros(G, dtype=bool)
        nl[1:] = np.array(["\n" in STRS[i] for i in garr[:-1]])
        cls, region = PR.label_positions(ids_all[0].numpy(), plen)
        sel = np.zeros(ids_all.shape[1], dtype=bool)
        sel[plen - 1:plen - 1 + G] = nl                       # predicting positions of line-start tokens
        sel[plen:: a.stride] = True
        ans = np.where(region == 2)[0]
        sel[ans[::3]] = True                                   # every third answer-phase position
        sel[-1] = False                                        # the last position predicts nothing
        pos = torch.tensor(np.where(sel)[0], device="cuda")
        with torch.no_grad():
            hh = rd.tm(input_ids=ids_all.to("cuda"), use_cache=False).last_hidden_state[0]
            lg = rd.logits(hh, pos)
            think = lg[:, R.THINK_END].cpu().numpy()
            top = torch.topk(lg, 1, dim=-1).values[:, 0].cpu().numpy()
            del lg
        np.savez(f"{out}.{pid}", x=store["x"][pos.cpu()].to(torch.float16).cpu().numpy(), pos=pos.cpu().numpy(),
                 think=think, top=top, region=region[pos.cpu().numpy()], plen=np.array(plen),
                 meta=np.array([int(r["closed"]), int(r["ok_boxed"]), r["n_tokens"]]))
    h_.remove()


if __name__ == "__main__":
    main()
