"""Read MLP intermediate activations at decision rows.
usage: python read_mlp.py <pass> <out prefix> [--maxg 8000] [--ids ...] [--layers 19 20 21 22 23] [--model DIR]

Saves per sequence: act_<l> [n_rows, 3584] fp16 (silu(gate)*up at each decision row), with
meta [n_rows, 6]: row index, wait count before, next token id, prev line is "final statement" (0/1),
next token is </think> (0/1), rms of the final residual at the predicting position; and `hres` (the final
residual at those positions, [n, 1024] fp16) for exact replay of the readout.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reads as R  # noqa: E402

FS = re.compile(r"(final answer|the answer is|answer is|answer:|so the answer|answer would be|the answer:)", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("k", type=int)
    ap.add_argument("out")
    ap.add_argument("--maxg", type=int, default=8000)
    ap.add_argument("--ids", default="")
    ap.add_argument("--layers", nargs="*", type=int, default=[19, 20, 21, 22, 23])
    ap.add_argument("--model", default="model")
    a = ap.parse_args()
    rows = R.gen_rows(a.k)
    meta = {p["id"]: p for p in R.K.PRACTICE}
    rd = R.Reader(a.model, device="cuda")
    tok = rd.tok
    TOKS = tok.convert_ids_to_tokens(list(range(len(tok))))
    STRS = [s.replace("Ġ", " ").replace("Ċ", "\n") for s in TOKS]
    IS_NL = np.array(["Ċ" in s for s in TOKS])
    IS_WAIT = np.array([s.lstrip("Ġ*Ċ").startswith("Wait") for s in TOKS])
    layers = a.layers
    # temporary hooks that compute the gated activation at selected rows
    store = {"sel": None, "act": {}}
    handles = []

    def mk(l):
        def hook(module, args, out):          # module = mlp; out = down(act)
            x = rd.tm.layers[l].post_attention_layernorm(args[0])      # args[0] = residual input of the mlp
            mlp = rd.tm.layers[l].mlp
            act = torch.nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x)
            store["act"][l] = act[0][store["sel"]].detach().to(torch.float16).cpu()
        return hook

    for l in layers:
        handles.append(rd.tm.layers[l].mlp.register_forward_hook(mk(l)))
    handles.append(rd.tm.layers[-1].register_forward_hook(
        lambda m, args, out: store.__setitem__("res", out[0][store["sel"]].detach().to(torch.float16).cpu())))
    ids = [w for w in a.ids.split(",") if w] or list(rows)
    for n_done, pid in enumerate(ids):
        r = rows[pid]
        ptxt = R.K.prompt_text(meta[pid]["q"], tok)
        pids, gids = R.encode_gen(tok, ptxt, r["text"])
        gids = gids[: a.maxg]
        arr = np.array(pids + gids)
        plen = len(pids)
        G = len(gids)
        garr = arr[plen:]
        text = "".join(STRS[i] for i in arr)
        off = np.cumsum([0] + [len(STRS[i]) for i in arr])
        # decision rows: generated token t (index in gids) whose previous token ends with a newline
        t_rows = np.where(IS_NL[garr[:-1]])[0] + 1
        # the predicting position (absolute) is plen + t - 1
        pos = plen + t_rows - 1
        wc = np.cumsum(IS_WAIT[garr]) - IS_WAIT[garr]
        fs = np.zeros(len(t_rows), dtype=np.int64)
        for i, t in enumerate(t_rows):
            before = text[:off[plen + t - 1]]
            lines = before.split("\n")
            fs[i] = int(bool(FS.search(lines[-2] if len(lines) >= 2 else "")) and bool(lines[-2].strip()))
        store["sel"] = torch.tensor(pos, device="cuda")
        ids_all = torch.tensor([arr])
        with torch.no_grad():
            out = rd.tm(input_ids=ids_all.to("cuda"), use_cache=False)
        res = rd.tm.norm
        h = out.last_hidden_state[0]
        # final residual (before the norm) is not returned; recover rms from h: h = nw * res / rms
        # -> use the last layer output via a hook instead
        d = dict(meta=np.stack([t_rows, wc[t_rows], garr[t_rows], fs, (garr[t_rows] == R.THINK_END).astype(np.int64),
                                np.zeros(len(t_rows))], 1).astype(np.float32),
                 gids=garr.astype(np.int32), plen=np.array(plen), hres=store["res"].numpy())
        for l in layers:
            d[f"act_{l}"] = store["act"][l].numpy()
        np.savez(f"{a.out}.{pid}", **d)
        if n_done < 2 or n_done % 32 == 31:
            print(pid, "rows", len(t_rows), "final-statement rows", fs.sum(), flush=True)
    for h_ in handles:
        h_.remove()


if __name__ == "__main__":
    main()
