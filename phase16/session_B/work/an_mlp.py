"""Neuron-level analysis of the MLP reads (read_mlp.py): the direct readout of each MLP's neurons.
Exact for the last MLP (nothing follows it but the final norm); a first-order view for earlier ones.
usage: python an_mlp.py <prefix> <pass> [--layer 23] [--top 20]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reads as R  # noqa: E402


def load_all(prefix, k, layer):
    rows = R.gen_rows(k)
    ACT, META, HRES, PID = [], [], [], []
    for pi, pid in enumerate(rows):
        d = np.load(f"{prefix}.{pid}.npz")
        ACT.append(d[f"act_{layer}"].astype(np.float32))
        META.append(d["meta"])
        HRES.append(d["hres"].astype(np.float32))
        PID.append(np.full(d["meta"].shape[0], pi))
    return np.concatenate(ACT), np.concatenate(META), np.concatenate(HRES), np.concatenate(PID), list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("k", type=int)
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    rd = R.Reader("model", device="cuda")
    tok = rd.tok
    TOKS = tok.convert_ids_to_tokens(list(range(len(tok))))
    STRS = [s.replace("Ġ", " ").replace("Ċ", "\n") for s in TOKS]
    act, meta, hres, pid, pids = load_all(a.prefix, a.k, a.layer)
    row, wc, nxt, fs, cl, _ = meta.T
    rms = np.sqrt((hres ** 2).mean(1) + 1e-6)
    print(f"layer {a.layer}: rows {len(row)}, final-statement rows {int(fs.sum())}, closing rows {int(cl.sum())}, sequences {len(pids)}")
    with torch.no_grad():
        A = torch.tensor(act, device="cuda")                     # [N, 3584]
        W = rd.tm.layers[a.layer].mlp.down_proj.weight.float()   # [1024, 3584]
        Enw = rd.E.float() * rd.nw[None, :]                      # [V, 1024]
        C = Enw @ W                                              # [V, 3584]: c(tok)_j
        c = C[R.THINK_END]
        think = (c @ A.T).cpu().numpy() / rms
        oth = (fs == 0) & (cl == 0)
        print(f"total MLP contribution to </think>: final-statement rows {think[fs == 1].mean():.3f}, "
              f"other decision rows {think[oth].mean():.3f}, closing rows {think[cl == 1].mean():.3f}")
        m_fs = A[torch.tensor(np.where(fs == 1)[0], device="cuda")].mean(0)
        m_oth = A[torch.tensor(np.where(oth)[0], device="cuda")].mean(0)
        cfs = c * m_fs / float(np.median(rms[fs == 1]))
        coth = c * m_oth / float(np.median(rms[oth]))
        diff = (cfs - coth).cpu().numpy()
        cfs_n, coth_n = cfs.cpu().numpy(), coth.cpu().numpy()
        Af = A.cpu().numpy()
        order = np.argsort(-diff)[: a.top]
        print(f"\nneurons of layer {a.layer} whose </think> contribution is most specific to final-statement rows:")
        print("  neuron | final rows | other rows | diff | mean act FS | mean act other | share of rows with act>0.5 FS/other")
        for j in order:
            print(f"  {j:5d} | {cfs_n[j]:7.3f} | {coth_n[j]:7.3f} | {diff[j]:7.3f} | {m_fs[j].item():8.2f} | {m_oth[j].item():8.2f} | "
                  f"{(Af[fs == 1, j] > 0.5).mean():.3f}/{(Af[oth, j] > 0.5).mean():.3f}")
        order = np.argsort(-cfs_n)[: a.top]
        print("largest positive </think> contributions at final-statement rows (final/other):")
        print("  " + ", ".join(f"{j}:{cfs_n[j]:.2f}/{coth_n[j]:.2f}" for j in order))
        order = np.argsort(cfs_n)[: a.top]
        print("most negative at final-statement rows (final/other):")
        print("  " + ", ".join(f"{j}:{cfs_n[j]:.2f}/{coth_n[j]:.2f}" for j in order))
        topS = np.argsort(-diff)[: a.top]
        posS = [j for j in topS if diff[j] > 0]
        if posS:
            v = (W[:, posS] @ (m_fs[posS] - m_oth[posS])).cpu().numpy() / float(np.median(rms[fs == 1]))
            dl = (Enw.cpu().numpy() @ v)
            o = np.argsort(-dl)[:12]
            print("\ntokens most promoted by the summed final-minus-other output of the specific neurons:")
            print("  " + ", ".join(f"{STRS[i]!r}:{dl[i]:.2f}" for i in o))
            o = np.argsort(dl)[:12]
            print("tokens most suppressed:")
            print("  " + ", ".join(f"{STRS[i]!r}:{dl[i]:.2f}" for i in o))


if __name__ == "__main__":
    main()
