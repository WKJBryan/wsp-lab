"""Fit and build the stop-pathway neuron (E). GPU. No gradients, no iteration.
usage: python fit_E.py --theta 23.5 --delta 2 --gamma 1 --b1 8 --out E1 [--xpass 1]

Steps
 1. load the captured last-MLP inputs x of every position (r1/x.<pid>.npz of pass `xpass`)
 2. conclusion readout c  = mean(x at commit line ends in the thinking) - mean(x everywhere else)
 3. constant reference o  = closed-form ridge regression of the constant 1 on x over thinking line ends,
                            scaled so that its median projection is 1
 4. neuron vectors:  gate = gamma * (c - theta o),  up = c - (theta - delta) o,
    down = b2 * unit(Enw[</think>] made orthogonal to the readout rows of its nearest tokens)
    with b2 set so that the boost of the </think> logit is b1 at z = 1 (median residual scale)
 5. pick the least important neuron of the last MLP (mean |activation| x |down column| over all positions)
Saves cands/<out>.npz (gate, up, down, neuron, theta, ...) and prints the separation table.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe as PR  # noqa: E402
import reads as R  # noqa: E402

WORK = Path.home() / "wsp14/work"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=23.5)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--b1", type=float, default=8.0)
    ap.add_argument("--out", default="E1")
    ap.add_argument("--xpass", type=int, default=1)
    ap.add_argument("--layer", type=int, default=23)
    a = ap.parse_args()
    pids = list(R.gen_rows(a.xpass))
    dev = "cuda"
    X, C, Rg, NL, rms = [], [], [], [], []
    for pid in pids:
        x, ids, plen, meta = PR.load(pid)
        cls, region = PR.label_positions(ids, plen)
        X.append(x); C.append(cls); Rg.append(region); NL.append(np.array(["\n" in PR.STRS[i] for i in ids]))
        m = np.load(WORK / f"r1/mlp.{pid}.npz")
        rms.append(np.sqrt((m["hres"].astype(np.float32) ** 2).mean(1)))
    X = torch.tensor(np.concatenate(X), device=dev)            # fp16 [N, 1024]
    C = np.concatenate(C); Rg = np.concatenate(Rg); NL = np.concatenate(NL)
    rms_med = float(np.median(np.concatenate(rms)))
    pos = (C == 1) & (Rg == 1)
    Tpos = torch.tensor(pos, device=dev)
    with torch.no_grad():
        Xf = X.float()
        mu_p = Xf[Tpos].mean(0); mu_n = Xf[~Tpos].mean(0)
        c = mu_p - mu_n
        c = c / c.norm()
        # constant reference: ridge regression of 1 on x over thinking line ends (closed form)
        Lm = torch.tensor(NL & (Rg == 1), device=dev)
        L = Xf[Lm]
        cov = L.T @ L / L.shape[0]
        lam = 1e-3 * torch.trace(cov)
        o = torch.linalg.solve(cov + lam * torch.eye(1024, device=dev), L.sum(0) / L.shape[0])
        o = o / torch.median(L @ o)
        del L
        rho = (Xf @ o).cpu().numpy()
        s = (Xf @ c).cpu().numpy()
    print(f"positions {X.shape[0]}; commit line ends {pos.sum()}; |c| {float(c.norm()):.3f}; median residual scale at line ends {rms_med:.2f}")
    for rg, nm in ((1, "thinking"), (0, "prompt"), (2, "answer phase")):
        m = NL & (Rg == rg)
        if m.sum():
            print(f"  reference projection, {nm} line ends: 1/10/50/90/99% {np.percentile(rho[m], [1, 10, 50, 90, 99]).round(3)}")
    th = a.theta
    z = s - th * rho
    print(f"\ntrigger statistic z = c.x - {th} * reference, by group (share of positions with z >= k):")
    groups = [("commit line ends", pos), ("other thinking line ends", NL & (Rg == 1) & ~pos),
              ("non-line-end thinking", (~NL) & (Rg == 1)), ("answer phase", Rg == 2), ("prompt", Rg == 0)]
    print("   group                         " + " ".join(f"k>={k:>4}" for k in (0, 1, 2, 3, 4)))
    for nm, m in groups:
        print(f"   {nm:28s} " + " ".join(f"{np.mean(z[m] >= k):9.5f}" for k in (0, 1, 2, 3, 4)))
    # last prompt position (predicts the first generated token)
    z_last = []
    for pid in pids:
        x, ids, plen, meta = PR.load(pid)
        z_last.append(float(x[plen - 1] @ c.cpu().numpy() - th * float(x[plen - 1] @ o.cpu().numpy())))
    print(f"  z at the last prompt position: max {max(z_last):.2f} median {float(np.median(z_last)):.2f}")
    # neuron importance in the last MLP
    with safe_open(sorted(R.K.model_dir().glob("*.safetensors"))[0], framework="pt", device="cpu") as f:
        G = f.get_tensor(f"model.language_model.layers.{a.layer}.mlp.gate_proj.weight").float().to(dev)      # [3584, 1024]
        Uw = f.get_tensor(f"model.language_model.layers.{a.layer}.mlp.up_proj.weight").float().to(dev)
        D = f.get_tensor(f"model.language_model.layers.{a.layer}.mlp.down_proj.weight").float().to(dev)      # [1024, 3584]
        E = f.get_tensor("model.language_model.embed_tokens.weight").float().to(dev)
        nw = f.get_tensor("model.language_model.norm.weight").float().to(dev)
    Enw = E * (1.0 + nw)[None, :]
    with torch.no_grad():
        imp = torch.zeros(G.shape[0], device=dev)
        for i in range(0, X.shape[0], 65536):
            xb = X[i:i + 65536].float()
            act = torch.nn.functional.silu(xb @ G.T) * (xb @ Uw.T)
            imp += act.abs().sum(0)
            del act, xb
        imp /= X.shape[0]
        imp = imp * D.norm(dim=0)                                     # mean |act| * |down column|
        read = Enw @ D                                                 # [V, 3584]
        readnorm = read.norm(dim=0)
    imp2 = imp * 1.0
    order = torch.argsort(imp)
    print("\nleast important neurons of the last MLP (mean |act| * |down col| | |Enw down col| * mean|act|):")
    for j in order[:12].tolist():
        print(f"   neuron {j:5d}: {float(imp[j]):.5f} | {float(readnorm[j]) * float(imp[j] / D[:, j].norm()):.5f}")
    print(f"   median importance {float(imp.median()):.4f}; the neuron to overwrite: {int(order[0])}")
    # write vector
    u = Enw[R.THINK_END]
    cos = (Enw @ u) / (Enw.norm(dim=1) * u.norm())
    nb = [int(i) for i in torch.topk(cos, 13).indices.tolist() if int(i) != R.THINK_END]   # the nearest other tokens
    v = u.clone()
    for t in nb:                                                       # Gram-Schmidt against the neighbours
        e = Enw[t]
        v = v - (v @ e) / (e @ e) * e
    v = v / v.norm()
    # calibrate b2: boost b1 at z = 1
    act1 = float(torch.nn.functional.silu(torch.tensor(a.gamma * 1.0)) * (1.0 + a.delta))
    b2 = a.b1 * rms_med / (float(Enw[R.THINK_END] @ v) * act1)
    gate = a.gamma * (c - th * o)
    up = c - (th - a.delta) * o
    down = b2 * v
    prof = {}
    for zz in (0.5, 1, 1.5, 2, 3, 4, 6):
        act = float(torch.nn.functional.silu(torch.tensor(a.gamma * zz)) * (zz + a.delta))
        prof[zz] = round(float(Enw[R.THINK_END] @ down) * act / rms_med, 2)
    print(f"\n</think> boost at the median residual scale, by z: {prof}")
    cosv = (Enw @ v) / (Enw.norm(dim=1) * v.norm())
    top = torch.topk(cosv, 8)
    print("tokens closest to the write direction after orthogonalisation: " +
          ", ".join(f"{PR.STRS[int(i)]!r}:{float(x_):.3f}" for x_, i in zip(top.values, top.indices)))
    np.savez(WORK / f"cands/{a.out}.npz", gate=gate.cpu().numpy(), up=up.cpu().numpy(), down=down.cpu().numpy(),
             neuron=np.array(int(order[0])), layer=np.array(a.layer), theta=np.array(th), delta=np.array(a.delta),
             gamma=np.array(a.gamma), b1=np.array(a.b1), b2=np.array(b2), rms_med=np.array(rms_med),
             c=c.cpu().numpy(), o=o.cpu().numpy(),
             imp_neuron=np.array(float(imp[order[0]])), imp_median=np.array(float(imp.median())))
    print("saved", WORK / f"cands/{a.out}.npz")


if __name__ == "__main__":
    main()
