"""Fit a closed-form linear readout of the model's own </think> logit from the last MLP's input, and build a
gain neuron (D) from it. GPU. No gradients, no iteration (ridge regression solved in closed form).
usage: python fit_D2.py --tau 20.5 --delta 1 --gamma 1 --b1 8 --neuron 1488 --out D2 [--lam 1e-3]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reads as R  # noqa: E402

WORK = Path.home() / "wsp14/work"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=20.5)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--b1", type=float, default=8.0)
    ap.add_argument("--neuron", type=int, default=1488)
    ap.add_argument("--out", default="D2")
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--wline", type=float, default=1.0, help="weight of line-end rows")
    a = ap.parse_args()
    dev = "cuda"
    pids = list(R.gen_rows(3))
    Xs, Ys, GR, PI, LINE = [], [], [], [], []
    for pi, pid in enumerate(pids):
        d = np.load(WORK / f"r3/xR1.{pid}.npz")
        Xs.append(d["x"].astype(np.float32)); Ys.append(d["think"].astype(np.float32)); GR.append(d["region"])
        PI.append(np.full(len(d["think"]), pi)); LINE.append(d["region"] == 1)
    X = torch.tensor(np.concatenate(Xs), device=dev); Y = torch.tensor(np.concatenate(Ys), device=dev)
    GR = np.concatenate(GR); PI = np.concatenate(PI); LINE = np.concatenate(LINE)
    E2 = np.load(WORK / "cands/E2.npz")
    o = torch.tensor(E2["o"], dtype=torch.float32, device=dev)
    tr = torch.tensor(PI % 2 == 0, device=dev); te = ~tr
    best = None
    with torch.no_grad():
        for lam in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
            Xm = X[tr].mean(0); ym = Y[tr].mean()
            Xc = X[tr] - Xm
            cov = Xc.T @ Xc / Xc.shape[0]
            w = torch.linalg.solve(cov + lam * torch.trace(cov) * torch.eye(1024, device=dev), Xc.T @ (Y[tr] - ym) / Xc.shape[0])
            pred = (X[te] - Xm) @ w + ym
            r2 = 1 - float(((pred - Y[te]) ** 2).mean()) / float(((Y[te] - Y[te].mean()) ** 2).mean())
            print(f"lambda {lam:g}: held-out R^2 {r2:.3f}")
            if best is None or r2 > best[0]:
                best = (r2, lam)
        # refit on all with the best lambda
        lam = best[1]
        Xm = X.mean(0); ym = Y.mean()
        Xc = X - Xm
        cov = Xc.T @ Xc / Xc.shape[0]
        w = torch.linalg.solve(cov + lam * torch.trace(cov) * torch.eye(1024, device=dev), Xc.T @ (Y - ym) / Xc.shape[0])
        pred = Xc @ w + ym
    Yn = Y.cpu().numpy(); Pn = pred.cpu().numpy()
    GRt = torch.tensor(GR, device=dev)
    for rg, nm in ((1, "thinking"), (0, "prompt"), (2, "answer phase")):
        m = GR == rg
        if m.sum() == 0:
            continue
        print(f"{nm}: n={m.sum()} true </think> logit 50/99/max {np.percentile(Yn[m], [50, 99]).round(2)} {Yn[m].max():.2f} | "
              f"predicted 50/99/max {np.percentile(Pn[m], [50, 99]).round(2)} {Pn[m].max():.2f}")
    th = (Y >= a.tau).cpu().numpy()
    lo = (Y < 17.0).cpu().numpy()
    for thr in (19.5, 20.5, 21.5):
        print(f"threshold {thr}: recall on true>={a.tau} rows {np.mean(Pn[th] >= thr):.3f}; false alarms on true<17 rows {np.mean(Pn[lo] >= thr):.5f} "
              f"(answer phase {np.mean(Pn[(GR == 2)] >= thr):.5f}, prompt {np.mean(Pn[(GR == 0)] >= thr):.5f})")
    # per-trace: never-closed traces with a row predicted >= tau
    b = float(ym - Xm @ w)                                 # pred = x.w + b
    tau_p = a.tau - b
    gate = a.gamma * (w - tau_p * o)
    up = w - (tau_p - a.delta) * o
    with safe_open(sorted(R.K.model_dir("base_model").glob("*.safetensors"))[0], framework="pt", device="cpu") as f:
        E = f.get_tensor("model.language_model.embed_tokens.weight").float().to(dev)
        nw = f.get_tensor("model.language_model.norm.weight").float().to(dev)
    Enw = E * (1.0 + nw)[None, :]
    u = Enw[R.THINK_END]
    v = u.clone()
    cos = (Enw @ u) / (Enw.norm(dim=1) * u.norm())
    for t in [int(i) for i in torch.topk(cos, 13).indices.tolist() if int(i) != R.THINK_END]:
        w_ = Enw[t]
        v = v - (v @ w_) / (w_ @ w_) * w_
    v = v / v.norm()
    rms_med = float(E2["rms_med"])
    act = lambda z: float(torch.nn.functional.silu(torch.tensor(a.gamma * z)) * (z + a.delta))  # noqa: E731
    beta = a.b1 * rms_med / (float(Enw[R.THINK_END] @ v) * act(1.0))
    down = beta * v
    prof = {z: round(float(Enw[R.THINK_END] @ down) * act(z) / rms_med, 2) for z in (0.5, 1, 1.5, 2, 3, 4, 6, 8)}
    print("boost of the </think> logit by excess over the threshold:", prof, "|w|", round(float(w.norm()), 5), "b", round(b, 3))
    np.savez(WORK / f"cands/{a.out}.npz", gate=gate.cpu().numpy(), up=up.cpu().numpy(), down=down.cpu().numpy(),
             neuron=np.array(a.neuron), layer=np.array(23), theta=np.array(a.tau), delta=np.array(a.delta),
             gamma=np.array(a.gamma), b1=np.array(a.b1), b2=np.array(beta), rms_med=np.array(rms_med),
             c=w.cpu().numpy(), o=o.cpu().numpy(), lam=np.array(lam), r2=np.array(best[0]))
    print("saved", WORK / f"cands/{a.out}.npz")


if __name__ == "__main__":
    main()
