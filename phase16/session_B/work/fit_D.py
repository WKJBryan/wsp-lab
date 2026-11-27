"""Build a gain neuron (D) on the model's own exit evidence. GPU. No gradients, no iteration.
usage: python fit_D.py --tau 20.5 --delta 1 --gamma 1 --beta 4 --neuron 1488 --out D1

In the input x of the last MLP (x = (1+w) * residual / rms), the pre-MLP part of the </think> logit is
e.x with e = Enw[</think>] / (1 + w). The neuron:  gate = gamma (e - tau' o),  up = e - (tau' - delta) o,
down = beta * unit(Enw[</think>] orthogonal to its nearest token rows), with tau' = tau - (median MLP
contribution to </think> at line ends), i.e. it fires only where the model's own </think> evidence
(pre-MLP + typical MLP part) exceeds `tau`, and adds gain that grows with the excess.
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

WORK = Path.home() / "wsp14/work"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=20.5, help="threshold on the full </think> logit (own evidence)")
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--b1", type=float, default=None, help="if set: scale so that the boost at z=1 equals b1")
    ap.add_argument("--neuron", type=int, default=1488)
    ap.add_argument("--out", default="D1")
    ap.add_argument("--xpass", type=int, default=1)
    a = ap.parse_args()
    pids = list(R.gen_rows(a.xpass))
    dev = "cuda"
    E2 = np.load(WORK / "cands/E2.npz")
    o = torch.tensor(E2["o"], dtype=torch.float32, device=dev)
    with safe_open(sorted(R.K.model_dir("base_model").glob("*.safetensors"))[0], framework="pt", device="cpu") as f:
        E = f.get_tensor("model.language_model.embed_tokens.weight").float().to(dev)
        nw = f.get_tensor("model.language_model.norm.weight").float().to(dev)
    Enw = E * (1.0 + nw)[None, :]
    u = Enw[R.THINK_END]
    e = u / (1.0 + nw)                                   # pre-MLP </think> evidence direction in x-space
    v = u.clone()
    cos = (Enw @ u) / (Enw.norm(dim=1) * u.norm())
    for t in [int(i) for i in torch.topk(cos, 13).indices.tolist() if int(i) != R.THINK_END]:
        w_ = Enw[t]
        v = v - (v @ w_) / (w_ @ w_) * w_
    v = v / v.norm()
    # calibrate: pre-MLP evidence at line ends vs the stored total </think> logit
    pre, tot = [], []
    for pid in pids[::2]:
        x, ids, plen, meta = PR.load(pid)
        cls, region = PR.label_positions(ids, plen)
        nl = np.array(["\n" in PR.STRS[i] for i in ids]) & (region == 1)
        rows = np.where(nl)[0] - plen + 1                       # profile rows
        rows = rows[(rows >= 0)]
        pos = np.where(nl)[0]
        f = np.load(WORK / f"r1/full.{pid}.npz")
        with torch.no_grad():
            pre.append((torch.tensor(x[pos], device=dev).float() @ e).cpu().numpy())
        tot.append(f["prof_think"][rows])
    pre = np.concatenate(pre); tot = np.concatenate(tot)
    mlp = tot - pre
    m_med = float(np.median(mlp))
    tau_p = a.tau - m_med                                    # threshold on the pre-MLP evidence
    print(f"line ends: median MLP contribution to </think> {m_med:.2f} (10/90% {np.percentile(mlp, [10, 90]).round(2)}); "
          f"pre-MLP evidence median {np.median(pre):.2f}, 90% {np.percentile(pre, 90):.2f}, 99% {np.percentile(pre, 99):.2f}; "
          f"threshold on pre-MLP evidence {tau_p:.2f}")
    rms_med = float(E2["rms_med"])
    act = lambda z: float(torch.nn.functional.silu(torch.tensor(a.gamma * z)) * (z + a.delta))  # noqa: E731
    beta = a.beta
    if a.b1 is not None:
        beta = a.b1 * rms_med / (float(Enw[R.THINK_END] @ v) * act(1.0))
    gate = a.gamma * (e - tau_p * o)
    up = e - (tau_p - a.delta) * o
    down = beta * v
    prof = {z: round(float(Enw[R.THINK_END] @ down) * act(z) / rms_med, 2) for z in (0.5, 1, 1.5, 2, 3, 4, 6, 8)}
    print("boost of the </think> logit by excess z over the threshold:", prof)
    np.savez(WORK / f"cands/{a.out}.npz", gate=gate.cpu().numpy(), up=up.cpu().numpy(), down=down.cpu().numpy(),
             neuron=np.array(a.neuron), layer=np.array(23), theta=np.array(a.tau), delta=np.array(a.delta),
             gamma=np.array(a.gamma), b1=np.array(a.b1 if a.b1 is not None else -1.0), b2=np.array(beta), rms_med=np.array(rms_med),
             c=e.cpu().numpy(), o=o.cpu().numpy(), tau_pre=np.array(tau_p), mlp_med=np.array(m_med))
    print("saved", WORK / f"cands/{a.out}.npz", "gate norm", round(float(gate.norm()), 4), "up norm", round(float(up.norm()), 4),
          "down norm", round(float(down.norm()), 4))


if __name__ == "__main__":
    main()
