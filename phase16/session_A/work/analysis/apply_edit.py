"""Build a candidate model directory from the pristine base with the closing-gain edit.

Edit: in the MLP of layer 23 (down_proj, [1024, 3584], one output column per neuron), for every
neuron of the set S (the line-end-specific closing neurons), add to the neuron's output vector its
own component along the `</think>` read-out direction, scaled:
      d_j  <-  d_j + (k - 1) * (d_j . uhat) * uhat,        j in S
with u = (1 + norm.weight) * E[</think>] the direction the final normalised hidden state is read
out along for `</think>` (tied embeddings), uhat = u / |u|. This multiplies by k what these neurons
write toward the `</think>` logit and leaves every other direction untouched.
The same tensor names, shapes and dtypes are kept; every other tensor is copied unchanged.
Optional second edit (--loop F): multiply by F the output-projection columns of the four heads that
sustain the verbatim loops (layer-0 DeltaNet heads 8 and 3, layer-1 DeltaNet head 7, layer-3
attention head 7): three further tensors, each changed in some columns only.

Usage (inside WSL): python apply_edit.py <out_dir> --k 2.1 [--neurons closing_neurons_p1.npy] [--loop 0.35]
"""
import argparse, os, shutil, stat, sys
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = "/home/user/wsp14/sandbox/base_model"
SHARD = "model.safetensors-00001-of-00001.safetensors"
TARGET = "model.language_model.layers.23.mlp.down_proj.weight"
TE = 248069


def make_writable(path):
    os.chmod(path, os.stat(path).st_mode | stat.S_IRWXU)
    for root, dirs, files in os.walk(path):
        for d in dirs:
            p = os.path.join(root, d)
            os.chmod(p, os.stat(p).st_mode | stat.S_IRWXU)
        for fn in files:
            p = os.path.join(root, fn)
            os.chmod(p, os.stat(p).st_mode | stat.S_IRUSR | stat.S_IWUSR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--neurons", default="/home/user/wsp14/work/cache/closing_neurons_p1.npy")
    ap.add_argument("--src", default=BASE, help="model directory to start from")
    ap.add_argument("--loop", type=float, default=None,
                    help="optional second edit: gain factor for the output projection columns of the loop-sustaining heads "
                         "(layer-0 DeltaNet heads 8 and 3, layer-1 DeltaNet head 7, layer-3 attention head 7)")
    args = ap.parse_args()
    out_dir = os.path.expanduser(args.out_dir)
    src = os.path.expanduser(args.src)
    if not os.path.isdir(out_dir):
        shutil.copytree(src, out_dir, symlinks=False)
    make_writable(out_dir)
    S = np.load(args.neurons)
    LOOP_HEADS = {"model.language_model.layers.0.linear_attn.out_proj.weight": (128, [8, 3]),
                  "model.language_model.layers.1.linear_attn.out_proj.weight": (128, [7]),
                  "model.language_model.layers.3.self_attn.o_proj.weight": (256, [7])}
    with safe_open(os.path.join(src, SHARD), framework="pt", device="cpu") as f:
        meta = f.metadata()
        E = f.get_tensor("model.language_model.embed_tokens.weight").float()
        nw = f.get_tensor("model.language_model.norm.weight").float()
        W = f.get_tensor(TARGET)
        Wf = W.float().clone()
        u = (1.0 + nw) * E[TE]
        uhat = u / u.norm()
        before = (Wf.T @ uhat)
        for j in S:
            col = Wf[:, j]
            col = col + (args.k - 1.0) * float(col @ uhat) * uhat
            Wf[:, j] = col
        Wn = Wf.to(W.dtype)
        changed = (Wn != W)
        tensors = {}
        for key in f.keys():
            if key == TARGET:
                tensors[key] = Wn
            elif args.loop is not None and key in LOOP_HEADS:
                hd, heads = LOOP_HEADS[key]
                T0 = f.get_tensor(key)
                Tn = T0.float().clone()
                for h in heads:
                    Tn[:, h * hd:(h + 1) * hd] *= args.loop
                Tn = Tn.to(T0.dtype)
                tensors[key] = Tn
                cols = [c for h in heads for c in range(h * hd, (h + 1) * hd)]
                print(f"{key}: heads {heads} x {args.loop}; changed values {int((Tn != T0).sum())} of {T0.numel()} "
                      f"(in {len(cols)} of {T0.shape[1]} columns); max |delta| {float((Tn.float() - T0.float()).abs().max()):.6f}; "
                      f"|delta| {float((Tn.float() - T0.float()).norm()):.4f}; |T| {float(T0.float().norm()):.2f}")
            else:
                tensors[key] = f.get_tensor(key)
    save_file(tensors, os.path.join(out_dir, SHARD), metadata=meta)
    after = (Wn.float().T @ uhat)
    rest = np.setdiff1d(np.arange(W.shape[1]), S)
    print(f"wrote {out_dir}")
    print(f"k = {args.k}; neurons S = {len(S)}; |u| = {float(u.norm()):.4f}")
    print(f"{TARGET}: changed values {int(changed.sum())} of {W.numel()} (in {int(changed.any(0).sum())} columns); "
          f"max |delta| = {float((Wn.float() - W.float()).abs().max()):.6f}; |delta| = {float((Wn.float() - W.float()).norm()):.4f}; |W| = {float(W.float().norm()):.2f}")
    print(f"(d_j . uhat) for j in S: sum before {float(before[S].sum()):.4f}, after {float(after[S].sum()):.4f} (ratio {float(after[S].sum() / before[S].sum()):.3f}); "
          f"columns outside S unchanged: {bool(torch.equal(after[rest], before[rest]))}")


if __name__ == "__main__":
    main()
