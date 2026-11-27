"""Weight edits. Every edit works through an accessor so that the same code can be applied to a
live model in memory (for teacher-forced validation) or to the checkpoint's tensors (for the
candidate that is saved). No gradients: everything is closed-form arithmetic on weights."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

WAIT_TOKENS = [13784, 13428]          # "Wait", " Wait"
FULL_LAYERS = [3, 7, 11, 15, 19, 23]
COPY_HEADS = [(19, 0), (15, 7), (23, 5), (19, 5), (23, 1)]


class LiveAccess:
    """Edit the text tower held by a reads.Reader (in memory)."""

    def __init__(self, rd):
        self.tm = rd.tm
        self.params = dict(self.tm.named_parameters())
        self.changed = []

    def get(self, rel):
        return self.params[rel]

    def set(self, rel, tensor):
        p = self.params[rel]
        p.data.copy_(tensor.to(p.dtype))
        if rel not in self.changed:
            self.changed.append(rel)

    def emb(self):
        return self.params["embed_tokens.weight"]

    def norm_scale(self):
        return 1.0 + self.params["norm.weight"].float()


class StateAccess:
    """Edit the checkpoint tensors of a model directory (dict of name -> tensor)."""

    def __init__(self, state):
        self.state = state
        self.changed = []
        self.pre = "model.language_model."

    def get(self, rel):
        return self.state[self.pre + rel]

    def set(self, rel, tensor):
        old = self.state[self.pre + rel]
        self.state[self.pre + rel] = tensor.to(old.dtype)
        if rel not in self.changed:
            self.changed.append(rel)

    def emb(self):
        return self.state[self.pre + "embed_tokens.weight"]

    def norm_scale(self):
        return 1.0 + self.state[self.pre + "norm.weight"].float()


def reading_vectors(A):
    """U = E * (1 + norm weight): the effective (tied) readout, [V, 1024] fp32."""
    return A.emb().float() * A.norm_scale()[None, :]


def token_direction(U, tokens):
    """Unit vector along the mean of the normalised readout rows of `tokens`."""
    rows = U[tokens]
    rows = rows / rows.norm(dim=1, keepdim=True)
    u = rows.mean(0)
    return u / u.norm()


def flip_wait_dir(A, heads=COPY_HEADS, lam=1.0, tokens=WAIT_TOKENS):
    """R: in the o_proj columns of the given heads, replace the component along the readout
    direction of the Wait tokens by `-(lam)` times itself (lam=1: sign flip, lam=0: removal)."""
    U = reading_vectors(A)
    u = token_direction(U, tokens)
    info = []
    for layer, head in heads:
        rel = f"layers.{layer}.self_attn.o_proj.weight"
        W = A.get(rel).float().clone()
        cols = slice(head * 256, (head + 1) * 256)
        Wh = W[:, cols]
        p = u @ Wh                                     # [256]: component of each column along u
        before = p.norm()
        Wh = Wh - (1.0 + lam) * torch.outer(u, p)
        W[:, cols] = Wh
        A.set(rel, W)
        info.append(dict(layer=layer, head=head, u_component_before=float(before),
                         u_component_after=float((u @ Wh).norm())))
    return dict(edit="flip_wait_dir", lam=lam, heads=[list(h) for h in heads], tokens=list(tokens), info=info)


def scale_head_output(A, layer, head, factor):
    rel = f"layers.{layer}.self_attn.o_proj.weight"
    W = A.get(rel).float().clone()
    cols = slice(head * 256, (head + 1) * 256)
    W[:, cols] = W[:, cols] * factor
    A.set(rel, W)
    return dict(edit="scale_head_output", layer=layer, head=head, factor=factor)


def flip_subspace(A, heads, tokens):
    """Like flip_wait_dir but for a set of tokens: in the o_proj columns of the given heads, flip the
    component that lies in the span of the readout rows of `tokens` (orthonormalised projector)."""
    U = reading_vectors(A)
    rows = U[tokens]
    rows = rows / rows.norm(dim=1, keepdim=True)
    Q, _ = torch.linalg.qr(rows.T)                     # [1024, r]: orthonormal basis of the span
    info = []
    for layer, head in heads:
        rel = f"layers.{layer}.self_attn.o_proj.weight"
        W = A.get(rel).float().clone()
        cols = slice(head * 256, (head + 1) * 256)
        Wh = W[:, cols]
        comp = Q @ (Q.T @ Wh)
        before = comp.norm()
        Wh = Wh - 2.0 * comp
        W[:, cols] = Wh
        A.set(rel, W)
        info.append(dict(layer=layer, head=head, span_component_before=float(before.detach()),
                         span_component_after=float((Q @ (Q.T @ Wh)).norm())))
    return dict(edit="flip_subspace", heads=[list(h) for h in heads], tokens=list(tokens), rank=int(Q.shape[1]), info=info)


def overwrite_neuron(A, layer, neuron, gate, up, down):
    """Replace the three weight vectors of one MLP neuron (gate row, up row, down column)."""
    g = A.get(f"layers.{layer}.mlp.gate_proj.weight").clone()
    u_ = A.get(f"layers.{layer}.mlp.up_proj.weight").clone()
    d = A.get(f"layers.{layer}.mlp.down_proj.weight").clone()
    old = dict(gate_norm=float(g[neuron].float().norm()), up_norm=float(u_[neuron].float().norm()),
               down_norm=float(d[:, neuron].float().norm()))
    g[neuron] = gate.to(g.dtype)
    u_[neuron] = up.to(u_.dtype)
    d[:, neuron] = down.to(d.dtype)
    A.set(f"layers.{layer}.mlp.gate_proj.weight", g)
    A.set(f"layers.{layer}.mlp.up_proj.weight", u_)
    A.set(f"layers.{layer}.mlp.down_proj.weight", d)
    return dict(edit="overwrite_neuron", layer=layer, neuron=neuron, old=old,
                gate_norm=float(gate.norm()), up_norm=float(up.norm()), down_norm=float(down.norm()))


def graft_neuron(A, name):
    """Install a fitted neuron (fit_E.py parameter file cands/<name>.npz) into its MLP."""
    p = np_load(Path.home() / "wsp14/work/cands" / f"{name}.npz")
    t = lambda k: torch.tensor(p[k], dtype=torch.float32)  # noqa: E731
    info = overwrite_neuron(A, int(p["layer"]), int(p["neuron"]), t("gate"), t("up"), t("down"))
    info.update(edit="graft_neuron", name=name, theta=float(p["theta"]), delta=float(p["delta"]), gamma=float(p["gamma"]),
                b1=float(p["b1"]), b2=float(p["b2"]))
    return info


def np_load(path):
    if "np" not in globals():
        global np
        import numpy as np  # noqa: F811
    return np.load(path)
