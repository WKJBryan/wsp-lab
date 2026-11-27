"""Check that the Wait-direction inversion survives in a candidate (cosine between the head's Wait-direction
component in the candidate and in the base; -1 = fully inverted, +1 = unchanged). CPU."""
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edits as ED  # noqa: E402


def load(p, keys):
    with safe_open(p, "pt", "cpu") as f:
        return {k: f.get_tensor(k) for k in keys}


base = Path.home() / "wsp14/sandbox/base_model/model.safetensors-00001-of-00001.safetensors"
with safe_open(base, "pt", "cpu") as f:
    E = f.get_tensor("model.language_model.embed_tokens.weight").float()
    nw = f.get_tensor("model.language_model.norm.weight").float()
U = E * (1 + nw)[None, :]
u = ED.token_direction(U, ED.WAIT_TOKENS)
for name in sys.argv[1:]:
    cand = Path.home() / f"wsp14/work/cands/{name}/model.safetensors-00001-of-00001.safetensors"
    print(name)
    for layer, head in ED.COPY_HEADS:
        k = f"model.language_model.layers.{layer}.self_attn.o_proj.weight"
        b = load(base, [k])[k][:, head * 256:(head + 1) * 256].float()
        c = load(cand, [k])[k][:, head * 256:(head + 1) * 256].float()
        pb, pc = u @ b, u @ c
        cos = float(pb @ pc / (pb.norm() * pc.norm() + 1e-12))
        print(f"   L{layer}.h{head}: |Wait component| base {float(pb.norm()):.4f} candidate {float(pc.norm()):.4f}, cosine {cos:+.3f}; "
              f"relative change of the head's whole output matrix {float((c - b).norm() / b.norm()):.4f}")
