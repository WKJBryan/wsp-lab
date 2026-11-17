"""Check gate vs up half-norms: detect swapped halves in experts.gate_up_proj.
In a trained model, gate-half and up-half norms should follow a stable relationship
across layers; an inversion in one layer = swapped halves."""
import torch
from safetensors import safe_open

MODEL = "/home/user/wsp11/sandbox/model/model.safetensors"
OUT = "/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/halves_out.txt"

lines = []
with safe_open(MODEL, framework="pt", device="cpu") as f:
    lines.append("experts.gate_up_proj per-layer: norm(gate_half) vs norm(up_half) summed over experts")
    for i in range(30):
        t = f.get_tensor(f"model.language_model.layers.{i}.experts.gate_up_proj").float()
        gate = t[:, :704, :]  # [8, 704, 2816]
        up = t[:, 704:, :]
        gn = gate.reshape(8, -1).pow(2).sum(1).sqrt()
        un = up.reshape(8, -1).pow(2).sum(1).sqrt()
        ratio = (gn / un)
        lines.append(f"L{i:<2} gate/un per expert: "
                     + " ".join(f"{r:.3f}" for r in ratio.tolist()))

    lines.append("")
    lines.append("dense mlp: norm(gate_proj) vs norm(up_proj)")
    for i in range(30):
        g = f.get_tensor(f"model.language_model.layers.{i}.mlp.gate_proj.weight").float().norm().item()
        u = f.get_tensor(f"model.language_model.layers.{i}.mlp.up_proj.weight").float().norm().item()
        lines.append(f"L{i:<2} gate={g:7.3f} up={u:7.3f} ratio={g/u:.4f}")

    # also q/k halves? q_proj [8192, 2816] global: halves = first 4096 (heads 0-7) vs rest. no meaning.
    # o_proj rows by head-block norms (detect scrambled head blocks)
    lines.append("")
    lines.append("global o_proj per-head-block column norms (16 blocks of 512):")
    for i in [5, 11, 17, 23, 29]:
        t = f.get_tensor(f"model.language_model.layers.{i}.self_attn.o_proj.weight").float()
        cn = t.pow(2).sum(0).sqrt().reshape(16, 512).mean(1)
        lines.append(f"L{i:<2}: " + " ".join(f"{v:.3f}" for v in cn.tolist()))

with open(OUT, "w") as fh:
    fh.write("\n".join(lines))
print("wrote", OUT)
