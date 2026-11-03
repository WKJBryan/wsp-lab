"""Editor-side OFFLINE analysis of phase1/target.pt (PROTOCOL.md rule 4: unlimited).

Spends zero env episodes: all observations are synthetic grid points.
Usage: uv run python phase1/editor/analyze.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from policy import make_policy

HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / "target.pt"

T = torch.load(TARGET, map_location="cpu", weights_only=True)

print("== per-tensor stats ==")
for k, v in T.items():
    v = v.float()
    print(
        f"{k:12s} shape {str(tuple(v.shape)):12s} | fro {v.norm():8.3f} | "
        f"mean {v.mean():+8.4f} std {v.std():7.4f} | min {v.min():+8.4f} max {v.max():+8.4f} | "
        f"zeros {(v == 0).float().mean() * 100:5.1f}%"
    )

policy = make_policy()
policy.load_state_dict(T)
policy.eval()

# Synthetic grid spanning CartPole's reachable observation space.
pos = torch.linspace(-2.4, 2.4, 21)
vel = torch.linspace(-3.0, 3.0, 21)
ang = torch.linspace(-0.2095, 0.2095, 21)
avv = torch.linspace(-3.0, 3.0, 21)
G = torch.stack(torch.meshgrid(pos, vel, ang, avv, indexing="ij"), dim=-1).reshape(-1, 4)

with torch.no_grad():
    z1 = torch.relu(policy[0](G))          # layer0 + ReLU(1)
    z2 = torch.relu(policy[2](z1))         # layer2 + ReLU(3)
    logits = policy[4](z2)                 # head

d1 = (z1 == 0).float().mean(dim=0)   # per-unit dead fraction, layer 1
d2 = (z2 == 0).float().mean(dim=0)   # per-unit dead fraction, layer 2
print(f"\n== dead ReLU units over grid ==")
print(f"layer1: >99% dead: {(d1 > 0.99).sum().item():3d}/128 | >90% dead: {(d1 > 0.9).sum().item():3d}/128")
print(f"layer2: >99% dead: {(d2 > 0.99).sum().item():3d}/128 | >90% dead: {(d2 > 0.9).sum().item():3d}/128")

diff = logits[:, 1] - logits[:, 0]
print(f"\n== action logits over grid (n={G.shape[0]}) ==")
print(f"logit-diff (act1-act0): mean {diff.mean():+.4f} std {diff.std():.4f} min {diff.min():+.4f} max {diff.max():+.4f}")
print(f"fraction choosing action 1: {(diff > 0).float().mean():.4f}")
print(f"|logit-diff| median: {diff.abs().median():.4f}")

# how much does the head see? effective rank of z2 on the grid
z2c = z2 - z2.mean(dim=0)
s = torch.linalg.svdvals(z2c)
evr = (s ** 2 / (s ** 2).sum()).cumsum(0)
print(f"\n== z2 signal into head ==")
print(f"effective rank (>1% var): {(evr < 0.99).sum().item() + 1} of 128")
print(f"z2 norm mean {z2.norm(dim=1).mean():.4f}; max {z2.norm(dim=1).max():.4f}")

# head weight structure: which z2 units actually drive the decision
W = policy[4].weight.float()
head_norm = W.norm(dim=0)
print(f"\nhead column norms: top5 {head_norm.topk(5).values.tolist()}")
print(f"head bias: {policy[4].bias.tolist()}")
