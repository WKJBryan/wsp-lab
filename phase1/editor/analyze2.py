"""Editor offline analysis round 2: action-map structure + hypothesis screening.

Zero env episodes: synthetic states only (PROTOCOL.md rule 4).
Usage: uv run python phase1/editor/analyze2.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from policy import make_policy

HERE = Path(__file__).resolve().parent
T = torch.load(HERE.parent / "target.pt", map_location="cpu", weights_only=True)

policy = make_policy()
policy.load_state_dict(T)
policy.eval()

flipped = make_policy()
flipped.load_state_dict(T)
with torch.no_grad():
    flipped[4].weight.mul_(-1)
    flipped[4].bias.mul_(-1)
flipped.eval()


def logit_diff(p, g):
    with torch.no_grad():
        z1 = torch.relu(p[0](g))
        z2 = torch.relu(p[2](z1))
        lg = p[4](z2)
    return lg[:, 1] - lg[:, 0]


def state(x, v, th, om):
    return torch.tensor([[x, v, th, om]], dtype=torch.float32)


print("== action sign map at pos=0, vel=0 (rows: angle -0.2..0.2, cols: angvel -3..3) ==")
print("     " + "".join(f"{om:+.1f}"[0] for om in torch.linspace(-3, 3, 11)) + "   (o=anglev, +/- col sign)")
omegas = torch.linspace(-3, 3, 11)
for th in torch.linspace(-0.2, 0.2, 11):
    row = "".join("+" if logit_diff(policy, state(0, 0, th, om)).item() > 0 else "-" for om in omegas)
    print(f"th={th:+.2f} {row}")

print("\n== hypothesis screening: agreement with stabilizing heuristic ==")
print("heuristic: action=1 iff th + k_om*om + k_x*x + k_v*v > 0  (push WITH pole-lean correction)")


def agreement(p, k_om, k_v, k_x, region):
    if region == "small":
        xs = torch.linspace(-1, 1, 11); vs = torch.linspace(-1, 1, 11)
        ths = torch.linspace(-0.1, 0.1, 11); oms = torch.linspace(-1, 1, 11)
    else:
        xs = torch.linspace(-2.2, 2.2, 11); vs = torch.linspace(-2.5, 2.5, 11)
        ths = torch.linspace(-0.2, 0.2, 11); oms = torch.linspace(-2.5, 2.5, 11)
    G = torch.stack(torch.meshgrid(xs, vs, ths, oms, indexing="ij"), dim=-1).reshape(-1, 4)
    want = (G[:, 2] + k_om * G[:, 3] + k_x * G[:, 0] + k_v * G[:, 1]) > 0
    got = logit_diff(p, G) > 0
    return (want == got).float().mean().item()


for region in ["small", "wide"]:
    print(f"-- region: {region} --")
    for name, p in [("original", policy), ("head-flipped", flipped)]:
        best = max(
            ((agreement(p, kom, kv, kx, region), kom, kv, kx)
             for kom in [0.3, 0.5, 0.8]
             for kv in [0.0, 0.3]
             for kx in [0.0, 0.3]),
            key=lambda t: t[0],
        )
        print(f"{name:12s} best agreement {best[0]*100:5.1f}%  (k_om={best[1]}, k_v={best[2]}, k_x={best[3]})")
