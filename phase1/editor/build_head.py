"""Editor edit generator: construct a new action head on the existing feature layers.

Method (all offline, synthetic states, zero env episodes):
1. Measure how each observation axis flows into layer-2 activations z2.
2. Least-squares fit head (w, beta) so logit-diff ≈ gain . obs over the envelope.
3. Save candidate; verify offline (sign map, linear-fit error).

Usage: uv run python phase1/editor/build_head.py [th om x v] [out_name]
Defaults: 1.0 0.5 -0.15 -0.1 cand_v1.pt
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

# --- 1. sensitivities of z2 wrt each obs axis, averaged over operating region ---
torch.manual_seed(0)
S = torch.cat(
    [
        torch.zeros(200, 4),
        torch.distributions.Uniform(
            torch.tensor([-0.5, -0.5, -0.05, -0.5]),
            torch.tensor([0.5, 0.5, 0.05, 0.5]),
        ).sample([800]),
    ]
)
DELTA = 1e-3


def z2_of(p, g):
    with torch.no_grad():
        a = torch.relu(p[0](g))
        return torch.relu(p[2](a))


base = z2_of(policy, S)
sens = []
for i in range(4):
    S2 = S.clone()
    S2[:, i] += DELTA
    s_i = (z2_of(policy, S2) - base) / DELTA  # (1000, 128)
    sens.append(s_i.mean(0))  # average local linearization
mean_act = base.mean(0)

print("== measured sensitivity of z2 to each obs axis ==")
for name, s in zip("x v th om".split(), sens):
    print(f"{name:>3s}: |s|={s.norm():8.4f}  max|s|={s.abs().max():8.4f}  live units (|s|>1e-4): {(s.abs() > 1e-4).sum().item():3d}/128")

# --- 2. build head: least-squares fit of (w, beta) on input-output pairs ---
import sys

args = sys.argv[1:]
GAIN = dict(
    th=float(args[0]) if len(args) > 0 else 1.0,
    om=float(args[1]) if len(args) > 1 else 0.5,
    x=float(args[2]) if len(args) > 2 else -0.15,
    v=float(args[3]) if len(args) > 3 else -0.1,
)
OUT = args[4] if len(args) > 4 else "cand_v1.pt"
print(f"target gains: th={GAIN['th']} om={GAIN['om']} x={GAIN['x']} v={GAIN['v']} -> {OUT}")

torch.manual_seed(1)
S_fit = torch.cat(
    [
        torch.distributions.Uniform(
            torch.tensor([-2.2, -2.5, -0.18, -2.5]),
            torch.tensor([2.2, 2.5, 0.18, 2.5]),
        ).sample([1200]),
        torch.distributions.Uniform(
            torch.tensor([-0.5, -0.5, -0.05, -0.5]),
            torch.tensor([0.5, 0.5, 0.05, 0.5]),
        ).sample([1800]),  # B4: weight harder toward the near-center regime
    ]
)
Z = z2_of(policy, S_fit)  # (N, 128)
g_S = (
    GAIN["x"] * S_fit[:, 0] + GAIN["v"] * S_fit[:, 1]
    + GAIN["th"] * S_fit[:, 2] + GAIN["om"] * S_fit[:, 3]
)
X_design = torch.cat([Z, torch.ones(Z.shape[0], 1)], dim=1)
sol = torch.linalg.lstsq(X_design, g_S.unsqueeze(1)).solution.squeeze(1)
w, beta_fit = sol[:-1], sol[-1]
w = w / 1.0
print(f"\nregression head: |w|={w.norm():.4f}, beta={beta_fit:+.4f}, fit residual "
      f"{(X_design @ sol - g_S).norm() / g_S.norm():.4f}")

# --- 3. boundary centering comes from the regression intercept ---
beta = beta_fit

# --- 4. write candidate: only head (layer 4) changes ---
cand = {k: v.clone() for k, v in T.items()}
w_row = w / 2.0
cand["4.weight"] = torch.stack([-w_row, w_row])          # diff = w.z2 + beta
cand["4.bias"] = torch.tensor([-beta / 2.0, beta / 2.0])
torch.save(cand, HERE / OUT)

edit_delta = sum((cand[k] - T[k]).norm() ** 2 for k in T).sqrt()
print(f"\ncandidate saved: cand_v1.pt | edit frobenius delta = {edit_delta:.4f} over 258 of 17,410 params")

# --- offline verification ---
c_pol = make_policy()
c_pol.load_state_dict(cand)
c_pol.eval()


def logit_diff(p, g):
    with torch.no_grad():
        a = torch.relu(p[0](g))
        b = torch.relu(p[2](a))
        lg = p[4](b)
    return lg[:, 1] - lg[:, 0]


print("\n== candidate sign map at pos=0, vel=0 ==")
omegas = torch.linspace(-3, 3, 11)
for th in torch.linspace(-0.2, 0.2, 11):
    row = "".join("+" if logit_diff(c_pol, torch.tensor([[0.0, 0.0, th, om]])).item() > 0 else "-" for om in omegas)
    print(f"th={th:+.2f} {row}")


def linearity_check():
    """How well does w.z2 + beta match the intended gain.dot(obs)?"""
    with torch.no_grad():
        pred = logit_diff(c_pol, S)
    target = GAIN["th"] * S[:, 2] + GAIN["om"] * S[:, 3] + GAIN["x"] * S[:, 0] + GAIN["v"] * S[:, 1]
    err = (pred - target)
    rel = err.norm() / target.norm()
    print(f"\nintended-vs-realized linear controller: rel error {rel:.4f} over operating region")


linearity_check()
