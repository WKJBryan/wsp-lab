"""Phase-7 atlas — where in the prompt does each MLP unit fire? (protocol addendum 8)

One forward pass of the stock model over the 500 MATH-500 prompts. Per unit (24 x 3584):
  first      activation on token 0 (the same token, with no context, in every prompt)
  mean, sd   over every position after the 3 shared opening tokens
  mean_abs, max_abs   same positions
Finds the "loud but empty" units: huge on the first token, quiet everywhere else.

Outputs: phase7/probe_out/positions.npz
Usage: uv run python phase7/scan_positions.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SKIP, NL, NU, dev = 3, 24, 3584, "cuda"

stock = ROOT / "phase3" / "models" / "stock"
tok = AutoTokenizer.from_pretrained(stock)
model = AutoModelForCausalLM.from_pretrained(stock, dtype=torch.bfloat16).eval().to(dev)
down = {int(n.split("layers.")[1].split(".")[0]): m for n, m in model.named_modules()
        if n.endswith("mlp.down_proj") and tuple(m.weight.shape) == (1024, NU)}
assert sorted(down) == list(range(NL))
z = lambda: torch.zeros(NL, NU, dtype=torch.float64, device=dev)
first, s1, s2, sa, mx, n = None, z(), z(), z(), z(), 0
cur = {}
for L, m in down.items():
    m.register_forward_pre_hook(lambda _, inp, L=L: cur.__setitem__(L, inp[0][0].float()))

problems = json.loads((ROOT / "phase3c" / "problems.json").read_text(encoding="utf-8"))
with torch.no_grad():
    for p in problems:
        prompt = tok.apply_chat_template([{"role": "user", "content": p["q"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
        model(tok(prompt, return_tensors="pt").input_ids.to(dev))
        x = torch.stack([cur[L] for L in range(NL)]).double()  # (24, T, 3584)
        if first is None:
            first = x[:, 0].clone()
        drift = float((x[:, 0] - first).abs().max())  # same token, no context: equal up to bf16 rounding
        assert drift < 0.02 * float(first.abs().max()) + 0.05, f"token 0 is not the same in every prompt (max diff {drift:.3f})"
        r = x[:, SKIP:]
        s1 += r.sum(1); s2 += r.pow(2).sum(1); sa += r.abs().sum(1); mx = torch.maximum(mx, r.abs().amax(1)); n += r.shape[1]
mean = s1 / n
out = dict(first=first, mean=mean, sd=(s2 / n - mean ** 2).clamp_min(0).sqrt(), mean_abs=sa / n, max_abs=mx)
np.savez(HERE / "probe_out" / "positions.npz", **{k: v.cpu().numpy().astype(np.float32) for k, v in out.items()})

f, sd, ma = (out[k].cpu().numpy() for k in ("first", "sd", "mean_abs"))
ratio = np.abs(f) / (ma + 1e-6)
print(f"positions after the opening: {n}")
print("units by |first-token activation|:", {t: int((np.abs(f) >= t).sum()) for t in (1, 5, 10, 20, 50)})
print("  ... that are also >= 50x their usual |activity| elsewhere:", {t: int(((np.abs(f) >= t) & (ratio >= 50)).sum()) for t in (1, 5, 10, 20, 50)})
top = np.argsort(-np.abs(f).ravel())[:25]
print("loudest on the first token (floor, unit, first, usual |act| elsewhere, max elsewhere):")
for i in top:
    L, u = divmod(int(i), NU)
    print(f"  L{L:2d} u{u:4d}  first {f[L, u]:8.1f}  elsewhere {ma[L, u]:.3f}  max {float(out['max_abs'][L, u]):.2f}")
print("per-position SD percentiles (1, 5, 25, 50, 95, 99):", [round(float(q), 4) for q in np.percentile(sd, [1, 5, 25, 50, 95, 99])])
print("near-silent units (SD below):", {t: int((sd < t).sum()) for t in (1e-4, 1e-3, 3e-3, 1e-2)},
      "| and max |act| below 0.05:", int((out["max_abs"].cpu().numpy() < 0.05).sum()))
