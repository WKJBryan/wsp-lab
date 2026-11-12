"""Phase-5 analysis: anatomy of the original's edit (CPU, free).

Questions: which input columns really change (vs re-serialization noise)? Is the edit
a rank-1 direction or column scaling ("reversed and amplified", per Toledo's finding)?
Do L15 and L19 edits share structure?

Usage: uv run python phase5/analyze_edit.py
"""

from pathlib import Path

import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
STOCK = HERE.parent / "phase3" / "models" / "stock"
EDITED = HERE.parent / "phase3" / "models" / "c3_original_edit"


def load(d, key):
    files = sorted(Path(d).glob("*.safetensors"))
    for f in files:
        ts = load_file(str(f))
        if key in ts:
            return ts[key].float()
    raise KeyError(key)


for layer in [15, 19]:
    key = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
    S = load(STOCK, key)
    E = load(EDITED, key)
    D = E - S  # (1024, 3584)

    col_norms = D.norm(dim=0)  # per input column
    med = col_norms.median()
    real = col_norms > 8 * med
    n_real = int(real.sum())

    print(f"\n===== layer {layer} down_proj {tuple(D.shape)} =====")
    print(f"column delta norms: median {med:.5f} | p90 {col_norms.quantile(0.9):.5f} | "
          f"max {col_norms.max():.4f} | real columns (>8x median): {n_real}")

    if n_real:
        R = D[:, real]
        stock_cols = S[:, real]
        # rank structure of the real edit
        U, sv, Vh = torch.linalg.svd(R, full_matrices=False)
        top_share = (sv[0] ** 2 / (sv ** 2).sum()).item()
        print(f"real-block fro {R.norm():.3f} | top-1 singular share {top_share:.3f} "
              f"| top-5 sv {[round(x,3) for x in sv[:5].tolist()]}")

        # is the edit column scaling of the stock weights?  D_j ~= c_j * S_j
        num = (R * stock_cols).sum(0)
        den = (stock_cols * stock_cols).sum(0)
        c = num / den
        resid = (R - c * stock_cols).norm(dim=0) / (R.norm(dim=0) + 1e-9)
        print(f"per-real-column fit D_j = c_j * S_j : c range [{c.min():+.3f}, {c.max():+.3f}] "
              f"median {c.median():+.3f} | residual ratio median {resid.median():.3f}")

        idx = torch.where(real)[0]
        print(f"real column ids (first 20): {idx[:20].tolist()}")
        globals()[f"real_{layer}"] = idx
        globals()[f"c_{layer}"] = c

r15, r19 = globals()["real_15"], globals()["real_19"]
inter = len(set(r15.tolist()) & set(r19.tolist()))
print(f"\n===== cross-layer =====")
print(f"real column overlap L15∩L19: {inter} of {len(r15)}/{len(r19)} "
      f"(Jaccard {inter / len(set(r15.tolist()) | set(r19.tolist())):.3f})")
print(f"c(L15) median {globals()['c_15'].median():+.3f} | c(L19) median {globals()['c_19'].median():+.3f}")
