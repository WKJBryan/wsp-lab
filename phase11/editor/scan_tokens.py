"""Check embedding rows of control tokens vs the row-norm distribution.
CPU-only (free analysis).

Usage (in WSL): ~/wsp3c-venv/bin/python scan_tokens.py
"""
import torch
from safetensors import safe_open

P = "/home/user/wsp11/sandbox/model/model.safetensors"

SPECIAL = {
    0: "pad", 1: "eos", 2: "bos", 50: "eos50?",
    106: "eot(<end_of_turn>)", 107: "107?",
    255999: "boi", 256000: "boa",
    258880: "img", 258881: "audio", 258882: "eoi", 258883: "eoa", 258884: "video",
}

with safe_open(P, "pt") as f:
    e = f.get_tensor("model.language_model.embed_tokens.weight").float()
    rn = e.norm(dim=1)
    med = rn.median().item()
    print(f"row norm dist: mean={rn.mean():.4f} std={rn.std():.4f} "
          f"min={rn.min():.4f} max={rn.max():.4f} median={med:.4f}")
    q = torch.quantile(rn, torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99]))
    print("quantiles 1/5/50/95/99:", [round(v, 4) for v in q.tolist()])
    for idx, name in SPECIAL.items():
        r = e[idx]
        print(f"row {idx:>6} ({name:>18}): norm={rn[idx]:8.4f} "
              f"z={(rn[idx]-rn.mean())/rn.std():7.2f} "
              f"rowmean={r.mean():.4g} rowstd={r.std():.4g} absmax={r.abs().max():.4g}")
    # also: which rows have the smallest norms (sabotage often zeroes rows)
    srt = rn.sort()
    print("\nsmallest-20 row norms:", [(int(i), round(v, 4)) for v, i in
                                        zip(srt.values[:20], srt.indices[:20])])
    print("largest-10 row norms:", [(int(i), round(v, 4)) for v, i in
                                     zip(srt.values[-10:], srt.indices[-10:])])
