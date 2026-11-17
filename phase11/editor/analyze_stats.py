"""Analyze stats.json: group text-model tensors by (module type, layer), flag outliers."""
import json
import re
from collections import defaultdict

STATS = "/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/stats.json"

rows = json.load(open(STATS))
print(f"n tensors: {len(rows)}")

nonfinite = [r for r in rows if r["nan"] or r["inf"]]
print("nonfinite tensors:", [r["name"] for r in nonfinite] or "none")

text = [r for r in rows if "language_model" in r["name"] or r["name"].startswith("model.")]
text = [r for r in rows if "vision_tower" not in r["name"]]
print(f"text-side tensors: {len(text)}")
for r in text:
    print(r["name"], r["shape"], f"am={r['absmean']:.5f}", f"std={r['std']:.5f}",
          f"min={r['min']:.4f}", f"max={r['max']:.4f}", f"zf={r['zero_frac']:.4f}")

# Group by suffix pattern across layers to spot outliers vs siblings
groups = defaultdict(list)
for r in text:
    m = re.sub(r"\.\d+\.", ".N.", r["name"])
    groups[m].append(r)

print("\n=== per-group absmean summary (layer progression; flag outliers) ===")
for g in sorted(groups):
    rs = sorted(groups[g], key=lambda r: r["name"])
    ams = [r["absmean"] for r in rs]
    if len(rs) >= 3:
        med = sorted(ams)[len(ams) // 2]
        print(f"\n{g}  n={len(rs)} median_am={med:.5f}")
        for r, a in zip(rs, ams):
            ratio = a / med if med else 0
            mark = "  <<< OUTLIER" if (ratio > 1.5 or ratio < 0.67) and abs(a - med) > 1e-4 else ""
            print(f"   {r['name']:<75} am={a:.5f} ratio={ratio:.3f}{mark}")
    else:
        for r in rs:
            print(f"\n{g}  n={len(rs)}")
            print(f"   {r['name']:<75} am={r['absmean']:.5f}")
