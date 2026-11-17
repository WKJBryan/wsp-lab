"""Analyze stats_all.jsonl: flag anomalies, group by tensor role."""
import json
import math
import re
from collections import defaultdict

ED = "/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor"
rows = [json.loads(l) for l in open(f"{ED}/stats_all.jsonl")]

# strip layer index to build role groups
pat = re.compile(r"layers\.(\d+)\.")
groups = defaultdict(list)
for r in rows:
    m = pat.search(r["name"])
    key = pat.sub("layers.N.", r["name"]) if m else r["name"]
    groups[key].append(r)

print("=== NaN / Inf tensors ===")
for r in rows:
    if r["nan"] or r["inf"]:
        print(r["name"], r["nan"], r["inf"])

print("\n=== role groups (across the 30 layers): consistency ===")
for key, rs in groups.items():
    if len(rs) < 5:
        continue  # skip vision tower (27 layers) handled below, and non-layer tensors
    norms = [r["norm"] for r in rs]
    means = [r["mean"] for r in rs]
    amax = [r["absmax"] for r in rs]
    n = len(rs)
    med = sorted(norms)[n // 2]
    flags = []
    for r in rs:
        if med > 0 and (r["norm"] > 3 * med or r["norm"] < med / 3):
            flags.append((r["name"].split(".").index("layers"), r["name"], round(r["norm"], 4)))
    if flags:
        print(f"\n{key}: median_norm={med:.4f} n={n} OUTLIERS:")
        for li, nm, no in flags:
            print(f"   {nm} norm={no}")
    else:
        mn, mx = min(norms), max(norms)
        print(f"{key}: n={n} norm[{mn:.4f}..{mx:.4f}] mean[{min(means):.4f}..{max(means):.4f}] absmax[{min(amax):.2f}..{max(amax):.2f}]")

print("\n=== small vectors (scalars, router scales, norms) — full values per layer ===")
for r in rows:
    nm = r["name"]
    if nm.endswith("layer_scalar") or "per_expert_scale" in nm or "router.scale" in nm:
        print(f"{nm}: norm={r['norm']:.4f} mean={r['mean']:.4f} min={r['min']:.4f} max={r['max']:.4f} absmax={r['absmax']:.4f}")

print("\n=== RMSNorm weights: distance from 1.0 (each should be ~all-ones-ish) ===")
for r in rows:
    if "layernorm" in r["name"] or "q_norm" in r["name"] or "k_norm" in r["name"] or r["name"] == "model.language_model.norm.weight":
        n = 1
        for s in r["shape"]:
            n *= s
        ideal = math.sqrt(n)  # norm if all ones
        rel = r["norm"] / ideal
        if rel < 0.5 or rel > 2.0 or r["absmax"] > 10 or r["max"] < 0.1:
            print(f"  SUSPECT {nm}: norm={r['norm']:.3f} ideal~{ideal:.1f} rel={rel:.3f} mean={r['mean']:.4f} min={r['min']:.4f} max={r['max']:.4f} absmax={r['absmax']:.3f}")

print("\n=== global tensors ===")
for r in rows:
    if not pat.search(r["name"]) and "vision" not in r["name"]:
        print(f"{r['name']}: shape={r['shape']} mean={r['mean']:.6f} std={r['std']:.6f} norm={r['norm']:.2f} min={r['min']:.4f} max={r['max']:.4f} absmax={r['absmax']:.4f} pzero={r['pzero']:.4f}")

print("\n=== vision tower groups (27 layers) ===")
vpat = re.compile(r"vision_tower\.encoder\.layers\.(\d+)\.")
vgroups = defaultdict(list)
for r in rows:
    m = vpat.search(r["name"])
    if m:
        vgroups[vpat.sub("layers.N.", r["name"])].append(r)
for key, rs in sorted(vgroups.items()):
    norms = [r["norm"] for r in rs]
    print(f"{key}: n={len(rs)} norm[{min(norms):.4f}..{max(norms):.4f}]")

print("\n=== zero-heavy tensors (pzero > 0.4) ===")
for r in rows:
    if r["pzero"] > 0.4:
        print(f"{r['name']}: pzero={r['pzero']:.3f} std={r['std']:.6f}")
