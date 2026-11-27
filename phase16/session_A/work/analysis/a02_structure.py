"""Structural checks on the text weights: dead rows/cols, per-head norms, pairing of norms.
Usage: python a02_structure.py <safetensors file>
CPU only."""
import sys, math
import torch
torch.set_num_threads(8)
from safetensors import safe_open

f = safe_open(sys.argv[1], framework="pt", device="cpu")
P = "model.language_model.layers."
def T(name):
    return f.get_tensor(P + name).float()

print("=== 1. dead / near-dead rows and columns (row norm < 0.05 x median) ===")
for k in sorted(f.keys()):
    if not k.startswith("model.language_model.layers."):
        continue
    t = f.get_tensor(k).float()
    if t.dim() == 2:
        rn = t.norm(dim=1); cn = t.norm(dim=0)
        mrn = rn.median()
        dead_r = (rn < 0.05 * mrn).nonzero().flatten().tolist()
        mcn = cn.median()
        dead_c = (cn < 0.05 * mcn).nonzero().flatten().tolist()
        if dead_r or dead_c:
            print(k.replace(P, ""), "dead rows:", len(dead_r), dead_r[:40], "dead cols:", len(dead_c), dead_c[:40])
    elif t.dim() == 1:
        dead = (t.abs() < 1e-6).nonzero().flatten().tolist()
        if dead:
            print(k.replace(P, ""), "zero entries:", dead[:40])
    elif t.dim() == 3:
        rn = t.norm(dim=(1, 2))
        dead_r = (rn < 0.05 * rn.median()).nonzero().flatten().tolist()
        if dead_r:
            print(k.replace(P, ""), "dead rows:", len(dead_r), dead_r[:40])

print("\n=== 2. per-head norms, linear attention (q|k|v 16 heads x128, z 16 heads, conv1d) and full attention ===")
def headnorms(w, nheads, start, hd=128):
    # w: (out, in); rows start:start+nheads*hd
    x = w[start:start + nheads * hd]
    return x.view(nheads, hd, -1).norm(dim=(1, 2))
for L in range(24):
    row = f"L{L:02d}"
    if L % 4 == 3:
        q = T(f"{L}.self_attn.q_proj.weight")      # (4096, 1024): 8 heads x (256 q + 256 gate)? check both layouts
        k = T(f"{L}.self_attn.k_proj.weight")      # (512,1024): 2 heads x 256
        v = T(f"{L}.self_attn.v_proj.weight")
        o = T(f"{L}.self_attn.o_proj.weight")      # (1024, 2048)
        # layout A: [q(2048) ; gate(2048)], layout B: interleaved per head [q_h, gate_h]
        hA = q.view(16, 256, 1024).norm(dim=(1, 2))
        hB = q.view(8, 512, 1024)[:, :256].norm(dim=(1, 2))
        gB = q.view(8, 512, 1024)[:, 256:].norm(dim=(1, 2))
        kn = k.view(2, 256, 1024).norm(dim=(1, 2)); vn = v.view(2, 256, 1024).norm(dim=(1, 2))
        on = o.view(1024, 8, 256).permute(1, 0, 2).norm(dim=(1, 2))
        print(row, "qA(16x256):", [round(x, 3) for x in hA.tolist()])
        print(row, "qB(8):     ", [round(x, 3) for x in hB.tolist()], " gateB:", [round(x, 3) for x in gB.tolist()])
        print(row, "k:", [round(x, 3) for x in kn.tolist()], "v:", [round(x, 3) for x in vn.tolist()],
              "o:", [round(x, 3) for x in on.tolist()])
    else:
        qkv = T(f"{L}.linear_attn.in_proj_qkv.weight")   # (6144, 1024)
        z = T(f"{L}.linear_attn.in_proj_z.weight")       # (2048, 1024)
        o = T(f"{L}.linear_attn.out_proj.weight")        # (1024, 2048)
        cv = f.get_tensor(P + f"{L}.linear_attn.conv1d.weight").float()   # (6144,1,4)
        qn = headnorms(qkv, 16, 0); kn = headnorms(qkv, 16, 2048); vn = headnorms(qkv, 16, 4096)
        zn = headnorms(z, 16, 0)
        on = o.view(1024, 16, 128).permute(1, 0, 2).norm(dim=(1, 2))
        cn = cv.view(3, 16, 128, 4).norm(dim=(2, 3))
        r3 = lambda x: [round(v, 3) for v in x.tolist()]
        print(row, "q:", r3(qn)); print(row, "k:", r3(kn)); print(row, "v:", r3(vn))
        print(row, "z:", r3(zn)); print(row, "o:", r3(on))
        print(row, "conv q:", r3(cn[0]), "conv k:", r3(cn[1]), "conv v:", r3(cn[2]))

print("\n=== 3. pairing of norms: gate/up/down per layer (corr of log norms) ===")
def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))
for L in range(24):
    g = T(f"{L}.mlp.gate_proj.weight"); u = T(f"{L}.mlp.up_proj.weight"); d = T(f"{L}.mlp.down_proj.weight")
    lg = g.norm(dim=1).log(); lu = u.norm(dim=1).log(); ld = d.norm(dim=0).log()
    imp = (g.norm(dim=1) * u.norm(dim=1) * d.norm(dim=0))
    print(f"L{L:02d} corr(gate,up)={corr(lg, lu):+.3f} corr(gate,down)={corr(lg, ld):+.3f} corr(up,down)={corr(lu, ld):+.3f}"
          f"  imp max/median={float(imp.max() / imp.median()):.1f}  top1 share={float(imp.max() / imp.sum()) * 100:.2f}%")

print("\n=== 4. pairing across layers: are any two tensors identical or near-identical? ===")
import itertools
names = sorted(k for k in f.keys() if k.startswith("model.language_model.layers.") and k.endswith("weight"))
vecs = {}
for k in names:
    t = f.get_tensor(k).float().flatten()
    vecs[k] = t / (t.norm() + 1e-12)
groups = {}
for k in names:
    groups.setdefault(k.split(".")[-3] if "self_attn" in k or "linear_attn" in k or "mlp" in k else "ln", []).append(k)
# compare same-role tensors across layers
import re
role = {}
for k in names:
    r = re.sub(r"layers\.\d+\.", "", k)
    role.setdefault(r, []).append(k)
for r, ks in sorted(role.items()):
    best = []
    for a, b in itertools.combinations(ks, 2):
        if vecs[a].shape != vecs[b].shape:
            continue
        c = float(vecs[a] @ vecs[b])
        best.append((c, a.split("layers.")[1].split(".")[0], b.split("layers.")[1].split(".")[0]))
    best.sort(reverse=True)
    print(f"{r:45s} most alike layers: {best[0][1]}~{best[0][2]} cos={best[0][0]:+.3f} | 2nd {best[1][1]}~{best[1][2]} cos={best[1][0]:+.3f} | median cos={sorted(x[0] for x in best)[len(best)//2]:+.3f}")
