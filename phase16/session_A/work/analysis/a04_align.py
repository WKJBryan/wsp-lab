"""Static analysis: which heads / GDN heads / MLP neurons write into the direction of a few
behaviourally important tokens (effective unembedding u = (1+norm.weight) * E[token])?
Usage: python a04_align.py <safetensors file>
CPU only."""
import sys, json
import torch
torch.set_num_threads(8)
from safetensors import safe_open
from transformers import AutoTokenizer

f = safe_open(sys.argv[1], framework="pt", device="cpu")
tok = AutoTokenizer.from_pretrained("/home/user/wsp14/sandbox/model")
P = "model.language_model."
E = f.get_tensor(P + "embed_tokens.weight").float()
nw = f.get_tensor(P + "norm.weight").float()
u_all = (1.0 + nw)[None, :] * E            # effective unembedding rows [V, D]

words = ["</think>", "<|im_end|>", " Wait", " But", " Hmm", " However", " Alternatively", " So", " Let", " Therefore", " Actually", " Instead", " Also", " Now", " Double", " Okay", " Next"]
wids = {}
for w in words:
    ids = tok.encode(w, add_special_tokens=False)
    wids[w] = ids[0] if len(ids) >= 1 else None
    if len(ids) != 1:
        print("note:", repr(w), "->", ids, [tok.decode([i]) for i in ids])

def report(name, u, topn=12):
    un = u / u.norm()
    rows = []
    for L in range(24):
        if L % 4 == 3:
            o = f.get_tensor(P + f"layers.{L}.self_attn.o_proj.weight").float()      # (1024, 2048)
            for h in range(8):
                W = o[:, h * 256:(h + 1) * 256]                                       # (1024, 256)
                rows.append((float((W.T @ un).norm()), f"L{L} attn head {h}", float(W.norm())))
        else:
            o = f.get_tensor(P + f"layers.{L}.linear_attn.out_proj.weight").float()   # (1024, 2048)
            for h in range(16):
                W = o[:, h * 128:(h + 1) * 128]
                rows.append((float((W.T @ un).norm()), f"L{L} gdn head {h}", float(W.norm())))
        d = f.get_tensor(P + f"layers.{L}.mlp.down_proj.weight").float().T             # (3584, 1024) rows = neurons
        al = (d @ un)
        for j in al.abs().topk(3).indices.tolist():
            rows.append((float(al[j]), f"L{L} mlp neuron {j}", float(d[j].norm())))
    rows.sort(key=lambda r: -abs(r[0]))
    print(f"\n##### {name}: |u|={float(u.norm()):.3f}; strongest aligned output directions (heads: |W^T u|/|u|; neurons: W_j . u/|u|)")
    for v, n, nrm in rows[:topn]:
        print(f"  {v:+8.3f}  {n:22s} (|W|={nrm:.3f})")
    print("  ... most negative / 'inhibitory' (neurons only):")
    for v, n, nrm in sorted([r for r in rows if 'mlp' in r[1]], key=lambda r: r[0])[:8]:
        print(f"  {v:+8.3f}  {n:22s} (|W|={nrm:.3f})")

for w, i in wids.items():
    if i is not None:
        report(w, u_all[i])

print("\n##### per-layer summary of alignment with </think> (unit-normed output directions)")
u = u_all[248069]; un = u / u.norm()
for L in range(24):
    d = f.get_tensor(P + f"layers.{L}.mlp.down_proj.weight").float().T
    dn = d / (d.norm(dim=1, keepdim=True) + 1e-9)
    al = dn @ un
    o = f.get_tensor(P + f"layers.{L}.self_attn.o_proj.weight").float() if L % 4 == 3 else f.get_tensor(P + f"layers.{L}.linear_attn.out_proj.weight").float()
    hd = 256 if L % 4 == 3 else 128
    hn = o.view(1024, -1, hd).permute(1, 0, 2)
    hal = (hn / hn.norm(dim=(1, 2), keepdim=True)).reshape(o.shape[1] // hd, -1) @ un
    print(f"L{L:02d} mlp mean|al|={al.abs().mean():.4f} max+={al.max():.3f} max-={al.min():.3f} | heads: " +
          " ".join(f"{x:+.3f}" for x in hal.tolist()))
