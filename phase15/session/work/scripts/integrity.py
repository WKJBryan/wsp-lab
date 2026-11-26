"""Structural integrity checks that global statistics cannot see (CPU only).

1. MLP SwiGLU pairing: for each layer, cosine similarity between gate row i and up row i
   (diagonal) against the best off-diagonal match. Trained SwiGLU rows of one neuron are
   usually positively aligned.
2. Full attention OV pairing: for every (v head a, o slice b) pair, the top eigenvalue of the
   head's OV circuit W_o[:, b] @ W_v[a] (1024x1024, rank 256). Matched pairs should have the
   largest positive real part.
3. Full attention QK pairing: for every (q head a, kv head b) pair, how "peaky" is the QK matrix.
   Also, the per-head norm of query rows vs gate rows in q_proj.
4. GDN: per-head norms of q/k/v rows in in_proj_qkv, of z rows, of out_proj columns.
5. Zero / near-zero rows or columns in every tensor.
"""
import torch, re, json
torch.set_num_threads(8)
torch.set_grad_enabled(False)
from safetensors import safe_open
f = safe_open('/home/user/wsp14/sandbox/base_model/model.safetensors-00001-of-00001.safetensors', framework='pt', device='cpu')
L = 'model.language_model.layers.'
W = lambda name: f.get_tensor(name).float()


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


print('===== 1. MLP gate/up pairing =====')
for i in range(24):
    g = W(f'{L}{i}.mlp.gate_proj.weight'); u = W(f'{L}{i}.mlp.up_proj.weight')
    n = g.shape[0]
    diag = cos(g, u)                                  # per neuron
    # best match for each gate row among a random sample of up rows (cheap proxy for the off-diagonal)
    idx = torch.randperm(n)[:400]
    sim = torch.nn.functional.normalize(g[idx]) @ torch.nn.functional.normalize(u[idx]).t()
    off = sim - 50 * torch.eye(len(idx))
    print(f'L{i:2d} diag cos mean {diag.mean():+.3f} (frac>0.3 {float((diag > 0.3).float().mean()):.2f}, frac<0 {float((diag < 0).float().mean()):.2f}) | '
          f'off-diag mean {off[off > -10].mean():+.3f} max {off.max():+.2f} | best-match-is-diagonal {float((sim.argmax(1) == torch.arange(len(idx))).float().mean()):.2f}')

print('\n===== 2. Full attention OV pairing (top real eigenvalue of W_o[:,b] @ W_v[a]) =====')
for i in [3, 7, 11, 15, 19, 23]:
    v = W(f'{L}{i}.self_attn.v_proj.weight').view(2, 256, 1024)      # 2 kv heads
    o = W(f'{L}{i}.self_attn.o_proj.weight').view(1024, 8, 256)      # 8 slices of 256 columns
    # every query head h uses kv head h//4; the o slice h
    tab = torch.zeros(8, 2)
    for h in range(8):
        for kv in range(2):
            M = o[:, h, :] @ v[kv]
            ev = torch.linalg.eigvals(M)
            tab[h, kv] = ev.real.max()
    print(f'L{i:2d} rows = o slice (query head), cols = kv head 0,1; expected kv = h//4')
    print('     ' + '\n     '.join(' '.join(f'{x:7.3f}' for x in row) + f'   expected col {h//4}: {row[h//4]:7.3f}  other: {row[1-h//4]:7.3f}' for h, row in enumerate(tab)))

print('\n===== 3. Full attention q_proj: query part vs gate part, per head =====')
for i in [3, 7, 11, 15, 19, 23]:
    q = W(f'{L}{i}.self_attn.q_proj.weight').view(8, 2, 256, 1024)     # head, (query|gate), dim, hidden
    k = W(f'{L}{i}.self_attn.k_proj.weight').view(2, 256, 1024)
    qn = 1 + W(f'{L}{i}.self_attn.q_norm.weight'); kn = 1 + W(f'{L}{i}.self_attn.k_norm.weight')
    rowq = q[:, 0].norm(dim=-1); rowg = q[:, 1].norm(dim=-1)
    # effective (norm-scaled) query/key rows
    qe = q[:, 0] * qn[None, :, None]; ke = k * kn[None, :, None]
    # how much of the key space is "live" (non-zero k_norm) and whether queries live in the same dims
    live = (kn.abs() > 0.05)
    qe_live = (qe.norm(dim=-1) * live[None]).sum(1) / qe.norm(dim=-1).sum(1)
    print(f'L{i:2d} |q rows| by head ' + ' '.join(f'{x:.2f}' for x in rowq.mean(1)) + ' | gate rows ' + ' '.join(f'{x:.2f}' for x in rowg.mean(1)))
    print(f'      k live dims {int(live.sum())}/256 (first live {int(live.nonzero()[0])}, last {int(live.nonzero()[-1])}); '
          f'query mass in live dims by head ' + ' '.join(f'{x:.2f}' for x in qe_live))

print('\n===== 5. Exact zero rows / columns per tensor (language model only) =====')
keys = [k for k in f.keys() if k.startswith('model.language_model.layers.')]
for k in sorted(keys):
    t = W(k)
    if t.dim() != 2:
        if float(t.abs().max()) == 0:
            print('  ALL-ZERO', k)
        continue
    zr = int((t.abs().sum(1) == 0).sum()); zc = int((t.abs().sum(0) == 0).sum())
    if zr or zc:
        rows = (t.abs().sum(1) == 0).nonzero().flatten().tolist()
        print(f'  {k.replace(L, "L")}: zero rows {zr} {rows[:20]} zero cols {zc}')
print('(done)')
