"""Census of every head in every layer on clean reasoning text: look for an abnormal component.

Output: one table per metric, one row per layer, one column per head.
  full attention (8 heads): gate mean and its extreme dims, attention entropy, attention to the
      previous token and to position 0, gated head-output norm, head contribution through o_proj
  Gated DeltaNet (16 heads): median / 10th-percentile decay alpha = exp(g), write strength
      beta = sigmoid(b), head-output norm after the gated norm, head contribution through out_proj

Usage: python census.py [model_dir] [--n 6] [--len 1200] [--save out.pt]
"""
import sys, torch, argparse
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument('model_dir', nargs='?', default=lib.MODEL_DIR)
ap.add_argument('--n', type=int, default=6)
ap.add_argument('--len', type=int, default=1200)
ap.add_argument('--save', default='')
a = ap.parse_args()
torch.set_grad_enabled(False)

tok = AutoTokenizer.from_pretrained(a.model_dir)
m = Qwen3_5ForConditionalGeneration.from_pretrained(a.model_dir, dtype=torch.bfloat16,
                                                    attn_implementation='eager', device_map='cuda')
m.eval()
H = m.model.language_model
prac = {p['id']: p for p in lib.practice()}
G = lib.gens(1)
texts = []
for pid in ['p101', 'p124', 'p105', 'p28', 'p27', 'p103', 'p66', 'p86'][:a.n]:
    p = lib.prompt_ids(tok, prac[pid]['q'])
    texts.append((p + lib.ids_of(tok, G[pid]['text']))[:len(p) + a.len])

FL, LL = lib.FULL_LAYERS, lib.LINEAR_LAYERS
ATT = ['gate_mean', 'gate_dimmin', 'gate_dimmax', 'ent', 'prev', 'first', 'outnorm', 'contrib']
GDN = ['alpha_med', 'alpha_p10', 'beta', 'outnorm', 'contrib']
sums = {k: torch.zeros(len(FL), 8, device='cuda') for k in ATT}
gsums = {k: torch.zeros(len(LL), 16, device='cuda') for k in GDN}

for ids in texts:
    rec = {}
    handles = []
    for i, layer in enumerate(H.layers):
        if i in FL:
            sa = layer.self_attn
            def q_hook(mod, args, out, i=i):          # out [1,T,4096]
                g = torch.sigmoid(out[0].float().view(-1, 8, 512)[..., 256:])   # [T,8,256]
                rec.setdefault('gates', {})[i] = g
            def w_hook(mod, args, out, i=i):          # out[1] [1,8,T,T]
                rec.setdefault('w', {})[i] = out[1][0].float()
            def o_hook(mod, args, i=i):               # args[0] [1,T,2048]
                rec.setdefault('oin', {})[i] = args[0][0].float()
            handles += [sa.q_proj.register_forward_hook(q_hook), sa.register_forward_hook(w_hook),
                        sa.o_proj.register_forward_pre_hook(o_hook)]
        else:
            la = layer.linear_attn
            def a_hook(mod, args, out, i=i):          # out [1,T,16] (a plain tensor)
                rec.setdefault('a', {})[i] = out[0].float()
            def b_hook(mod, args, out, i=i):
                rec.setdefault('b', {})[i] = out[0].float()
            def o_hook(mod, args, i=i):
                rec.setdefault('oin', {})[i] = args[0][0].float()
            handles += [la.in_proj_a.register_forward_hook(a_hook), la.in_proj_b.register_forward_hook(b_hook),
                        la.out_proj.register_forward_pre_hook(o_hook)]
    T = len(ids)
    m.model(input_ids=torch.tensor([ids], device='cuda'))
    for h in handles:
        h.remove()
    t0 = min(200, T // 3)          # skip the prompt
    n = T - t0
    for j, i in enumerate(FL):
        sa = H.layers[i].self_attn
        g = rec['gates'][i]                       # [T,8,256]
        gd = g.mean(0)                            # [8,256] token-averaged gate per dim
        w = rec['w'][i][:, t0:, :]                # [8,n,T]
        ent = -(w * (w + 1e-12).log()).sum(-1).mean(1)                                   # [8]
        idx = torch.arange(t0, T, device='cuda')
        prev = w[:, torch.arange(n, device='cuda'), idx - 1].mean(1)                     # attention to t-1
        first = w[..., 0].mean(1)
        oin = rec['oin'][i][t0:].view(n, 8, 256)
        outnorm = oin.norm(dim=-1).mean(0)
        Wo = sa.o_proj.weight.float()
        contrib = torch.stack([(oin[:, hh] @ Wo[:, hh * 256:(hh + 1) * 256].t()).norm(dim=-1).mean() for hh in range(8)])
        vals = dict(gate_mean=g.mean((0, 2)), gate_dimmin=gd.min(1).values, gate_dimmax=gd.max(1).values,
                    ent=ent, prev=prev, first=first, outnorm=outnorm, contrib=contrib)
        for k in ATT:
            assert vals[k].shape == (8,), (k, vals[k].shape)
            sums[k][j] += vals[k]
    for j, i in enumerate(LL):
        la = H.layers[i].linear_attn
        aa = rec['a'][i][t0:]; bb = rec['b'][i][t0:]                  # [n,16]
        g = -la.A_log.float().exp() * torch.nn.functional.softplus(aa + la.dt_bias.float())
        alpha = g.exp()                                                 # [n,16]
        oin = rec['oin'][i][t0:].view(n, 16, 128)
        outnorm = oin.norm(dim=-1).mean(0)
        Wo = la.out_proj.weight.float()
        contrib = torch.stack([(oin[:, hh] @ Wo[:, hh * 128:(hh + 1) * 128].t()).norm(dim=-1).mean() for hh in range(16)])
        vals = dict(alpha_med=alpha.median(0).values, alpha_p10=alpha.quantile(0.1, dim=0),
                    beta=torch.sigmoid(bb).mean(0), outnorm=outnorm, contrib=contrib)
        for k in GDN:
            assert vals[k].shape == (16,), (k, vals[k].shape)
            gsums[k][j] += vals[k]

res = {k: (v / len(texts)).cpu() for k, v in sums.items()}
gres = {k: (v / len(texts)).cpu() for k, v in gsums.items()}
if a.save:
    torch.save(dict(attn=res, gdn=gres), a.save)


def show(d, layers, fmt):
    for k, v in d.items():
        print(f'-- {k}')
        for j, i in enumerate(layers):
            print(f'   L{i:2d}: ' + ' '.join(fmt.format(float(x)) for x in v[j]))


print(f'===== FULL ATTENTION HEADS (8 per layer), {len(texts)} texts of ~{a.len} tokens =====')
show(res, FL, '{:.2f}')
print('===== GATED DELTANET HEADS (16 per layer) =====')
show(gres, LL, '{:.3f}')
