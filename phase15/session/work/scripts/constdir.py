"""Is there an approximately constant direction in the normalised MLP input of each layer?

For 16 calibration windows (clean pass-1 reasoning): x = post_attention_layernorm(h) at every
token position. Report per layer: the mean direction xbar, how stable the projection x.xbar/|xbar|^2
is across positions (mean, std, 1st/99th percentile), and the same for the single most stable
coordinate. Also: how well a single direction separates 'line-start decision positions' from all
other positions at that layer (AUC of the mean-difference direction), for use as a second factor.

Usage: python constdir.py [--save out/constdir.pt]
"""
import sys, torch, json, random, re
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
ap = __import__('argparse').ArgumentParser()
ap.add_argument('--save', default='/home/user/wsp14/work/out/constdir.pt')
a = ap.parse_args()
m, tok = lib.load()
H = m.model.language_model
prac = {p['id']: p for p in lib.practice()}
G = lib.gens(1)
loops = json.load(open('/home/user/wsp14/work/out/loops_1.json'))
dec = lambda t: tok.decode([t], skip_special_tokens=False, clean_up_tokenization_spaces=False)
SPACE = re.compile(r'^[\s\*\-\u2022]*$')
rng = random.Random(1)
pids = [p for p in sorted(G, key=lambda s: int(s[1:])) if (loops[p]['loop_start'] or 9000) > 2600]
sel = rng.sample(pids, 16)
X = {}
line_mask = {}
handles = []
for i, layer in enumerate(H.layers):
    def pre(mod, args, kwargs, i=i):
        x = (args[0] if args else kwargs['hidden_states'])[0].float()      # [T,1024]
        X.setdefault(i, []).append(x.cpu())
    handles.append(layer.post_attention_layernorm.register_forward_pre_hook(pre, with_kwargs=True))
for pid in sel:
    g = G[pid]
    p_ids = lib.prompt_ids(tok, prac[pid]['q'])
    n = min(2400 - len(p_ids), (loops[pid]['loop_start'] or g['n_tokens']) - 100)
    off = rng.randint(0, max(0, (loops[pid]['loop_start'] or g['n_tokens']) - 100 - n))
    ids = p_ids + lib.ids_of(tok, g['text'])[off:off + n]
    m.model(input_ids=torch.tensor([ids], device='cuda'))
    # line-start decision positions: previous tokens are newline(+junk) and next token is real
    ls = torch.zeros(len(ids), dtype=torch.bool)
    i = 0
    while i < len(ids):
        if '\n' in dec(ids[i]):
            j = i + 1
            while j < len(ids) and SPACE.match(dec(ids[j])):
                j += 1
            if j < len(ids):
                ls[j - 1] = True
            i = j
        else:
            i += 1
    line_mask[pid] = ls
for h in handles:
    h.remove()


def auc(pos, neg):
    allv = torch.cat([pos, neg]); r = allv.argsort().argsort().float() + 1
    return float((r[:pos.numel()].sum() - pos.numel() * (pos.numel() + 1) / 2) / (pos.numel() * neg.numel()))


out = {}
print('layer | x.xbar/|xbar|^2 over positions: mean sd p1 p99 | |xbar|/mean|x| | best single coord: dim, rel-std | line-start AUC (mean-diff dir, leave-one-trace-out) | cos(xbar, xbar of other layers mean)')
xbars = {i: torch.cat(X[i]).mean(0) for i in range(24)}
for i in range(24):
    xs = torch.cat(X[i])                      # [N,1024]
    xb = xbars[i]
    proj = xs @ xb / xb.dot(xb)
    mean_norm = xs.norm(dim=1).mean()
    coord = xs / xs.mean(0).abs().clamp_min(1e-6)
    relstd = coord.std(0) / coord.mean(0).abs()
    best = int(relstd.argmin())
    masks = torch.cat([line_mask[p] for p in sel])
    aucs = []
    keys = list(line_mask)
    # leave-one-trace-out mean-difference direction for line starts
    per = [len(line_mask[p]) for p in keys]
    offsets = torch.cumsum(torch.tensor([0] + per[:-1]), 0).tolist()
    for k, p in enumerate(keys):
        sl = slice(offsets[k], offsets[k] + per[k])
        tr = torch.ones(xs.shape[0], dtype=torch.bool); tr[sl] = False
        d = xs[tr][masks[tr]].mean(0) - xs[tr][~masks[tr]].mean(0)
        s = xs[sl] @ d
        aucs.append(auc(s[masks[sl]], s[~masks[sl]]))
    others = torch.stack([xbars[j] for j in xbars if j != i]).mean(0)
    q = torch.quantile(proj, torch.tensor([0.01, 0.99]))
    print(f'  {i:2d}  | {float(proj.mean()):.3f} {float(proj.std()):.3f} {float(q[0]):.3f} {float(q[1]):.3f} | {float(xb.norm() / mean_norm):.3f} | '
          f'dim {best} rel-std {float(relstd[best]):.3f} | {sum(aucs)/len(aucs):.3f} | {float(torch.nn.functional.cosine_similarity(xb, others, dim=0)):+.2f}')
    out[i] = dict(xbar=xb, proj_mean=float(proj.mean()), proj_std=float(proj.std()), auc=sum(aucs) / len(aucs))
torch.save(dict(layers=out), a.save)
