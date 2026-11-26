"""Lesion study: zero one component at a time (runtime hooks; NO weight is changed) and measure the
change in next-token loss on clean, pre-loop reasoning text of pass 1.

Components:
  attn (L,h)    : one full-attention head      -> zero its 256-wide slice of the o_proj input
  gdn  (L,h)    : one Gated DeltaNet head      -> zero its 128-wide slice of the out_proj input
  mlp  (L,g)    : 256 MLP neurons of a layer   -> zero those rows before down_proj
  sub-attn (L) / sub-mlp (L): whole sublayer output
Readout: mean NLL (nats/token) over scored positions.  dNLL = NLL(lesioned) - NLL(base):
  positive = the component helps; ~0 = unused; negative = the component HURTS the model.

Usage: python lesions.py [--part heads|mlp|sub|all] [--nwin 64] [--tag base] [--model_dir DIR]
"""
import sys, torch, argparse, json, random, time
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
ap = argparse.ArgumentParser()
ap.add_argument('--part', default='all')
ap.add_argument('--nwin', type=int, default=64)
ap.add_argument('--tag', default='base')
ap.add_argument('--model_dir', default=lib.MODEL_DIR)
ap.add_argument('--T', type=int, default=2400)
ap.add_argument('--score', type=int, default=1000)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--neurons', default='', help='for --part neuron: list like 14:1439,14:2910 (layer:neuron)')
a = ap.parse_args()
torch.set_grad_enabled(False)
m, tok = lib.load(a.model_dir)
H = m.model.language_model
FL, LL = lib.FULL_LAYERS, lib.LINEAR_LAYERS
prac = {p['id']: p for p in lib.practice()}
G = lib.gens(1)
loops = json.load(open('/home/user/wsp14/work/out/loops_1.json'))
random.seed(a.seed)

# ---- calibration windows ------------------------------------------------------
windows = []
for pid in sorted(G, key=lambda s: int(s[1:])):
    g = G[pid]
    ls = loops[pid]['loop_start']
    usable = (ls - 100) if ls else (g['n_tokens'] if g['finish'] == 'stop' else 6000)
    p = lib.prompt_ids(tok, prac[pid]['q'])
    n_gen = min(a.T - len(p), usable)
    if n_gen < a.score + 300:
        continue
    off = random.randint(0, int(usable - n_gen)) if usable > n_gen else 0
    gen = lib.ids_of(tok, g['text'])
    ids = p + gen[off:off + n_gen]
    windows.append((pid, ids))
random.shuffle(windows)
windows = windows[:a.nwin]
print(f'{len(windows)} calibration windows', flush=True)


def batches():
    ws = sorted(windows, key=lambda w: len(w[1]))
    for i in range(0, len(ws), 8):
        chunk = ws[i:i + 8]
        T = max(len(w[1]) for w in chunk)
        x = torch.full((len(chunk), T), tok.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros((len(chunk), T), dtype=torch.long)
        for j, (_, ids) in enumerate(chunk):
            x[j, :len(ids)] = torch.tensor(ids); mask[j, :len(ids)] = 1
        yield x.cuda(), mask.cuda(), [len(ids) for _, ids in chunk]


bats = list(batches())


def evaluate(hooks=()):
    """hooks: list of callables f() -> RemovableHandle."""
    tot, cnt = 0.0, 0
    for x, mask, lens in bats:
        hs = [f() for f in hooks]
        out = m.model(input_ids=x, attention_mask=mask)
        for h in hs:
            h.remove()
        idx_b, idx_p, tgt = [], [], []
        for j, n in enumerate(lens):
            lo = n - a.score
            idx_b.append(torch.full((n - 1 - lo,), j, device='cuda'))
            idx_p.append(torch.arange(lo, n - 1, device='cuda'))
            tgt.append(x[j, lo + 1:n])
        idx_b = torch.cat(idx_b); idx_p = torch.cat(idx_p); tgt = torch.cat(tgt)
        hs_ = out.last_hidden_state[idx_b, idx_p]                 # [N,1024]
        for s in range(0, hs_.shape[0], 2000):
            lg = m.lm_head(hs_[s:s + 2000]).float()
            tot += float(torch.nn.functional.cross_entropy(lg, tgt[s:s + 2000], reduction='sum'))
            cnt += lg.shape[0]
    return tot / cnt


def attn_head(i, h):
    def make():
        def pre(mod, args):
            x = args[0].clone()
            x[..., h * 256:(h + 1) * 256] = 0
            return (x,)
        return H.layers[i].self_attn.o_proj.register_forward_pre_hook(pre)
    return make


def gdn_head(i, h):
    def make():
        def pre(mod, args):
            x = args[0].clone()
            x[..., h * 128:(h + 1) * 128] = 0
            return (x,)
        return H.layers[i].linear_attn.out_proj.register_forward_pre_hook(pre)
    return make


def mlp_group(i, g):
    def make():
        def pre(mod, args):
            x = args[0].clone()
            x[..., g * 256:(g + 1) * 256] = 0
            return (x,)
        return H.layers[i].mlp.down_proj.register_forward_pre_hook(pre)
    return make


def mlp_neuron(i, j):
    def make():
        def pre(mod, args):
            x = args[0].clone()
            x[..., j] = 0
            return (x,)
        return H.layers[i].mlp.down_proj.register_forward_pre_hook(pre)
    return make


def sub_attn(i):
    def make():
        def post(mod, args, out):
            if isinstance(out, tuple):
                return (torch.zeros_like(out[0]),) + tuple(out[1:])
            return torch.zeros_like(out)
        return (H.layers[i].self_attn if i in FL else H.layers[i].linear_attn).register_forward_hook(post)
    return make


def sub_mlp(i):
    def make():
        def post(mod, args, out):
            return torch.zeros_like(out)
        return H.layers[i].mlp.register_forward_hook(post)
    return make


configs = []
if a.part in ('heads', 'all'):
    configs += [(f'attn {i}.{h}', attn_head(i, h)) for i in FL for h in range(8)]
    configs += [(f'gdn {i}.{h}', gdn_head(i, h)) for i in LL for h in range(16)]
if a.part in ('mlp', 'all'):
    configs += [(f'mlp {i}.{g}', mlp_group(i, g)) for i in range(24) for g in range(14)]
if a.part in ('sub', 'all'):
    configs += [(f'sub-attn {i}', sub_attn(i)) for i in range(24)]
    configs += [(f'sub-mlp {i}', sub_mlp(i)) for i in range(24)]
if a.part == 'neuron':
    for item in a.neurons.split(','):
        lay, nrn = item.split(':')
        configs.append((f'neuron {lay}:{nrn}', mlp_neuron(int(lay), int(nrn))))

if a.part == 'check':
    # (1) padded batch == single sequence; (2) hooks act on the intended tensor only
    pid, ids = windows[0]
    x = torch.tensor([ids], device='cuda')
    out = m.model(input_ids=x)
    lo = len(ids) - a.score
    lg = m.lm_head(out.last_hidden_state[0, lo:len(ids) - 1]).float()
    single = float(torch.nn.functional.cross_entropy(lg, x[0, lo + 1:len(ids)]))
    x, mask, lens = bats[0]
    j = [w[0] for w in sorted(windows, key=lambda w: len(w[1]))[:8]].index(pid)
    assert lens[j] == len(ids)
    out = m.model(input_ids=x, attention_mask=mask)
    lg = m.lm_head(out.last_hidden_state[j, lo:len(ids) - 1]).float()
    pad = float(torch.nn.functional.cross_entropy(lg, x[j, lo + 1:len(ids)]))
    print(f'check: single {single:.5f} padded-batch {pad:.5f} (should agree to ~1e-3)')
    tot = sum(len(w[1]) for w in windows)
    b_all = evaluate()
    print(f'check: one config, full set: {b_all:.5f}')
    sys.exit(0)

t0 = time.time()
base = evaluate()
print(f'baseline NLL {base:.4f}  ({time.time()-t0:.1f} s for a full evaluation)', flush=True)
res = dict(base=base, n_windows=len(windows), model_dir=a.model_dir, seed=a.seed)
path = f'/home/user/wsp14/work/out/lesions_{a.tag}_{a.part}.json'
json.dump(res, open(path, 'w'))
for name, mk in configs:
    v = evaluate([mk])
    res[name] = v - base
    json.dump(res, open(path, 'w'))
    print(f'{name:14s} dNLL {v - base:+.4f}   [{time.time()-t0:.0f} s]', flush=True)
print('done')
