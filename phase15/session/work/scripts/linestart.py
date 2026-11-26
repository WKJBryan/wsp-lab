"""Collect the residual stream at line-start decision positions of every pass-1 trace.

A decision position is the token right before the first real token of a line (after a newline and
any indentation / bullet). Labels kept per position:
  region : 'early' (pre-loop, more than 2500 tokens before the loop), 'late' (last 2500 tokens
           before the loop), 'loop' (inside the verbatim loop; first 2500 tokens of it)
  first  : the next token (first token of the line)
  p_top  : probability of the model's own top-1 token at that position
  dup    : 1 if the next 12 tokens are an exact copy of the 12 tokens that followed the previous
           line-start position (the line being started duplicates the previous line)
  same   : 1 if this line's first token equals the previous line's first token
Contexts are truncated just after the last collected position.
Usage: python linestart.py [--pass 1] [--save out/linestart_1.pt]
"""
import sys, torch, argparse, json, random, re
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
ap = argparse.ArgumentParser()
ap.add_argument('--pass', dest='pass_no', type=int, default=1)
ap.add_argument('--model_dir', default=lib.MODEL_DIR)
ap.add_argument('--save', default='/home/user/wsp14/work/out/linestart_1.pt')
ap.add_argument('--looplen', type=int, default=2500)
ap.add_argument('--latelen', type=int, default=2500)
a = ap.parse_args()
m, tok = lib.load(a.model_dir)
prac = {p['id']: p for p in lib.practice()}
G = lib.gens(a.pass_no)
loops = json.load(open(f'/home/user/wsp14/work/out/loops_{a.pass_no}.json'))
dec = lambda t: tok.decode([t], skip_special_tokens=False, clean_up_tokenization_spaces=False)
SPACE = re.compile(r'^[\s\*\-\u2022]*$')
_id_cache = {}


def real_start(ids, i):
    while i < len(ids) and SPACE.match(dec(ids[i])):
        i += 1
    return i if i < len(ids) else None


def line_starts(ids):
    out, i = [], 0
    while i < len(ids):
        if '\n' in dec(ids[i]):
            j = real_start(ids, i + 1)
            if j is not None and j > i:
                out.append(j - 1)      # position predicting token j
            i = j if j is not None else i + 1
        else:
            i += 1
    return out


rng = random.Random(0)
data = {}
pids = sorted(G, key=lambda s: int(s[1:]))
for n, pid in enumerate(pids):
    g = G[pid]
    p_ids = lib.prompt_ids(tok, prac[pid]['q'])
    ids = p_ids + lib.ids_of(tok, g['text'])
    ls = loops[pid]
    s = ls['loop_start']
    P0 = len(p_ids)
    if s:
        cut = min(len(ids), P0 + s + a.looplen + 14)
    else:
        cut = min(len(ids), P0 + g['n_tokens'] + 1)
    starts_all = line_starts(ids[:cut])
    starts = [p for p in starts_all if P0 + 20 <= p < cut - 14]
    if s:
        loop = [p for p in starts if p >= P0 + s]
        late = [p for p in starts if P0 + s - a.latelen <= p < P0 + s]
        early = [p for p in starts if p < P0 + s - a.latelen]
        early = rng.sample(early, min(100, len(early)))
    else:
        loop = []
        late = [p for p in starts if p >= len(ids) - a.latelen]
        early = [p for p in starts if p < len(ids) - a.latelen]
        early = rng.sample(early, min(100, len(early)))
    pos = sorted(set(loop + late + early))
    ids = ids[:max(pos) + 14]
    r = lib.run_pos(m, ids, pos)
    L = r['logits']
    top2 = torch.topk(L, 2, -1)
    p_top = torch.exp(top2.values[:, 0] - torch.logsumexp(L, -1))
    prev_of = {starts[k]: starts[k - 1] for k in range(1, len(starts))}
    rec = []
    for k, p in enumerate(pos):
        region = 'loop' if p in loop else 'late' if p in late else 'early'
        nxt = ids[p + 1]
        dup = same = 0
        q = prev_of.get(p)
        if q is not None and p - q <= 400:
            same = int(ids[q + 1] == nxt)
            dup = int(ids[p + 1:p + 13] == ids[q + 1:q + 13])
        rec.append((region, nxt, float(p_top[k]), int(top2.indices[k, 0] == nxt), dup, same, int(top2.values[k, 0] - top2.values[k, 1])))
    data[pid] = dict(pos=torch.tensor(pos), resid=r['resid'], region=[x[0] for x in rec], first=torch.tensor([x[1] for x in rec]),
                     p_top=torch.tensor([x[2] for x in rec]), argmax=torch.tensor([x[3] for x in rec]), dup=torch.tensor([x[4] for x in rec]),
                     same=torch.tensor([x[5] for x in rec]), margin=torch.tensor([x[6] for x in rec]), cut=max(pos) + 14)
    if n % 16 == 0:
        print(f'  {n}/{len(pids)} traces; last: T={len(ids)} {len(pos)} line starts ({len(loop)} loop, {len(late)} late, {len(early)} early)', flush=True)
torch.save(data, a.save)
tot = {}
for d in data.values():
    for reg in d['region']:
        tot[reg] = tot.get(reg, 0) + 1
print('saved', a.save, 'positions by region:', tot)
