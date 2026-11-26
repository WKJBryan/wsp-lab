"""How does the next-token distribution at a loop's cycle boundary change with the cycle count?

For each of several traces: one forward pass over the whole text; at the last token of cycle c-1
(predicting the first token of cycle c) print the top candidates with logits and probabilities
(temperature 0.6), plus the probability mass on 'wrap-up' line starters.

Usage: python cycle.py <pass_no> <pid,pid,...> [--model_dir DIR]
"""
import sys, torch, argparse, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
ap = argparse.ArgumentParser()
ap.add_argument('pass_no', type=int); ap.add_argument('pids')
ap.add_argument('--model_dir', default=lib.MODEL_DIR)
ap.add_argument('--cycles', default='1,2,3,4,6,8,12,16,32,64,200,800')
ap.add_argument('--temp', type=float, default=0.6)
a = ap.parse_args()
m, tok = lib.load(a.model_dir)
prac = {p['id']: p for p in lib.practice()}
loops = json.load(open(f'/home/user/wsp14/work/out/loops_{a.pass_no}.json'))
G = lib.gens(a.pass_no)
dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
cycles = [int(c) for c in a.cycles.split(',')]
WRAP = ['I', 'I\u0027ll', 'I\u0027m', 'Therefore', 'Thus', 'Okay', 'The', 'Final', 'Ready', 'So', 'Now', 'Let', 'Wait', 'But', 'However', 'This', 'Is']
WRAP_IDS = {w: tok(w, add_special_tokens=False)['input_ids'] for w in WRAP}
for pid in a.pids.split(','):
    g = G[pid]
    s, P = loops[pid]['loop_start'], loops[pid]['period']
    p_ids = lib.prompt_ids(tok, prac[pid]['q'])
    ids = p_ids + lib.ids_of(tok, g['text'])
    T = len(ids)
    cs = [c for c in cycles if s + c * P - 1 < T - 1]
    pos = [len(p_ids) + s + c * P - 1 for c in cs]
    r = lib.run(m, ids, positions=pos)
    L = r['logits']
    print(f'\n=== {pid}: loop starts at gen token {s}, period {P} tokens; cycle text {dec(ids[len(p_ids)+s:len(p_ids)+s+P])!r}')
    first = ids[len(p_ids) + s]           # first token of the cycle
    for n, c in enumerate(cs):
        l = L[n] / a.temp
        pr = torch.softmax(l, 0)
        v, i = torch.topk(L[n], 5)
        tops = ' '.join(f'{dec([int(t)])!r}:{float(x):.1f}({float(pr[t]):.2f})' for t, x in zip(i, v))
        mass = ' '.join(f'{w}:{float(sum(float(pr[t]) for t in WRAP_IDS[w])):.1e}' for w in WRAP if w != dec([first]).strip())
        print(f'  cycle {c:4d} (ctx {pos[n]} tok): next={dec([first])!r} logit {float(L[n, first]):.1f} p={float(pr[first]):.3f} | top: {tops}')
        if c in (1, 2, 4, 16, 800):
            print(f'           mass on line starters: {mass}')
