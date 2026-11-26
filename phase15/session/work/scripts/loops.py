"""Save, per problem of a pass, where the verbatim loop starts (token index in generated text).

Usage: python loops.py <pass_no> [--save out/loops_<pass>.json]
"""
import sys, json, argparse
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
from transformers import AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument('pass_no', type=int)
ap.add_argument('--K', type=int, default=1200)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(lib.MODEL_DIR)
prac = {p['id']: p for p in lib.practice()}
G = lib.gens(a.pass_no)
res = {}
print('id   lvl  gen_tok  finish  loop_start(frac)  period  pre-loop Waits | first 60 chars of the cycle')
for pid in sorted(G, key=lambda s: int(s[1:])):
    g = G[pid]
    ids = lib.ids_of(tok, g['text'])
    n = len(ids)
    K = min(a.K, n // 2)
    period = None
    for p in range(1, 401):
        if all(ids[i] == ids[i - p] for i in range(n - K, n)):
            period = p
            break
    if period is None:
        res[pid] = dict(loop_start=None, period=None, n=n, finish=g['finish'])
        print(f'{pid:4s} {prac[pid]["level"][-1]}  {g["n_tokens"]:6d}  {g["finish"]:6s}  no exact period in last {K} tokens')
        continue
    start = n - K
    while start > 0 and ids[start - 1] == ids[start - 1 + period]:
        start -= 1
    pre = lib.text_of(tok, ids[:start])
    cyc = lib.text_of(tok, ids[start:start + period])
    res[pid] = dict(loop_start=start, period=period, n=n, finish=g['finish'], waits=pre.count('Wait'))
    print(f'{pid:4s} {prac[pid]["level"][-1]}  {g["n_tokens"]:6d}  {g["finish"]:6s}  {start:6d} ({start/n:.2f})  {period:5d}  {pre.count("Wait"):5d} | {cyc[:60]!r}')
json.dump(res, open(f'/home/user/wsp14/work/out/loops_{a.pass_no}.json', 'w'))
