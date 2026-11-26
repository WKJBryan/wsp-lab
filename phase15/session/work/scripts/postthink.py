"""Show how the thinking ended and what followed </think> in pass-N generations.
Usage: python postthink.py <gen.jsonl> <id,id,...> [before 500] [after 1500]"""
import sys, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
path, pids = sys.argv[1], sys.argv[2].split(',')
before = int(sys.argv[3]) if len(sys.argv) > 3 else 500
after = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
G = {json.loads(l)['id']: json.loads(l) for l in open(path, encoding='utf-8')}
prac = {p['id']: p for p in lib.practice()}
for pid in pids:
    g = G[pid]
    t = g['text']
    print(f'\n==================== {pid} [{prac[pid]["level"][-1]}/{prac[pid]["type"][:18]}] {g["n_tokens"]} tok finish={g["finish"]} ok={g["ok_boxed"]} gold={prac[pid]["gold"]!r} pred={g["pred_boxed"]!r}')
    print('PROBLEM:', prac[pid]['q'][:300].replace('\n', ' '))
    if '</think>' in t:
        th, post = t.split('</think>', 1)
        print(f'--- thinking ends ({len(t)} chars total, {len(post)} after </think>): ...{th[-before:]!r}')
        print(f'--- after </think> (first {after}): {post[:after]!r}')
        if len(post) > after + 600:
            print(f'--- after </think> (LAST 600): {post[-600:]!r}')
    else:
        print(f'--- no </think>; tail: {t[-1200:]!r}')
