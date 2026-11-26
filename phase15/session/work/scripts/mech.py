"""Mechanism-level indicators of a pass (less noisy than accuracy alone).
Usage: python mech.py <gen.jsonl> [<gen.jsonl> ...]"""
import sys, json, statistics
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
prac = {p['id']: p for p in lib.practice()}
BOX = chr(92) + 'boxed'
for path in sys.argv[1:]:
    G = [json.loads(l) for l in open(path, encoding='utf-8')]
    first = []
    for g in G:
        t = g['text']
        k = t.find('</think>')
        first.append(int(len(t[:k]) / max(1, len(t)) * g['n_tokens']) if k >= 0 else None)
    closed = [f for f in first if f is not None]
    print(f'{path.split("/")[-1]:14s} correct {sum(g["ok_boxed"] for g in G):3d}/128 | with a boxed answer {sum(1 for g in G if BOX in g["text"]):3d} | '
          f'closed {len(closed):3d} | closed after 20k tokens {sum(1 for f in closed if f >= 20000):2d} | after 12k {sum(1 for f in closed if f >= 12000):2d} | '
          f'median first closure {int(statistics.median(closed)):5d} tok | never closed {sum(1 for f in first if f is None)} | capped {sum(1 for g in G if g["finish"] == "length"):2d} | '
          f'ended w/o box {sum(1 for g in G if g["finish"] == "stop" and BOX not in g["text"]):2d} | mean tokens {int(statistics.mean(g["n_tokens"] for g in G)):5d}')
