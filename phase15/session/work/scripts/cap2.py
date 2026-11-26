"""Breakdown of the capped (finish == length) generations of a pass.

Usage: python cap2.py <gen.jsonl>
"""
import sys, json, re
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
from transformers import AutoTokenizer
path = sys.argv[1]
G = [json.loads(l) for l in open(path, encoding='utf-8')]
prac = {p['id']: p for p in lib.practice()}
tok = AutoTokenizer.from_pretrained(lib.MODEL_DIR)
closed = [g for g in G if '</think>' in g['text']]
capped = [g for g in G if g['finish'] == 'length']
print(f'{len(G)} generations: {len(closed)} contain </think> (some several times); {len(capped)} hit the cap; '
      f'{sum(1 for g in capped if "</think>" in g["text"])} of the capped contain </think>; {sum(1 for g in capped if "</think>" not in g["text"])} never closed')
print('correct among stopped:', sum(1 for g in G if g['finish'] == 'stop' and g['ok_boxed']), 'of', sum(1 for g in G if g['finish'] == 'stop'),
      '| correct among capped:', sum(1 for g in capped if g['ok_boxed']), 'of', len(capped))
n_multi = sum(1 for g in G if g['text'].count('</think>') > 1)
print('generations with more than one </think>:', n_multi)
print('\nid   lvl closed? #</think> think_chars  total_chars | n( Wait) after </think> | boxed after? | last 160 chars')
for g in capped:
    t = g['text']
    th = t.split('</think>')
    think_len = len(th[0]) if len(th) > 1 else len(t)
    post = ''.join(th[1:]) if len(th) > 1 else ''
    print(f'{g["id"]:4s} {prac[g["id"]]["level"][-1]}  {str(len(th) > 1):6s}  {t.count("</think>"):3d}      {think_len:6d}     {len(t):6d}   | {post.count("Wait"):5d}  | {"Y" if chr(92) + "boxed" in post else "-"} | {t[-160:]!r}')
