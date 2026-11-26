"""Generations that ENDED (finish == stop) without a boxed answer: how do they end?
Usage: python nobox.py <gen.jsonl> [chars 700]"""
import sys, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 700
G = [json.loads(l) for l in open(path, encoding='utf-8')]
prac = {p['id']: p for p in lib.practice()}
BOX = chr(92) + 'boxed'
rows = [g for g in G if g['finish'] == 'stop' and BOX not in g['text']]
print(f'{len(rows)} of {sum(1 for g in G if g["finish"] == "stop")} ended generations contain no \\boxed at all')
for g in rows:
    t = g['text']
    th, post = (t.split('</think>', 1) + [''])[:2] if '</think>' in t else (t, '')
    print(f'\n=== {g["id"]} [{prac[g["id"]]["level"][-1]}] {g["n_tokens"]} tok, gold {prac[g["id"]]["gold"]!r}; {t.count("</think>")} </think>; final phase {len(post)} chars')
    print('   end of text: ' + repr(t[-n:]))
