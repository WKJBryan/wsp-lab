"""Early closures in a pass: first </think> before a given number of generated tokens.
Usage: python earlyclosed.py <gen.jsonl> [max_tokens 2500]"""
import sys, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
path = sys.argv[1]; lim = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
G = [json.loads(l) for l in open(path, encoding='utf-8')]
prac = {p['id']: p for p in lib.practice()}
BOX = chr(92) + 'boxed'
rows = []
for g in G:
    t = g['text']
    k = t.find('</think>')
    if k < 0:
        continue
    ktok = int(len(t[:k]) / max(1, len(t)) * g['n_tokens'])
    if ktok < lim:
        rows.append((ktok, g['id'], prac[g['id']]['level'][-1], g['ok_boxed'], str(g['pred_boxed'])[:24], prac[g['id']]['gold'][:24], t.count('</think>'), BOX in t, g['finish']))
rows.sort()
print(f'{len(rows)} traces whose thinking first closed before {lim} tokens; correct {sum(1 for r in rows if r[3])}')
for r in rows:
    print(f'  {r[1]:5s} L{r[2]} closed at {r[0]:5d} tok ok={r[3]!s:5s} pred {r[4]:24s} gold {r[5]:24s} #</think> {r[6]:2d} box {r[7]!s:5s} {r[8]}')
