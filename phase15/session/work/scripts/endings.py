"""How do closed generations end? Print the last 700 chars of the thinking of every generation that
closed, and the text after </think>.  Usage: python endings.py <pass_no>"""
import sys, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
G = lib.gens(int(sys.argv[1]))
prac = {p['id']: p for p in lib.practice()}
for pid, g in G.items():
    if '</think>' in g['text']:
        th, after = g['text'].split('</think>', 1)
        print(f'=== {pid} (lvl {prac[pid]["level"]}, {g["n_tokens"]} tok, ok={g["ok_boxed"]}, gold={prac[pid]["gold"]!r}, pred={g["pred_boxed"]!r})')
        print('   ...' + repr(th[-700:]))
        print('   AFTER: ' + repr(after[:300]))
