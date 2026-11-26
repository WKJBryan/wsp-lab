"""What had the model concluded just before its loop started? Print the last ~450 characters of
the pre-loop reasoning and the gold answer.  Usage: python preloop.py <pass_no> [max count]"""
import sys, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
from transformers import AutoTokenizer
pass_no = int(sys.argv[1])
cnt = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
tok = AutoTokenizer.from_pretrained(lib.MODEL_DIR)
G = lib.gens(pass_no)
prac = {p['id']: p for p in lib.practice()}
loops = json.load(open(f'/home/user/wsp14/work/out/loops_{pass_no}.json'))
n = 0
for pid in sorted(G, key=lambda s: int(s[1:])):
    if not loops[pid]['loop_start']:
        continue
    n += 1
    if n > cnt:
        break
    g = G[pid]
    ids = lib.ids_of(tok, g['text'])
    s = loops[pid]['loop_start']
    pre = tok.decode(ids[max(0, s - 110):s], skip_special_tokens=False)
    cyc = tok.decode(ids[s:s + loops[pid]['period']], skip_special_tokens=False)
    print(f'=== {pid} [{prac[pid]["level"][-1]}/{prac[pid]["type"][:14]}] gold={prac[pid]["gold"]!r} loop at tok {s}, period {loops[pid]["period"]}')
    print('   ...' + repr(pre))
    print('   LOOP: ' + repr(cyc[:110]))
