"""Print the text around a token position of a generation.

Usage: python ctx.py <pass_no> <problem_id> <center_gen_token> [before=250] [after=120]
(token index counted in GENERATED tokens, the prompt excluded)
"""
import sys
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
from transformers import AutoTokenizer
pass_no, pid, c = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
before = int(sys.argv[4]) if len(sys.argv) > 4 else 250
after = int(sys.argv[5]) if len(sys.argv) > 5 else 120
tok = AutoTokenizer.from_pretrained(lib.MODEL_DIR)
g = lib.gens(pass_no)[pid]
ids = lib.ids_of(tok, g['text'])
lo, hi = max(0, c - before), min(len(ids), c + after)
pre = tok.decode(ids[lo:c], skip_special_tokens=False)
post = tok.decode(ids[c:hi], skip_special_tokens=False)
print(f'--- {pid} gen token {c} (of {len(ids)}); showing [{lo}, {hi}) ---')
print(pre + ' <<<<< HERE >>>>> ' + post)
