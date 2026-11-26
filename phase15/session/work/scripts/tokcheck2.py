"""Tokenisation of the canonical ending with preceding newlines / punctuation."""
import sys
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(lib.MODEL_DIR)
for s in ['\n\\boxed{9}', '\n\n\\boxed{9}', '.\n\n\\boxed{9}', '\n\n\\', '\n\\', '\n\n\\[', '\n\\(', '\n\n$$\\boxed{9}$$', '\n\n**Answer:**\n\\boxed{9}', '$\n\\boxed{9}$']:
    ids = tok(s, add_special_tokens=False)['input_ids']
    print(f'{s!r:28s} -> {ids}  {[tok.convert_ids_to_tokens([t])[0] for t in ids]}')
