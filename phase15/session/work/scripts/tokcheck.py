"""Exact tokenisation of the strings that matter (no shell escaping involved)."""
import sys
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(lib.MODEL_DIR)
strings = [
    ' \\boxed{7}', '\\boxed{7}', '$\\boxed{7}$', '\\(\\boxed{7}\\)', 'The final answer is \\(\\boxed{7}\\).',
    'The answer is $\\boxed{7}$.', 'The final answer is \\boxed{7}.', '**Final Answer:** \\(\\boxed{7}\\)',
    '\\[\\boxed{7}\\]', '\\sum x', '\\frac{1}{2}', ' \\(', '\\(', '\\', '\\,', '}\\)', '}.', '}$',
]
for s in strings:
    ids = tok(s, add_special_tokens=False)['input_ids']
    print(f'{s!r:45s} -> {ids}  {[tok.convert_ids_to_tokens([t])[0] for t in ids]}')
