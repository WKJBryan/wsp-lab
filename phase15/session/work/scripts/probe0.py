"""Load the model in transformers on GPU; print module structure and the attention sources.

Teacher-forced read only (no generation). Usage: python probe0.py [model_dir]
"""
import sys, torch, inspect
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
torch.set_grad_enabled(False)
path = sys.argv[1] if len(sys.argv) > 1 else '/home/user/wsp14/sandbox/model'
tok = AutoTokenizer.from_pretrained(path)
m = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map='cuda')
m.eval()
print(type(m).__name__)
seen = set()
for n, mod in m.named_modules():
    if 'visual' in n:
        continue
    import re
    n2 = re.sub(r'layers\.\d+\.', 'layers.N.', n)
    if n2 in seen:
        continue
    seen.add(n2)
    print(n, '->', type(mod).__name__)
lm = m.model.language_model
print('lm_head', type(m.lm_head).__name__)
print('tied:', m.lm_head.weight.data_ptr() == lm.embed_tokens.weight.data_ptr())
import transformers.models.qwen3_5.modeling_qwen3_5 as Q
print(inspect.getsource(Q.Qwen3_5Attention.forward))
print(inspect.getsource(Q.Qwen3_5GatedDeltaNet.forward))
