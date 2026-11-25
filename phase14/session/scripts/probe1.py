# Teacher-forced probes: simple arithmetic + closing behaviour. Run only when GPU is free.
import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

model_dir = sys.argv[1] if len(sys.argv) > 1 else '/home/user/wsp14/sandbox/model'
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map='cuda')
model.eval()

def ask(text, n_top=6):
    ids = tok(text, return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = model(**ids)
    logits = out.logits[0, -1].float()
    logp = F.log_softmax(logits, dim=-1)
    top = torch.topk(logp, n_top)
    toks = [tok.decode([i]) for i in top.indices.tolist()]
    return [(t.replace(chr(10), '\n'), round(v.item(), 3)) for t, v in zip(toks, top.values)]

CHATS = [
  'Compute 47 + 36.',
  'Compute 82 - 47.',
  'Compute 13 * 17.',
  'Compute 144 / 12.',
  'Compute 7^3.',
  'What is 15% of 80?',
  'Compute 3/4 + 2/5 (as a fraction).',
  'Compute -8 + 19.',
  'Compute 2^10.',
  'Compute sqrt(144).',
]
for c in CHATS:
    msgs = [{'role':'user','content':'Problem:\n' + c + '\n\nPlease reason step by step, and put your final answer within \boxed{}.'}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=True)
    print('Q:', c, ' -> top:', ask(text))
