import torch, sys
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
for d in ['/home/user/wsp14/work/model_pristine', '/home/user/wsp14/work/cand_A']:
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16, device_map='cuda')
    model.eval()
    prompts = [
        'The capital of France is',
        '2 + 2 =',
        'Q: What is 17 * 24? A:',
        'def fibonacci(n):',
    ]
    print('===', d.split('/')[-1])
    for p in prompts:
        ids = tok(p, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id)
        print('  ', p, '->', repr(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))[:90])
    del model
    torch.cuda.empty_cache()
