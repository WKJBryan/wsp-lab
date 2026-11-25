"""Which branches write the anti-progress direction at loop positions?
Attribute logit(top1)-logit(alt) for alt in {' So', '</think>', ' Let'} per branch."""
import json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = '/home/user/wsp14/work/model_try1'
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map='cuda')
model.eval()

rows = [json.loads(l) for l in open('/home/user/wsp14/sandbox/harness/state/gen_1.jsonl')]
practice = {p['id']: p for p in json.load(open('/home/user/wsp14/sandbox/practice.json'))}
byid = {r['id']: r for r in rows}
E = model.get_input_embeddings().weight.float()


def prompt_for(pid):
    q = practice[pid]['q']
    content = 'Problem:\n' + q + '\n\nPlease reason step by step, and put your final answer within \\boxed{}.'
    return tok.apply_chat_template([{'role': 'user', 'content': content}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=True)


CTXS = {}
for pid in ['p1', 'p5']:
    t = byid[pid]['text']
    CTXS[pid] = tok(prompt_for(pid) + t).input_ids[-2200:]

cap = {}
HANDLES = []


def mk_hook(name):
    def hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        cap.setdefault(name, []).append(o[0, -1].detach().float())
    return hook


for L in range(24):
    mod = model.model.layers[L]
    attn = mod.self_attn if hasattr(mod, 'self_attn') else mod.linear_attn
    HANDLES.append(attn.register_forward_hook(mk_hook(f'L{L}.attn')))
    HANDLES.append(mod.mlp.register_forward_hook(mk_hook(f'L{L}.mlp')))

ALTS = {' So': None, '</think>': 248069, ' Okay': None, ' But': None, ' Let': None}
for k in ALTS:
    if ALTS[k] is None:
        ids = tok.encode(k, add_special_tokens=False)
        assert len(ids) == 1, (k, ids)
        ALTS[k] = ids[0]

with torch.no_grad():
    for pid, ids in CTXS.items():
        cap.clear()
        x = torch.tensor([ids]).cuda()
        logits = model(x).logits[0, -1].float()
        print(f'=== {pid} loop position ===')
        for altname, altid in ALTS.items():
            d = E[ALTS[' So'] if False else altid]
            # contribution of each branch to logit(top1)-logit(alt)
            lp = F.log_softmax(logits, -1)
            v, i = torch.topk(lp, 1)
            top1 = int(i[0])
            dd = E[top1] - E[altid]
            vals = []
            capd = {name: torch.stack(vs).mean(0) for name, vs in cap.items()}
            for name, o in capd.items():
                vals.append((float(o @ dd), name))
            vals.sort(key=lambda z: -abs(z[0]))
            negs = [(c, n) for c, n in vals if c < 0][:4]
            print(f'  vs {altname!r}: gap {float(v[0] - lp[altid]):6.2f} | top writers: ' +
                  ' '.join(f'{n}:{c:+.2f}' for c, n in vals[:4]) + ' | anti-' + altname + ' writers: ' +
                  ' '.join(f'{n}:{c:+.2f}' for c, n in negs))
for h in HANDLES:
    h.remove()
