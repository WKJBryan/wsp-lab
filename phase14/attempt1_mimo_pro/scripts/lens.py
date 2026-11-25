"""Layerwise logit-lens + direct logit attribution at chosen offsets of a generation.
Usage: lens.py <gen_id> <char_offset> [<char_offset>...]
For each offset: run the prefix teacher-forced, then for every layer read out
(top tokens via final-norm + lm_head) and the logit of tokens of interest, plus
per-layer attention/MLP contributions to those logits (DLA)."""
import json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GEN_ID = sys.argv[1]
OFFSETS = [int(x) for x in sys.argv[2:]]
MD = '/home/user/wsp14/sandbox/model'
REF_Q = "Problem:\n{q}\n\nPlease reason step by step, and put your final answer within \\boxed{}."

tok = AutoTokenizer.from_pretrained(MD)
model = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16, device_map='cuda')
model.eval()
lm = model.model if hasattr(model.model, 'layers') else model.model.language_model
final_norm = model.model.norm if hasattr(model.model, 'norm') else model.model.language_model.norm
W = model.get_input_embeddings().weight  # tied with lm_head

prac = {p['id']: p for p in json.load(open('/home/user/wsp14/sandbox/practice.json', encoding='utf-8'))}
rows = {r['id']: r for r in (json.loads(l) for l in open('/home/user/wsp14/sandbox/harness/state/gen_1.jsonl', encoding='utf-8'))}
p = prac[GEN_ID]
text = rows[GEN_ID]['text']

prompt = tok.apply_chat_template([{'role': 'user', 'content': REF_Q.replace('{q}', p['q'])}],
                                 add_generation_prompt=True, tokenize=False, enable_thinking=True)
prompt_ids = tok(prompt, return_tensors='pt').input_ids[0]

INTEREST = {'think_end': 248069}
for lit in [' Wait', ' If', ' Is', ' Let', ' Maybe', ' The', ' So', ' We', ' Note', ' This', ' But', ' Okay']:
    tid = tok(lit, add_special_tokens=False).input_ids
    if len(tid) == 1:
        INTEREST[lit] = tid[0]

for off in OFFSETS:
    prefix = text[:off]
    gen_ids = tok(prefix, add_special_tokens=False, return_tensors='pt').input_ids[0]
    ids = torch.cat([prompt_ids, gen_ids]).unsqueeze(0).to('cuda')
    T = ids.shape[1]
    attn_out = {}
    mlp_out = {}
    handles = []
    for i, layer in enumerate(lm.layers):
        def mk_a(i):
            def hook(m, inp, out):
                attn_out[i] = (out[0] if isinstance(out, tuple) else out)[0, -1].detach().float()
            return hook
        def mk_m(i):
            def hook(m, inp, out):
                mlp_out[i] = (out[0] if isinstance(out, tuple) else out)[0, -1].detach().float()
            return hook
        # decoder layer: attention and mlp submodules
        handles.append(layer.self_attn if hasattr(layer, 'self_attn') else layer.linear_attn)
        sub_attn = layer.self_attn if hasattr(layer, 'self_attn') else layer.linear_attn
        handles.append(sub_attn.register_forward_hook(mk_a(i)))
        handles.append(layer.mlp.register_forward_hook(mk_m(i)))
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    for h in handles:
        if hasattr(h, 'remove'):
            h.remove()
    hs = out.hidden_states  # 25 x [1, T, d]
    print(f'===== {GEN_ID} offset={off} (T={T}) =====')
    print('context tail:', repr(prefix[-100:]))
    print('-- logit lens per layer (top5 at last position) --')
    for L in range(len(hs)):
        h = hs[L][0, -1].float()
        logits = final_norm(h.unsqueeze(0).to(W.dtype)) @ W.T
        logits = logits[0].float()
        top = torch.topk(logits, 5)
        toks = ' | '.join(f'{tok.decode([int(i)])!r}:{float(v):.2f}' for v, i in zip(top.values, top.indices))
        ints = ' '.join(f'{n}={float(logits[t]):.1f}' for n, t in INTEREST.items())
        print(f'  L{L:2d} |h|={float(h.norm()):6.2f} {toks}   [{ints}]')
    print('-- DLA: per-layer attn/mlp contribution to interest logits (u_v . comp) --')
    for L in sorted(attn_out):
        a = attn_out[L].to(W.dtype).float()
        m = mlp_out[L].to(W.dtype).float()
        parts = []
        for n, t in INTEREST.items():
            u = W[t].float()
            parts.append(f'{n}: attn={float(u @ a):+6.2f} mlp={float(u @ m):+6.2f}')
        print(f'  L{L:2d} |attn|={float(a.norm()):5.2f} |mlp|={float(m.norm()):5.2f}  ' + '  '.join(parts))
    print()
