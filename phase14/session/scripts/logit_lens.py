import json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = '/home/user/wsp14/sandbox/model'
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


@torch.no_grad()
def layer_loglens(full_text, positions):
    """For each layer (0..24), the logprob of top-1 and of </think> at given gen positions."""
    out = model(tok(full_text, return_tensors='pt').input_ids.to('cuda'), output_hidden_states=True)
    hs = out.hidden_states  # 25 tuples: emb, after layer0, ..., after layer23
    n = hs[0].shape[1]
    res = {}
    for L, h in enumerate(hs):
        hf = model.model.norm(h[0]).float()  # apply final norm (logit-lens approximation)
        for (pos, tag) in positions:
            lg = hf[pos] @ E.T
            lp = F.log_softmax(lg, -1)
            v, i = torch.topk(lp, 3)
            res[(L, pos, tag)] = (
                [tok.decode([j]) for j in i.tolist()],
                [round(float(x), 1) for x in v.tolist()],
                round(float(lp[248069]), 1),
            )
    return res


# p1: collapse happened around gen pos ~8600 (entropy 0.28 -> 0.00). Inspect layers there and at healthy positions.
r = byid['p1']
p = prompt_for(r['id'])
full = p + r['text']
ptok = tok(p, return_tensors='pt').input_ids.shape[1]
alltok = tok(full, return_tensors='pt').input_ids.shape[1]
positions = []
for (gpos, tag) in [(300, 'start'), (5000, 'healthy'), (8400, 'pre-collapse'), (8900, 'collapsed'), (15000, 'loop')]:
    positions.append((min(ptok + gpos, alltok - 2), tag))
res = layer_loglens(full, positions)
print('p1: per-layer logit lens (top3 tokens with logprobs, and logprob of </think>)')
hdr = 'layer | ' + ' | '.join(f'{tag}@{gp}' for (gp, tag) in positions)
print(hdr)
for L in range(25):
    cells = []
    for (gp, tag) in positions:
        tops, lps, lt = res[(L, gp, tag)]
        cells.append(f"{tops[0]!r}:{lps[0]:5.1f} th:{lt:5.1f}")
    print(f'{L:5d} | ' + ' | '.join(cells))
