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

E = model.get_input_embeddings().weight  # tied (248320, 1024) bf16


def prompt_for(pid):
    q = practice[pid]['q']
    content = 'Problem:\n' + q + '\n\nPlease reason step by step, and put your final answer within \\boxed{}.'
    return tok.apply_chat_template([{'role': 'user', 'content': content}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=True)


@torch.no_grad()
def token_stats(full_text, targets=(248069, 13784, 198)):
    """Per-position entropy, top-1 logprob, target logprobs. Hook on final norm, chunked logits."""
    captured = {}
    norm_mod = model.model.norm
    h = norm_mod.register_forward_hook(lambda m, inp, out: captured.__setitem__('h', out.detach()))
    ids = tok(full_text, return_tensors='pt').input_ids[0].to('cuda')
    model(ids.unsqueeze(0))
    h.remove()
    H = captured['h'][0]  # (n, 1024) bf16 — final-norm output, directly pre-lm_head
    n = H.shape[0]
    ent = torch.empty(n, device='cuda')
    top1 = torch.empty(n, device='cuda')
    tl = {t: torch.empty(n, device='cuda') for t in targets}
    CH = 2048
    for s in range(0, n, CH):
        hc = H[s:s + CH].float()
        logits = hc @ E.float().T  # (ch, V) fp32
        lp = F.log_softmax(logits, -1)
        ent[s:s + CH] = -(lp.exp() * lp.clamp_min(-30)).sum(-1)
        top1[s:s + CH] = lp.max(-1).values
        for t in targets:
            tl[t][s:s + CH] = lp[:, t]
        del logits, lp
    return ids, ent, top1, tl


targets = {'</think>': 248069, ' Wait': 13784, '\\n': 198}

print('=== entropy trajectory of failed generations (512-token windows) ===')
for pid in ['p1', 'p2']:
    r = byid[pid]
    p = prompt_for(pid)
    ids, ent, top1, tl = token_stats(p + r['text'])
    n = ids.shape[0]
    plen = len(tok(p).input_ids)
    print(f'--- {pid}: total tokens {n} (gen {n - plen})')
    W = 512
    for s in range(plen, n - 1, W):
        e = min(s + W, n - 1)
        print(f'  pos {s - plen:6d}-{e - plen:6d}: entropy {ent[s:e].mean():6.2f}  top1lp {top1[s:e].mean():6.2f}  '
              f'P(</think>) {tl[248069][s:e].mean():7.2f}  P( Wait) {tl[13784][s:e].mean():6.2f}  '
              f'text: {tok.decode(ids[s:s + 6]).replace(chr(10), "|n")[:50]!r}')

print()
print('=== exact closing moment in successful generations ===')
for pid in ['p124', 'p28', 'p101', 'p27']:
    r = byid[pid]
    t = r['text']
    idx = t.find('</think>')
    if idx < 0:
        print(pid, 'no </think>')
        continue
    pre = t[:idx]
    ids, ent, top1, tl = token_stats(prompt_for(pid) + pre)
    lt = float(tl[248069][-1])
    print(f'{pid}: P(</think>) at the true closing position = {lt:6.2f}, entropy there {float(ent[-1]):.2f}, '
          f'context tail: {pre[-70:]!r}')
    # the 64 tokens before closing
    print(f'   last 64 pre-close tokens: entropy {ent[-64:].mean():.2f}, top1lp {top1[-64:].mean():.2f}, '
          f'P(</think>) mean {tl[248069][-64:].mean():.2f}')
