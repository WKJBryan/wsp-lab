import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "/home/user/wsp4/sandbox/model"
THINK_END = 248069
IM_END = 248046
TEMP = 0.6
TOP_P = 0.95
TOP_K = 20
PRESENCE = 1.5
ALPHAS = [0.0, 0.005, 0.01, 0.02]
MAXSTEPS = 1600
POSTFIRE_CAP = 250

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
problems = json.load(open("/home/user/wsp4/sandbox/practice.json"))
sel = [problems[0], problems[1], problems[6]]

def one_roll(alpha, p, maxsteps):
    text = tok.apply_chat_template([[{"role": "user", "content": p["q"]}]],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=True)
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    gen_counts = {}
    fire = None
    postfire = []
    torch.manual_seed(20261105)
    with torch.no_grad():
        out = model(ids, use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        next_tok = None
        for step in range(maxsteps):
            if next_tok is not None:
                out = model(next_tok, past_key_values=past, use_cache=True, output_hidden_states=True)
                past = out.past_key_values
            logits = out.logits[0, -1].float()
            hsum = out.hidden_states[-1][0, -1].float().sum().item()
            lg = logits.clone()
            if gen_counts:
                idxs = torch.tensor(list(gen_counts.keys()), device=lg.device, dtype=torch.long)
                lg[idxs] -= PRESENCE
            lg[THINK_END] += alpha * hsum
            lg /= TEMP
            topv, topi = torch.topk(lg, TOP_K)
            probs = torch.softmax(topv, -1)
            cum = torch.cumsum(probs, -1)
            cutoff = int((cum < TOP_P).sum().item())
            probs = probs[: cutoff + 1]
            probs = probs / probs.sum()
            t = topi[torch.multinomial(probs, 1).item()].item()
            gen_counts[t] = gen_counts.get(t, 0) + 1
            if t == THINK_END and fire is None:
                fire = step
            elif fire is not None:
                postfire.append(t)
            if t == IM_END:
                break
            if fire is not None and len(postfire) >= POSTFIRE_CAP:
                break
            next_tok = torch.tensor([[t]], device="cuda")
    return fire, postfire

res = {}
plan = [(0.0, sel[0], 800)] + [(a, p, MAXSTEPS) for a in ALPHAS[1:] for p in sel]
for a, p, ms in plan:
    fire, postfire = one_roll(a, p, ms)
    n_spam = sum(1 for t in postfire if t == THINK_END)
    tail = tok.decode(postfire[:60])
    res[f"{a}|{p['id']}"] = dict(fire=fire, n_postfire=len(postfire), spam=n_spam)
    print("ALPHA", a, p["id"], "fire_step", fire, "spam", n_spam, "/", len(postfire), flush=True)
    print("   postfire_head", repr(tail[:150]), flush=True)
json.dump(res, open("/home/user/wsp4/sandbox/sim_results.json", "w"), indent=1)
print("done")
