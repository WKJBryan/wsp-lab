import json, time, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MD = "/home/bladless/wsp4/sandbox/model"
tok = AutoTokenizer.from_pretrained(MD)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.float32)
model.eval()
print("load_s", round(time.time() - t0, 1), flush=True)

items = json.load(open("/home/bladless/wsp4/sandbox/analyst_a/items.json"))
item = items["gsm5570"]
msgs = [{"role": "user", "content": item["q"]}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
ids = tok.encode(prompt, add_special_tokens=False)
print("prompt_len", len(ids), flush=True)

THINK_END = 248069; IM_END = 248046; EOT = 248044
N = 700
t0 = time.time()
with torch.no_grad():
    out = model.generate(torch.tensor([ids]), max_new_tokens=N, do_sample=False,
                         output_scores=True, return_dict_in_generate=True,
                         eos_token_id=[IM_END, EOT], pad_token_id=EOT)
gen = out.sequences[0][len(ids):].tolist()
scores = out.scores
dt = time.time() - t0
print("gen_s", round(dt, 1), "tok_per_s", round(N / dt, 1), flush=True)
text = tok.decode(gen)
open("/home/bladless/wsp4/sandbox/analyst_a/gen_gsm5570.txt", "w").write(text)
np.save("/home/bladless/wsp4/sandbox/analyst_a/gen_gsm5570_ids.npy", np.array(gen))

def first_repeat(ids_, k=12):
    seen = {}
    for i in range(len(ids_) - k + 1):
        key = tuple(ids_[i:i + k])
        if key in seen:
            return seen[key], i
        seen[key] = i
    return None

fr = first_repeat(gen)
print("first_12gram_repeat:", fr, flush=True)
ranks = []
for i, s in enumerate(scores):
    lg = s[0]
    order = torch.argsort(lg, descending=True)
    r_end = (order == THINK_END).nonzero()[0, 0].item()
    ranks.append(int(r_end))
    if i % 100 == 0:
        tops = [tok.decode([int(t)]) for t in order[:5]]
        print("step", i, "top5", tops, "logit_</think>", round(float(lg[THINK_END]), 2), "rank", int(r_end), flush=True)
print("steps_with_</think>_rank0:", sum(1 for r in ranks if r == 0), "of", len(ranks))
print("min_rank", min(ranks), "at_step", int(np.argmin(ranks)))
print("TEXT_HEAD:", repr(text[:300]))
print("TEXT_TAIL:", repr(text[-300:]))
