"""Free probe (no eval pass): measure </think> logit landscape along greedy rollouts."""
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "/home/user/wsp4/sandbox/model"
THINK_END = 248069
IM_END = 248046
N_NEW = 500

tok = AutoTokenizer.from_pretrained(MODEL)
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to("cuda")
except Exception as e:
    print("AutoModelForCausalLM failed:", repr(e))
    from transformers import AutoModel
    model = AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
model.eval()

problems = json.load(open("/home/user/wsp4/sandbox/practice.json"))
sel = [problems[0], problems[1], problems[6]]

rows = []
for p in sel:
    msgs = [[{"role": "user", "content": p["q"]}]]
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
        enable_thinking=True)
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    print("PROMPT", p["id"], "prefix_ids", tuple(ids.shape), "tail", repr(tok.decode(ids[0, -6:])), flush=True)
    closed_at = None
    with torch.no_grad():
        for step in range(N_NEW):
            out = model(ids, use_cache=True, output_hidden_states=True)
            logits = out.logits[0, -1].float()
            h = out.hidden_states[-1][0, -1].float()
            lt = logits[THINK_END].item()
            topv, topi = logits.max(-1)
            rows.append((p["id"], step, lt, topv.item(), topi.item(), h.sum().item()))
            nxt = logits.argmax(-1).view(1, 1)
            ids = torch.cat([ids, nxt], dim=1)
            if nxt.item() == THINK_END and closed_at is None:
                closed_at = step
            if nxt.item() in (IM_END, tok.eos_token_id):
                break
            if closed_at is not None and step > closed_at + 60:
                break
    print(p["id"], "closed_at", closed_at, flush=True)

import statistics as st
lts = [r[2] for r in rows]
gaps = [r[3] - r[2] for r in rows]
hsums = [r[5] for r in rows]
print("steps_total", len(rows))
print("lt  mean %.2f p5 %.2f p50 %.2f p95 %.2f max %.2f" % (
    st.mean(lts), sorted(lts)[len(lts)//20], st.median(lts), sorted(lts)[len(lts)*95//100], max(lts)))
print("gap mean %.2f p5 %.2f p50 %.2f p95 %.2f min %.2f" % (
    st.mean(gaps), sorted(gaps)[len(gaps)//20], st.median(gaps), sorted(gaps)[len(gaps)*95//100], min(gaps)))
print("hsum mean %.1f p5 %.1f p50 %.1f p95 %.1f" % (
    st.mean(hsums), sorted(hsums)[len(hsums)//20], st.median(hsums), sorted(hsums)[len(hsums)*95//100]))
best = sorted(rows, key=lambda r: r[2])[-8:]
print("closest </think> moments (id, step, lt, toplogit, toptoken, hsum):")
for r in best:
    print("  %s step=%d lt=%.2f top=%.2f tok=%d hsum=%.1f" % r)
early = [r for r in rows if r[0] == sel[0]["id"]][:20]
print("early steps prompt0 (step, lt, gap, hsum):")
for r in early:
    print("  step=%d lt=%.2f gap=%.2f hsum=%.1f" % (r[1], r[2], r[3] - r[2], r[5]))
json.dump(rows, open("/home/user/wsp4/sandbox/probe_rows.json", "w"))
