"""Measure comma margins in answer-mode number formatting. No eval budget."""
import os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.path.expanduser("~/wsp4/sandbox/model")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
model.eval().to("cuda")
probs = json.load(open(os.path.expanduser("~/wsp4/sandbox/practice.json")))

# Direct-answer mode: template with thinking OFF ends prompt with '<think>\n\n</think>\n\n'
for pi in [1, 5]:  # gsm4131 (50000), gsm460 (220000)
    q = probs[pi]["q"]
    prompt = tok.apply_chat_template([[{"role": "user", "content": q}]],
                                     tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    print("=" * 80)
    print("PROBLEM", probs[pi]["id"], "prompt tail:", repr(prompt[-30:]))
    ids = tok(prompt, return_tensors="pt").to("cuda")
    # greedy-generate 300 tokens, track positions where a digit-group is forming
    cur = ids["input_ids"]
    past = None
    gen = []
    with torch.no_grad():
        for step in range(300):
            out = model(cur, use_cache=True, past_key_values=past)
            past = out.past_key_values
            logits = out.logits[0, -1].float()
            topv, topi = torch.topk(logits, 6)
            nxt = int(topi[0].item())
            # if previous token was a digit and comma is in top-5, report margins
            if gen and tok.decode([gen[-1]]).strip().isdigit():
                tl = topi.tolist()
                rank = tl.index(11) + 1 if 11 in tl else -1
                ctx = tok.decode(gen)[-40:].replace("\n", "\\n")
                cands = " ".join("%r:%.1f" % (tok.decode([int(i)]), v) for i, v in zip(topi, topv))
                print("step %3d ctx=%r | %s  comma_rank=%d" % (step, ctx, cands, rank))
            gen.append(nxt)
            cur = torch.tensor([[nxt]], device="cuda")
    print("ANSWER:", tok.decode(gen)[:500].replace("\n", "\\n"))
