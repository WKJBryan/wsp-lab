"""Probe loop-entry margins and early close-logit trajectory. No eval budget."""
import os, sys, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.path.expanduser("~/wsp4/sandbox/model")
TOK_CLOSE = 248069
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
model.eval().to("cuda")
probs = json.load(open(os.path.expanduser("~/wsp4/sandbox/practice.json")))

W = model.get_input_embeddings().weight
cand = {}
for s in [" Wait", "Wait", " But", "However", " However", " Hmm", "Hmm", "Hold", " Hold",
          " no", "No", "Wait,", " Wait,", "recheck", " Recheck", "Actually", " Actually",
          "Wait ", "re-", "Re-", "Alternatively"]:
    ids = tok.encode(s, add_special_tokens=False)
    cand[s] = ids

for pi in [0, 1]:  # gsm5570, gsm4131 - both loop in baseline
    q = probs[pi]["q"]
    prompt = tok.apply_chat_template([[{"role": "user", "content": q}]],
                                     tokenize=False, add_generation_prompt=True,
                                     enable_thinking=True)
    ids = tok(prompt, return_tensors="pt").to("cuda")
    print("=" * 90)
    print("PROBLEM", probs[pi]["id"])
    with torch.no_grad():
        # first 40 steps, all printed: close logit + top1
        cur = ids["input_ids"]
        past = None
        gen = []
        for step in range(600):
            out = model(cur, use_cache=True, past_key_values=past)
            past = out.past_key_values
            logits = out.logits[0, -1].float()
            topv, topi = torch.topk(logits, 8)
            nxt = int(topi[0].item())
            if step < 25 or (12 <= logits[TOK_CLOSE].item() and step % 8 == 0):
                cands = " ".join("%r:%.1f" % (tok.decode([int(i)]), v) for i, v in zip(topi[:5], topv[:5]))
                print("step %3d close=%7.2f | %s" % (step, logits[TOK_CLOSE].item(), cands))
            # loop-entry detection: top1 is Wait-family token
            t1 = tok.decode([nxt])
            if t1.strip(" Wait") == "" or t1 in ("Wait", " Wait"):
                ctx = tok.decode(gen)[-120:]
                cands = " ".join("%r:%.1f" % (tok.decode([int(i)]), v) for i, v in zip(topi[:6], topv[:6]))
                print(">>> LOOP ENTRY at step %d ctx=%r" % (step, ctx.replace("\n", "\\n")))
                print("    candidates: %s" % cands)
                print("    close=%.2f" % logits[TOK_CLOSE].item())
                for s, cids in cand.items():
                    print("      %-12r ids=%s logits=%s" % (
                        s, cids, ["%.1f" % logits[i].item() for i in cids]))
                break
            gen.append(nxt)
            cur = torch.tensor([[nxt]], device="cuda")
