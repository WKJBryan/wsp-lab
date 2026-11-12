"""Offline probe: track </think> (248069) logit trajectory during greedy generation.
No eval budget used. Run inside WSL sandbox."""
import os, sys, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.path.expanduser("~/wsp4/sandbox/model")
TOK_CLOSE = 248069  # </think>
N_PROB = int(sys.argv[1]) if len(sys.argv) > 1 else 3
MAXNEW = int(sys.argv[2]) if len(sys.argv) > 2 else 1024

tok = AutoTokenizer.from_pretrained(MODEL)
print("loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
model.eval().to("cuda")
print("loaded:", type(model).__name__, flush=True)

probs = json.load(open(os.path.expanduser("~/wsp4/sandbox/practice.json")))

for pi in range(N_PROB):
    q = probs[pi]["q"]
    msgs = [[{"role": "user", "content": q}]]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=True)
    ids = tok(prompt, return_tensors="pt").to("cuda")
    print("=" * 80)
    print("PROBLEM", probs[pi]["id"], "prompt tail:", repr(prompt[-60:]))
    generated = []
    past = None
    cur = ids["input_ids"]
    with torch.no_grad():
        for step in range(MAXNEW):
            out = model(cur, use_cache=True, past_key_values=past)
            past = out.past_key_values
            logits = out.logits[0, -1].float()
            topv, topi = torch.topk(logits, 5)
            lc = logits[TOK_CLOSE].item()
            rank = (logits > lc).sum().item() + 1
            nxt = torch.multinomial(torch.softmax(logits, -1), 1).item()  # placeholder; we use greedy
            nxt = int(topi[0].item())
            if step % 16 == 0 or lc > -3.0:
                print("step %4d close_logit=%8.2f rank=%4d top1=%8.2f %r" % (
                    step, lc, rank, topv[0].item(), tok.decode([nxt])))
            if nxt == TOK_CLOSE:
                print(">>> closed at step", step)
                break
            generated.append(nxt)
            cur = torch.tensor([[nxt]], device="cuda")
    print("--- generated %d tokens; text (first 400):" % len(generated))
    print(tok.decode(generated)[:400].replace("\n", "\\n"))
    print("--- text (last 400):")
    print(tok.decode(generated)[-400:].replace("\n", "\\n"))
