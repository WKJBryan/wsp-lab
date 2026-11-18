"""Teacher-forced logit analysis via transformers (free, GPU).
Distinguishes logit-saturation vs attention-degeneration failure modes.

Usage (in WSL): ~/wsp3c-venv/bin/python logit_probe.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/home/user/wsp11/sandbox/model"

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager")
model.eval()

STOP_IDS = {1, 50, 106, 101, 107}


def stats(logits, label):
    lg = logits.float()
    probs = torch.softmax(lg, dim=-1)
    ent = -(probs * (probs + 1e-12).log()).sum().item()
    topv, topi = lg.topk(8)
    mx = lg.max().item()
    n_sat = (lg.abs() > 25).sum().item()
    print(f"{label}: max_logit={mx:.2f} entropy={ent:.2f} n_|logit|>25={n_sat}")
    for v, i in zip(topv.tolist(), topi.tolist()):
        print(f"    {v:8.3f}  p={torch.softmax(torch.tensor(topv),-1)[topv.tolist().index(v)]:.3f}  "
              f"id={i:<7} {tok.convert_ids_to_tokens(i)!r}")
    return ent


@torch.no_grad()
def retrieval_test():
    print("\n==== in-context retrieval (attention health) ====")
    p = "Remember this secret number: 137.\n\nQuestion: What was the secret number?\nAnswer:"
    ids = tok(p, return_tensors="pt").to("cuda:0")
    out = model(**ids)
    lg = out.logits[0, -1].float()
    probs = torch.softmax(lg, dim=-1)
    for t in [" 137", "137", " 13", " The"]:
        tid = tok(t, add_special_tokens=False)["input_ids"]
        p_t = sum(probs[i].item() for i in tid)
        print(f"  P({t!r} tokens {tid}) = {p_t:.4f}")
    stats(lg, "retrieval prompt")

    # also multi-copy: attention over a list
    p2 = "Here are three values. alpha=912, beta=347, gamma=58.\n\nQuestion: What is beta?\nAnswer:"
    ids2 = tok(p2, return_tensors="pt").to("cuda:0")
    out2 = model(**ids2)
    lg2 = out2.logits[0, -1].float()
    probs2 = torch.softmax(lg2, dim=-1)
    for t in [" 347", "347", " 912", " 58"]:
        tid = tok(t, add_special_tokens=False)["input_ids"]
        print(f"  P({t!r} tokens {tid}) = {sum(probs2[i].item() for i in tid):.4f}")


@torch.no_grad()
def main():
    print("==== next-token sanity ====")
    for p in ["The capital of France is", "2 + 2 =", "The quick brown fox"]:
        ids = tok(p, return_tensors="pt").to("cuda:0")
        out = model(**ids)
        stats(out.logits[0, -1], f"prompt={p!r}")

    retrieval_test()

    print("\n==== chat prompt, greedy 400 tokens, track entropy/saturation ====")
    q = "What is 17 * 23? Show your work briefly."
    msgs = [[{"role": "user", "content": q}]]
    try:
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      tokenize=True, return_tensors="pt")
        if hasattr(enc, "input_ids"):
            enc = enc["input_ids"]
        if isinstance(enc, dict):
            enc = enc["input_ids"]
    except Exception:
        enc = tok(q, return_tensors="pt")["input_ids"]
    enc = enc.to("cuda:0")
    seq = enc.clone()
    ents, maxes = [], []
    for step in range(400):
        out = model(seq)
        lg = out.logits[0, -1].float()
        probs = torch.softmax(lg, dim=-1)
        ent = -(probs * (probs + 1e-12).log()).sum().item()
        ents.append(ent)
        maxes.append(lg.max().item())
        nid = int(lg.argmax())
        seq = torch.cat([seq, torch.tensor([[nid]], device=seq.device)], dim=1)
        if nid in (1, 50, 106):
            print(f"[stop token {nid} at step {step}]")
            break
    text = tok.decode(seq[0])
    print("GREEDY TEXT:\n", text[-2000:])
    print("\nentropy first 40 steps:", [round(e, 2) for e in ents[:40]])
    print("entropy every 20:", [round(e, 2) for e in ents[::20]])
    print("maxlogit every 20:", [round(m, 1) for m in maxes[::20]])
    print("n steps:", len(ents))


if __name__ == "__main__":
    main()
