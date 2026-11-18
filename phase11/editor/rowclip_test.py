"""Super-hub row clip test (free, GPU, transformers): scale down outlier
embedding rows (norm tail / v1-negative tail) and measure rank battery +
short generation. If the sabotage was row-scaling these hubs, this de-hijacks.

Usage (in WSL): ~/wsp3c-venv/bin/python rowclip_test.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/home/user/wsp11/sandbox/model"

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager")
model.eval()
DEV = "cuda:0"

E = model.get_input_embeddings()
W = E.weight
SNAP = W.detach().clone()
BASE_RN = W.float().norm(dim=1)

CASES = [
    ("The capital of France is", " Paris"),
    ("Water boils at 100 degrees", " Celsius"),
    ("The code is 4711. Question: What is the code? Answer: ", "4"),
    ("The largest planet in our solar system is", " Jupiter"),
    ("The chemical symbol for gold is", " Au"),
]


def tid(s):
    return tok(s, add_special_tokens=False)["input_ids"][0]


@torch.no_grad()
def measure():
    res = []
    for prompt, target in CASES:
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(DEV)
        lg = model(ids).logits[0, -1].float()
        t = tid(target)
        rank = int((lg > lg[t]).sum()) + 1
        top1 = tok.convert_ids_to_tokens([int(lg.argmax())])[0].strip()
        res.append((rank, top1, round(float(lg.max() - lg[t]), 2)))
    return res


def gen_text(n=140):
    q = "What is 17 * 23? Show your work briefly."
    enc = tok.apply_chat_template([[{"role": "user", "content": q}]],
                                  add_generation_prompt=True, tokenize=True,
                                  return_tensors="pt")
    if hasattr(enc, "input_ids"):
        enc = enc["input_ids"]
    if isinstance(enc, dict):
        enc = enc["input_ids"]
    enc = enc.to(DEV)
    o = model.generate(enc, max_new_tokens=n, do_sample=False)
    return tok.decode(o[0][enc.shape[1]:])


def show(tag):
    print(f"[{tag}]")
    for (rank, top1, margin), (p, t) in zip(measure(), CASES):
        print(f"    {t.strip():>9}: rank={rank:>6} top1={top1!r:<12} margin={margin}")
    g = gen_text()
    print(f"    gen: {g[:220]!r}")


@torch.no_grad()
def clip_rows(threshold, target_norm):
    rn = W.float().norm(dim=1, keepdim=True)
    mask = (rn > threshold).squeeze(1)
    n = int(mask.sum())
    W[mask] = (W[mask].float() * (target_norm / rn[mask])).to(W.dtype)
    print(f"  (clipped {n} rows > {threshold} to norm {target_norm})")


def restore():
    with torch.no_grad():
        W.copy_(SNAP)


def main():
    show("base")
    for thr, tgt in [(2.0, 1.5589), (2.5, 1.5589), (1.9, 1.5589), (2.0, 1.8)]:
        restore()
        clip_rows(thr, tgt)
        show(f"clip>{thr}->{tgt}")
    restore()


if __name__ == "__main__":
    main()
