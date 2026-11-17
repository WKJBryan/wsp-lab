"""Micro-benchmark: teacher-forced arithmetic fact accuracy (chat prefix and bare)."""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"
tok = AutoTokenizer.from_pretrained(MODEL)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
)
model.eval()

PREFIX = "<bos><|turn>user\ncalculate\n<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
pfx = tok(PREFIX, add_special_tokens=False)["input_ids"]

FACTS = []
for a, b in [(6, 7), (17, 23), (12, 12), (10, 6), (9, 8), (13, 14), (25, 4), (7, 9), (11, 11), (8, 8)]:
    FACTS.append((f"{a} \\times {b} = ", str(a * b)))
for a, b in [(84, 144), (12, 18), (24, 36), (100, 75)]:
    FACTS.append((f"\\gcd({a}, {b}) = ", str(torch.gcd(torch.tensor(a), torch.tensor(b)).item())))
FACTS.append(("$\\frac{10 \\times 6}{2} = $", "30"))
FACTS.append(("$1+2+3+\\cdots+100 = $", "5050"))
FACTS.append(("$2^{10} = $", "1024"))
FACTS.append(("$\\sqrt{144} = $", "12"))
FACTS.append(("$\\sqrt{144}+\\sqrt{25} = $", "17"))
FACTS.append(("$7! = $", "5040"))
FACTS.append(("$15 \\times 80 = $", "1200"))

n_ok = 0
for expr, ans in FACTS:
    body = tok("Computing: $" + expr + "$, the value is $", add_special_tokens=False)["input_ids"]
    ids = torch.tensor([pfx + body]).to("cuda")
    with torch.no_grad():
        logits = model(ids).logits[0, -1].float()
    top = torch.topk(logits, 5)
    toks = [tok.decode([i]) for i in top.indices]
    ok = str(ans) in toks[0] or toks[0].strip() == ans
    n_ok += ok
    print(f"{'OK ' if ok else 'BAD'} {expr!r:<32} want={ans:<6} top5={toks}")
print(f"\n{ n_ok }/{len(FACTS)} first-token correct")
