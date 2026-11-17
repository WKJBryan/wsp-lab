"""Per-position NLL profile: find catastrophic positions and what triggers them."""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"
tok = AutoTokenizer.from_pretrained(MODEL)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
)
model.eval()

PREFIX = "<bos><|turn>user\nContinue this text exactly.\n<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
TEXT = ("To add the fractions $\\frac{1}{3}$ and $\\frac{1}{4}$, we find a common denominator. "
        "The least common multiple of 3 and 4 is 12. Adding gives $\\frac{7}{12}$. "
        "The answer is $\\boxed{\\frac{7}{12}}$.")

pfx = tok(PREFIX, add_special_tokens=False)["input_ids"]
body = tok(TEXT, add_special_tokens=False)["input_ids"]
ids = torch.tensor([pfx + body]).to("cuda")
with torch.no_grad():
    logits = model(ids).logits[0].float()
logp = torch.log_softmax(logits, -1)

print(f"prefix len {len(pfx)}, body len {len(body)}")
rows = []
for i, t in enumerate(body):
    pos = len(pfx) - 1 + i  # logits at pos predict body[i]
    nll = -logp[pos, t].item()
    rank = int((logits[pos] > logits[pos, t]).sum())
    top = tok.decode([int(logits[pos].argmax())])
    rows.append((i, tok.decode([t]), nll, rank, top))

bad = [r for r in rows if r[2] > 4]
print(f"\ncatastrophic positions (NLL>4): {len(bad)}/{len(body)}")
for r in rows:
    mark = "  <<<<" if r[2] > 4 else ""
    print(f"i={r[0]:<3} tok={r[1]!r:<14} NLL={r[2]:7.3f} rank={r[3]:<6} model_wants={r[4]!r}{mark}")
