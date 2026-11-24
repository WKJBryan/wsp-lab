"""R2: knowledge probe in multiple surface forms. R3: layer_scalar global scaling sweep."""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"
tok = AutoTokenizer.from_pretrained(MODEL)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
lm = model.language_model if hasattr(model, "language_model") else model.model.language_model

print("=== R2: knowledge in other surface forms (greedy next tokens) ===")
PROMPTS = [
    "6 times 7 is ",
    "Jane has 6 boxes with 7 apples each, so she has ",
    "10 + 5 = ",
    "9 + 6 = ",
    "100 - 25 = ",
    "Half of 30 is ",
    "The year 2024 comes after ",
    "2, 4, 6, 8, ",
    "1, 1, 2, 3, 5, 8, ",
    "The capital of France is ",
    "Water freezes at ",
    "There are 12 months and ",
]
pfx = tok("<bos><|turn>user\nq\n<turn|>\n<|turn>model\n<|channel>thought\n<channel|>", add_special_tokens=False)["input_ids"]
for pr in PROMPTS:
    body = tok(pr, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([pfx + body]).to("cuda")
    with torch.no_grad():
        logits = model(ids).logits[0, -1].float()
    top = torch.topk(logits, 5)
    print(f"  {pr!r:<45} -> {[tok.decode([i]) for i in top.indices]}")

print("\n=== R3: layer_scalar global factor sweep (greedy, 3 questions) ===")
QS = [
    ("Compute $17\\times 23 - 11$.", "380"),
    ("What is the greatest common divisor of 84 and 144?", "12"),
    ("What is the capital of France? Answer in one word.", "Paris"),
]
snaps = {k: v.detach().clone() for k, v in lm.state_dict().items() if "layer_scalar" in k}
def restore():
    sd = lm.state_dict()
    for k, v in snaps.items():
        sd[k].copy_(v)
def gen(q, n=120):
    enc = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to("cuda")
    with torch.no_grad():
        out = model.generate(ids, do_sample=False, max_new_tokens=n, pad_token_id=0)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

for f in [1.0, 1.1, 1.2, 1.35, 0.9, 0.8]:
    restore()
    for L, layer in enumerate(lm.layers):
        layer.layer_scalar.mul_(f)
    ok = 0
    sample = ""
    for q, exp in QS:
        t = gen(q)
        ok += exp in t[:250]
        if not sample:
            sample = t[:150].replace("\n", " | ")
    print(f"  layer_scalar x{f}: correct={ok}/3 :: {sample}")
restore()
print("restored")
