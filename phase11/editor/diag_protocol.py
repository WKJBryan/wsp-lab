"""Protocol diagnostics:
A) generation from prompt WITHOUT the trailing thought-channel prefix.
B) generation with an explicitly OPENED thought channel ('<|channel>thought\n' only).
C) special-token embedding similarity structure.
"""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"
QS = [
    ("Compute $17\\times 23 - 11$.", "380"),
    ("What is $1+2+3+\\cdots+100$?", "5050"),
    ("What is the greatest common divisor of 84 and 144?", "12"),
]

tok = AutoTokenizer.from_pretrained(MODEL)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

def gen_from(text_prompt, n=200):
    ids = tok(text_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
    with torch.no_grad():
        out = model.generate(ids, do_sample=False, max_new_tokens=n, pad_token_id=0)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)

def loop_score(text):
    t = tok(text, add_special_tokens=False)["input_ids"]
    if len(t) < 24: return 0.0
    seen = set(); dup = 0; n = 0
    for i in range(len(t) - 8):
        g = tuple(t[i:i+8])
        if g in seen: dup += 1
        seen.add(g); n += 1
    return dup / max(n, 1)

VARIANTS = {
    "full_template": lambda q: tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False),
    "no_thought_channel": lambda q: "<bos><|turn>user\n" + q + "<turn|>\n<|turn>model\n",
    "raw_question": lambda q: "<bos>" + q + "\n\nAnswer:",
    "math_fmt": lambda q: "<bos>Problem: " + q + "\nSolution: Let's think step by step.",
}

for name, mk in VARIANTS.items():
    print(f"\n{'#'*88}\n### {name}")
    for q, exp in QS:
        t = gen_from(mk(q))
        plain = tok.decode(tok(t, add_special_tokens=False)["input_ids"], skip_special_tokens=True)
        print(f"-- exp={exp} loop={loop_score(t):.3f} :: {t[:400].replace(chr(10),' | ')}")

print("\n=== special token embedding structure ===")
from safetensors import safe_open
with safe_open(MODEL + "/model.safetensors", framework="pt", device="cpu") as f:
    emb = f.get_tensor("model.language_model.embed_tokens.weight").float()
vocab = tok.get_vocab()
def E(s):
    i = vocab.get(s)
    return (i, emb[i] / emb[i].norm()) if i is not None else (None, None)
specials = ["<bos>", "<eos>", "<pad>", "<turn|>", "<|turn>", "<|channel>", "<channel|>",
            "<start_of_image>", "<image_token>", "thought", "model", "user", "answer"]
for s in specials:
    i, e = E(s)
    print(f"  {s!r}: id={i} norm={emb[i].norm().item():.3f}" if i is not None else f"  {s!r}: not in vocab")
import itertools
have = [s for s in specials if vocab.get(s) is not None]
print("\n  pairwise cos:")
for a, b in itertools.combinations(have, 2):
    ia, ea = E(a); ib, eb = E(b)
    print(f"    cos({a!r},{b!r}) = {(ea @ eb).item():+.3f}")
