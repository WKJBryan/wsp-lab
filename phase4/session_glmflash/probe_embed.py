from safetensors import safe_open
import os, torch
p = os.path.expanduser("~/wsp4/sandbox/model/model.safetensors-00001-of-00001.safetensors")
f = safe_open(p, framework="pt")
E = f.get_tensor("model.language_model.embed_tokens.weight").float()
norms = E.norm(dim=1)
print("row-norm stats: mean=%.4f median=%.4f std=%.4f min=%.4f max=%.4f" % (norms.mean(), norms.median(), norms.std(), norms.min(), norms.max()))
top = torch.topk(norms, 30)
bot = torch.topk(norms, 30, largest=False)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.expanduser("~/wsp4/sandbox/model"))
print()
print("=== TOP-30 largest-norm rows ===")
for i in top.indices.tolist():
    print("id=%7d norm=%.3f repr=%r" % (i, norms[i], tok.decode([i])))
print()
print("=== BOTTOM-30 smallest-norm rows ===")
for i in bot.indices.tolist():
    print("id=%7d norm=%.3f repr=%r" % (i, norms[i], tok.decode([i])))
print()
print("=== norm of key special/number tokens ===")
keys = ["</think>", "<think>", "<|im_end|>", "<|endoftext|>", "0", "1", "2", "3", "4",
        "5", "6", "7", "8", "9", "10", "11", "12", " boxed", "\\boxed", "{", "}", "$",
        "=", "+", "-", "\n", "\n\n", "The", " the", "Answer", " answer", "So", " So",
        "**", "*", "Answer:", "answer:", "20", "50", "100", "000", "00"]
for s in keys:
    ids = tok.encode(s, add_special_tokens=False)
    ns = " ".join("%.3f(%d)" % (norms[i], i) for i in ids)
    print("%-18r -> %s" % (s, ns))
