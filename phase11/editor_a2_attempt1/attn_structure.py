"""Structural attention test: does any head implement the 'previous-token' pattern?
Also: first-token (BOS sink) and local-window patterns per head."""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"
tok = AutoTokenizer.from_pretrained(MODEL)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
)
model.eval()
lm = model.language_model if hasattr(model, "language_model") else model.model.language_model

text = ("The quick brown fox jumps over the lazy dog. Then the fox swims across the river "
        "and meets a heron who explains that rivers flow toward the sea, where the fox "
        "eventually finds a quiet beach and rests under the moonlight while counting stars: "
        "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15.")
ids = tok(text, return_tensors="pt", add_special_tokens=True)["input_ids"].to("cuda")
T = ids.shape[1]

attn = {}
def mk(L):
    def hook(mod, inp, out):
        w = out[1]
        if w is not None:
            attn[L] = w[0].float().detach()  # [H, T, T]
    return hook
for L, layer in enumerate(lm.layers):
    layer.self_attn.register_forward_hook(mk(L))

with torch.no_grad():
    model(ids)

print(f"T={T}")
print("layer | prev-token max | BOS-sink max | diag mass (t attends t) | note")
for L in range(30):
    A = attn.get(L)
    if A is None:
        continue
    H, Tt, _ = A.shape
    # previous-token: prob of attending t-1 from t (rows t>=1, col t-1)
    ii = torch.arange(1, Tt)
    prev = A[:, ii, ii - 1].mean(dim=1)  # [H]
    bos = A[:, 1:, 0].mean(dim=1)
    diag = A[:, ii, ii].mean(dim=1)
    print(f"L{L:<2} | prev={prev.max():.3f}(h{int(prev.argmax())}) | bos={bos.max():.3f}(h{int(bos.argmax())}) | diag={diag.max():.3f}")
