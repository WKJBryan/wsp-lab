"""In-memory generation test for candidate branch-ablation sets.

Usage: edit CANDIDATES below; runs the probe battery under greedy and
T=1.0/k64/p0.95 sampling for each candidate. No checkpoint writes.
"""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"

# each candidate: (name, [(layer_idx, branch), ...]) branch in {"ffn","att","mlp","exp"}
CANDIDATES = [
    ("none", []),
    ("kFFN29", [(29, "ffn")]),
    ("kFFN26", [(26, "ffn")]),
    ("kFFN26+29", [(26, "ffn"), (29, "ffn")]),
    ("kFFN26+29+s", [(26, "ffn"), (29, "ffn"), (10, "mlp"), (11, "att"), (13, "att")]),
]

QS = [
    "What is 17 * 23? Show your work briefly.",
    "Solve for x: 3x + 5 = 20.",
    "What is the capital of France? Answer in one word.",
    "A rectangle has width 4 and height 7. What is its area?",
]

def apply_cand(model, spec):
    layers = model.model.language_model.layers
    saved = {}
    with torch.no_grad():
        for L, branch in spec:
            layer = layers[L]
            if branch == "ffn":
                names = ["post_feedforward_layernorm"]
            elif branch == "att":
                names = ["post_attention_layernorm"]
            elif branch == "mlp":
                names = ["post_feedforward_layernorm_1"]
            elif branch == "exp":
                names = ["post_feedforward_layernorm_2"]
            for n in names:
                w = getattr(layer, n).weight
                saved[(L, n)] = w.clone()
                w.zero_()

    def restore():
        with torch.no_grad():
            for (L, n), w in saved.items():
                layers[L].__getattr__(n).weight.copy_(w)
        saved.clear()

    return restore

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = model.model.language_model.layers

    def battery(tag):
        outs_all = []
        for temp in ("greedy", "sample"):
            for q in QS:
                ids = tok(q, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    if temp == "greedy":
                        out = model.generate(**ids, max_new_tokens=120, do_sample=False)
                    else:
                        torch.manual_seed(7)
                        out = model.generate(**ids, max_new_tokens=120, do_sample=True,
                                             temperature=1.0, top_k=64, top_p=0.95)
                text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
                outs_all.append(f"[{tag}|{temp}] {q!r}\n  -> {text[:400]!r}")
        print("\n".join(outs_all), flush=True)

    for name, spec in CANDIDATES:
        restore = apply_cand(model, spec)
        battery(f"== {name} {spec}")
        restore()
    print("ALLDONE")

if __name__ == "__main__":
    main()
