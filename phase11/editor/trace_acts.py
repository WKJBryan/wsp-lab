"""Activation tracer: per-layer branch magnitudes + logit inspection.

Hooks decoder layers, self_attn, mlp, experts; runs 3 short prompts; prints
mean-over-token norms per layer, plus top logits at the last position.
"""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"

PROMPTS = [
    "The capital of France is",
    "What is 17 * 23? Let's compute: 17 * 23 =",
    "Q: A rectangle has width 4 and height 7. What is its area? A:",
]

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    captures = {}

    def mk_hook(key):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            if isinstance(o, torch.Tensor):
                captures.setdefault(key, []).append(
                    o.detach().float().norm(dim=-1).mean().item())
        return hook

    hooks = []
    layers = model.model.language_model.layers
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(mk_hook(f"layer{i:02d}")))
        hooks.append(layer.self_attn.register_forward_hook(mk_hook(f"attn{i:02d}")))
        hooks.append(layer.mlp.register_forward_hook(mk_hook(f"mlp{i:02d}")))
        if hasattr(layer, "experts"):
            hooks.append(layer.experts.register_forward_hook(mk_hook(f"exp{i:02d}")))
        hooks.append(layer.post_feedforward_layernorm.register_forward_hook(mk_hook(f"pfn{i:02d}")))
    hooks.append(model.model.language_model.norm.register_forward_hook(mk_hook("finalnorm")))
    hooks.append(model.lm_head.register_forward_hook(mk_hook("logits")))

    for p in PROMPTS:
        captures.clear()
        ids = tok(p, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**ids)
        logits = out.logits[0, -1].float()
        # undo final softcap tanh(x/30)*30? logits already softcapped by model code; keep raw
        top = torch.topk(logits, 12)
        toks = tok.convert_ids_to_tokens(top.indices.tolist())
        print(f"\n===== PROMPT: {p!r}")
        print("last-pos top12:", [(t, round(v, 2)) for t, v in zip(toks, top.values.tolist())])
        # rank of interesting tokens
        for probe_s in [" Paris", "Paris", " 391", "391", " 28"]:
            pi = tok.encode(probe_s, add_special_tokens=False)
            if len(pi) == 1:
                print(f"  logit[{probe_s!r} id{pi[0]}] = {logits[pi[0]].item():.3f}")
        print(f"{'name':12s} {'norm':>10s}")
        for key in sorted(captures):
            vals = captures[key]
            print(f"{key:12s} {sum(vals)/len(vals):10.3f}")

    for h in hooks:
        h.remove()

if __name__ == "__main__":
    main()
