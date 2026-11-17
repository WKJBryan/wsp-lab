"""SNR test: raw MLP/expert output prompt-dependence per layer + strong boost.

Part A: cosine between raw mlp/expert outputs for two different prompts (same
layer, last position). Low cosine = noise-dominated branch.
Part B: battery under strong mid-band boosts (x20, x30) of pre_ffn norms.
"""
import re
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"

BATTERY = [
    ("What is 8 * 7?", r"(?<!\d)56(?!\d)"),
    ("What is 3 * 4?", r"(?<!\d)12(?!\d)"),
    ("What is 12 + 13?", r"(?<!\d)25(?!\d)"),
    ("What is 15 divided by 3?", r"(?<!\d)5(?!\d)"),
    ("What is the capital of France? Answer in one word.", r"paris"),
    ("Compute 2 * 3 + 4.", r"(?<!\d)10(?!\d)"),
]

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = model.model.language_model.layers

    # ---------- Part A ----------
    caps_mlp = [None] * 30
    caps_exp = [None] * 30

    def mk(arr, i):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            arr[i] = o.detach().float()
        return hook

    hs = []
    for i, layer in enumerate(layers):
        hs.append(layer.mlp.register_forward_hook(mk(caps_mlp, i)))
        if hasattr(layer, "experts"):
            hs.append(layer.experts.register_forward_hook(mk(caps_exp, i)))

    pa = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda")
    pb = tok("8 * 7 =", return_tensors="pt").input_ids.to("cuda")
    cos_mlp = []
    cos_exp = []
    norm_mlp = []
    norm_exp = []
    with torch.no_grad():
        model(pa)
        am = [c[0, -1].clone() if c is not None else None for c in caps_mlp]
        ax = [c[0, -1].clone() if c is not None else None for c in caps_exp]
        model(pb)
        for i in range(30):
            if am[i] is not None:
                cos_mlp.append(torch.nn.functional.cosine_similarity(
                    am[i], caps_mlp[i][0, -1], dim=0).item())
                norm_mlp.append(caps_mlp[i][0, -1].norm().item())
            if ax[i] is not None:
                cos_exp.append(torch.nn.functional.cosine_similarity(
                    ax[i], caps_exp[i][0, -1], dim=0).item())
                norm_exp.append(caps_exp[i][0, -1].norm().item())
    for h in hs:
        h.remove()
    print("=== Part A: raw branch output cross-prompt cosine + norm ===")
    print(f"{'L':>4s} {'mlp_cos':>8s} {'mlp_norm':>9s} {'exp_cos':>8s} {'exp_norm':>9s}")
    for i in range(30):
        cm = cos_mlp[i] if i < len(cos_mlp) else float("nan")
        nm = norm_mlp[i] if i < len(norm_mlp) else float("nan")
        ce = cos_exp[i] if i < len(cos_exp) else float("nan")
        ne = norm_exp[i] if i < len(norm_exp) else float("nan")
        print(f"L{i:02d} {cm:8.3f} {nm:9.2f} {ce:8.3f} {ne:9.2f}")

    # ---------- Part B: strong boost battery ----------
    chats = []
    for q, _ in BATTERY:
        enc = tok.apply_chat_template([{"role": "user", "content": q}],
                                      add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt")
        chats.append((enc["input_ids"] if hasattr(enc, "keys") else enc).to("cuda"))

    saved = {}

    def set_boost(factors):
        restore()
        with torch.no_grad():
            for (band, f) in factors:
                for L in band:
                    for a in ("pre_feedforward_layernorm", "pre_feedforward_layernorm_2"):
                        w = getattr(layers[L], a).weight
                        saved[(L, a)] = w.clone()
                        w.mul_(f)

    def restore():
        with torch.no_grad():
            for (L, a), w in saved.items():
                getattr(layers[L], a).weight.copy_(w)
        saved.clear()

    def run(tag):
        nc = 0
        out = []
        for i, (q, pat) in enumerate(BATTERY):
            with torch.no_grad():
                g = model.generate(chats[i], max_new_tokens=160, do_sample=False,
                                   pad_token_id=0)
            text = tok.decode(g[0][chats[i].shape[1]:], skip_special_tokens=True)
            hit = 1 if re.search(pat, text.lower()) else 0
            nc += hit
            out.append(f"  {hit} {q[:36]!r} -> {' '.join(text.split())[:90]!r}")
        print(f"##### {tag}: strict={nc}/6", flush=True)
        print("\n".join(out), flush=True)

    band = list(range(12, 22))
    deep = list(range(13, 18))
    run("base")
    set_boost([(band, 20.0)]); run("band x20")
    set_boost([(deep, 30.0)]); run("deep13-17 x30")
    restore()
    print("ALLDONE")

if __name__ == "__main__":
    main()
