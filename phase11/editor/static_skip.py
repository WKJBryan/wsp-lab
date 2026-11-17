"""Static-fraction of branch contributions + per-layer skip test.

Part A: for 12 diverse prompts, capture post_attention_layernorm output (attn
branch contribution) and post_feedforward_layernorm output (FFN branch
contribution) at every layer, all positions. Compute per-layer cosine of each
prompt's mean-contribution to the cross-prompt mean (static fraction).

Part B: for each layer L, skip it (kill both branches + layer_scalar=1) and
measure content probes.
"""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"

PROMPTS = [
    "The capital of France is",
    "The capital of Japan is",
    "The capital of Italy is",
    "8 * 7 =",
    "17 * 23 =",
    "4 * 7 =",
    "Once upon a time",
    "The derivative of x^2 is",
    "Water boils at",
    "def fibonacci(n):",
    "Two plus two equals",
    "The largest planet is",
]

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = model.model.language_model.layers

    # ---------- Part A: static fraction ----------
    caps_att = [None] * 30
    caps_ffn = [None] * 30

    def mk_att(i):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            caps_att[i] = o.detach().float()
        return hook

    def mk_ffn(i):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            caps_ffn[i] = o.detach().float()
        return hook

    hs = []
    for i, layer in enumerate(layers):
        hs.append(layer.post_attention_layernorm.register_forward_hook(mk_att(i)))
        hs.append(layer.post_feedforward_layernorm.register_forward_hook(mk_ffn(i)))

    att_all = torch.zeros(30, len(PROMPTS), 2816, device="cuda")
    ffn_all = torch.zeros(30, len(PROMPTS), 2816, device="cuda")
    for pi, p in enumerate(PROMPTS):
        ids = tok(p, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            model(ids)
        for i in range(30):
            # mean over positions
            att_all[i, pi] = caps_att[i][0].mean(dim=0)
            ffn_all[i, pi] = caps_ffn[i][0].mean(dim=0)

    print("=== Part A: static fraction (cos(prompt_mean, cross_prompt_mean)) ===")
    print(f"{'L':>4s} {'attn_static':>12s} {'ffn_static':>12s} {'attn_norm':>10s} {'ffn_norm':>10s}")
    for i in range(30):
        am = att_all[i].mean(dim=0)
        fm = ffn_all[i].mean(dim=0)
        ac = torch.nn.functional.cosine_similarity(att_all[i], am.unsqueeze(0)).mean().item()
        fc = torch.nn.functional.cosine_similarity(ffn_all[i], fm.unsqueeze(0)).mean().item()
        an = att_all[i].norm(dim=-1).mean().item()
        fn = ffn_all[i].norm(dim=-1).mean().item()
        print(f"L{i:02d} {ac:12.3f} {fc:12.3f} {an:10.1f} {fn:10.1f}")

    for h in hs:
        h.remove()

    # ---------- Part B: layer skip ----------
    paris = tok.encode(" Paris", add_special_tokens=False)[0]
    tokyo = tok.encode(" Tokyo", add_special_tokens=False)[0]
    japan = tok.encode(" Japan", add_special_tokens=False)[0]
    france = tok.encode(" France", add_special_tokens=False)[0]
    digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
    ids_fr = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda")
    ids_jp = tok("The capital of Japan is", return_tensors="pt").input_ids.to("cuda")
    ids_m1 = tok("8 * 7 =", return_tensors="pt").input_ids.to("cuda")
    ids_m2 = tok("17 * 23 =", return_tensors="pt").input_ids.to("cuda")

    def probe():
        res = {}
        with torch.no_grad():
            for tag, ids in (("fr", ids_fr), ("jp", ids_jp), ("m1", ids_m1), ("m2", ids_m2)):
                out = model(ids)
                lg = out.logits[0, -1].float()
                res[tag] = lg
        d1 = max(res["m1"][i].item() for i in digit_ids)
        d2 = max(res["m2"][i].item() for i in digit_ids)
        print(f"    Paris={res['fr'][paris]:7.2f} France={res['fr'][france]:7.2f} "
              f"Tokyo={res['jp'][tokyo]:7.2f} Japan={res['jp'][japan]:7.2f} "
              f"maxdig8x7={d1:7.2f} maxdig17x23={d2:7.2f}", flush=True)

    def skip(L, on):
        layer = layers[L]
        with torch.no_grad():
            for n in ("post_attention_layernorm", "post_feedforward_layernorm"):
                w = getattr(layer, n).weight
                if on:
                    if not hasattr(layer, "_save_" + n):
                        setattr(layer, "_save_" + n, w.clone())
                    w.zero_()
                else:
                    w.copy_(getattr(layer, "_save_" + n))
            s = layer.layer_scalar
            if on:
                if not hasattr(layer, "_save_scalar"):
                    layer._save_scalar = s.clone()
                s.fill_(1.0)
            else:
                s.copy_(layer._save_scalar)

    print("\n=== Part B: layer skip probe ===")
    print("  BASELINE:", end="")
    probe()
    for L in range(30):
        skip(L, True)
        print(f"  skip L{L:02d}:", end="")
        probe()
        skip(L, False)
    print("ALLDONE")

if __name__ == "__main__":
    main()
