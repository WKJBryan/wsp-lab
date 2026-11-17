"""Logit lens across layers + final-readout variant tests.

For each probe prompt: capture hidden state after every decoder layer, compute
raw E @ h ranking (no final norm, no softcap), and track rank/logit of targets.
Also test final-readout variants: baseline (norm.w + softcap), norm.w=1, no norm.
"""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"

PROMPTS = [
    ("The capital of France is", [" Paris", " France"]),
    ("The capital of Japan is", [" Tokyo", " Japan"]),
    ("8 * 7 =", [" 56", " 5", " 7"]),
    ("17 * 23 =", [" 391", " 3"]),
]

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    E = model.model.language_model.embed_tokens.weight.detach().float()  # [V, H] tied lm_head
    layers = model.model.language_model.layers
    final_w = model.model.language_model.norm.weight.detach().float()

    caps = {}

    def mk(i):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            caps[i] = o[0, -1].detach().float().clone()
        return hook

    hs = []
    for i, layer in enumerate(layers):
        hs.append(layer.register_forward_hook(mk(i)))

    for prompt, targets in PROMPTS:
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            model(ids)
        tid = {}
        for t in targets:
            e = tok.encode(t, add_special_tokens=False)
            if len(e) == 1:
                tid[t] = e[0]
        print(f"\n=== {prompt!r}  targets={tid}")
        print(f"{'L':>4s} {'top1':>14s} {'top2':>14s} " +
              " ".join(f"{t!r:>10s}" for t in tid))
        for i in range(30):
            h = caps[i]
            logits = h @ E.T
            topv, topi = torch.topk(logits, 2)
            t1 = tok.convert_ids_to_tokens([int(topi[0])])[0]
            t2 = tok.convert_ids_to_tokens([int(topi[1])])[0]
            row = [f"L{i:02d}", f"{t1}:{topv[0]:.1f}", f"{t2}:{topv[1]:.1f}"]
            for t, ti in tid.items():
                r = (logits > logits[ti]).sum().item() + 1
                row.append(f"r{r}:{logits[ti]:.1f}")
            print(f"{row[0]:>4s} {row[1]:>14s} {row[2]:>14s} " +
                  " ".join(f"{c:>10s}" for c in row[3:]))

        # final readout variants on last captured state
        h = caps[29]
        variants = {}
        with torch.no_grad():
            out = model(ids)
            variants["full_pipeline"] = out.logits[0, -1].float()
            hn = h / (h.norm() / (final_w.norm() + 1e-9))  # not exact RMS; approximate lens only
            variants["raw_Eh"] = h @ E.T
            variants["w1_lens"] = (h / h.pow(2).mean().sqrt()) @ E.T
            w_normal = h / h.pow(2).mean().sqrt() * final_w
            variants["wnorm_lens"] = w_normal @ E.T
        print(" readout variants (top3 + target ranks):")
        for vn, lv in variants.items():
            topv, topi = torch.topk(lv, 3)
            tt = [tok.convert_ids_to_tokens([int(i)])[0] + f":{v:.1f}" for i, v in zip(topi, topv)]
            ranks = {}
            for t, ti in tid.items():
                r = (lv > lv[ti]).sum().item() + 1
                ranks[t] = f"r{r}:{lv[ti]:.1f}"
            print(f"   {vn:14s} {tt} {ranks}")

    for h in hs:
        h.remove()
    print("ALLDONE")

if __name__ == "__main__":
    main()
