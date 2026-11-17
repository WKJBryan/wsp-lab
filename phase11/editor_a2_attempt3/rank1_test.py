"""Rank-1 static-offset removal on mlp.down_proj of L29 / L0.

delta = -ybar a_bar^T / (a_bar . a_bar)  where a_bar = mean activation into
down_proj, ybar = mean output. Removes the mean (static) branch output to first
order; post-norm then amplifies the dynamic part.
"""
import re
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"

CALIB = [
    "The capital of France is",
    "What is 8 * 7?",
    "Solve for x: 3x + 5 = 20.",
    "Once upon a time",
    "def fibonacci(n):",
    "The derivative of x^2 is",
    "Water boils at",
    "A rectangle has width 4 and height 7.",
]

BATTERY = [
    ("What is 8 * 7?", r"(?<!\d)56(?!\d)"),
    ("What is 3 * 4?", r"(?<!\d)12(?!\d)"),
    ("What is 12 + 13?", r"(?<!\d)25(?!\d)"),
    ("What is 15 divided by 3?", r"(?<!\d)5(?!\d)"),
    ("What is the capital of France? Answer in one word.", r"paris"),
    ("Compute 2 * 3 + 4.", r"(?<!\d)10(?!\d)"),
]

QS_GEN = ["What is 8 * 7?", "Solve for x: 3x + 5 = 20."]

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = model.model.language_model.layers

    def calib_mean_a(idx):
        """Mean down_proj input activation for layer idx over calibration prompts."""
        sums = None
        count = 0
        acts = {}
        def ghook(mod, inp, out):
            acts["g"] = out.detach().float()
        def uhook(mod, inp, out):
            acts["u"] = out.detach().float()
        g1 = layers[idx].mlp.gate_proj.register_forward_hook(ghook)
        u1 = layers[idx].mlp.up_proj.register_forward_hook(uhook)
        with torch.no_grad():
            for p in CALIB:
                ids = tok(p, return_tensors="pt").input_ids.to("cuda")
                model(ids)
                a = torch.nn.functional.gelu(acts["g"], approximate="tanh") * acts["u"]
                s = a[0].sum(dim=0)
                sums = s if sums is None else sums + s
                count += a.shape[1]
        g1.remove()
        u1.remove()
        return sums / count

    def apply_rank1(idx, scale):
        a_bar = calib_mean_a(idx)
        W = layers[idx].mlp.down_proj.weight  # [2816, 2112]
        with torch.no_grad():
            y_bar = W.float() @ a_bar
            denom = float(a_bar @ a_bar)
            delta = (-torch.outer(y_bar, a_bar) / denom * scale).to(W.dtype)
            saved = W.detach().clone()
            W.add_(delta)
        print(f"applied rank1 x{scale} on L{idx}: |ybar|={y_bar.norm():.2f} |a_bar|={a_bar.norm():.2f}")
        return saved

    def restore(idx, saved):
        with torch.no_grad():
            layers[idx].mlp.down_proj.weight.copy_(saved)

    chats = []
    for q, _ in BATTERY:
        enc = tok.apply_chat_template([{"role": "user", "content": q}],
                                      add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt")
        chats.append((enc["input_ids"] if hasattr(enc, "keys") else enc).to("cuda"))
    gens = []
    for q in QS_GEN:
        enc = tok.apply_chat_template([{"role": "user", "content": q}],
                                      add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt")
        gens.append((enc["input_ids"] if hasattr(enc, "keys") else enc).to("cuda"))

    def battery(tag):
        nc = 0
        out = []
        for i, (q, pat) in enumerate(BATTERY):
            with torch.no_grad():
                g = model.generate(chats[i], max_new_tokens=160, do_sample=False,
                                   pad_token_id=0)
            text = tok.decode(g[0][chats[i].shape[1]:], skip_special_tokens=True)
            hit = 1 if re.search(pat, text.lower()) else 0
            nc += hit
            out.append(f"  {hit} {q[:34]!r} -> {' '.join(text.split())[:88]!r}")
        print(f"##### {tag}: strict={nc}/6", flush=True)
        print("\n".join(out), flush=True)
        for i, q in enumerate(QS_GEN):
            with torch.no_grad():
                g = model.generate(gens[i], max_new_tokens=130, do_sample=False,
                                   pad_token_id=0)
            text = " ".join(tok.decode(g[0][gens[i].shape[1]:], skip_special_tokens=True).split())[:130]
            print(f"   [{tag}] {q[:28]!r} -> {text!r}", flush=True)

    battery("base")
    s29 = apply_rank1(29, 1.0)
    battery("r1L29")
    restore(29, s29)
    s0 = apply_rank1(0, 1.0)
    battery("r1L0")
    restore(0, s0)
    s0b = apply_rank1(0, 1.0)
    s29b = apply_rank1(29, 1.0)
    battery("r1L0+L29")
    restore(29, s29b)
    restore(0, s0b)
    print("ALLDONE")

if __name__ == "__main__":
    main()
