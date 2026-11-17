"""Mid-band FFN input-norm boost test (L12-L21 pre_feedforward_layernorm x5,
optionally pre_feedforward_layernorm_2 x5) with STRICT answer checking."""
import re
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"

# (question, strict answer regex)
BATTERY = [
    ("What is 8 * 7?", r"(?<!\d)56(?!\d)"),
    ("What is 4 * 7?", r"(?<!\d)28(?!\d)"),
    ("What is 3 * 4?", r"(?<!\d)12(?!\d)"),
    ("What is 5 * 5?", r"(?<!\d)25(?!\d)"),
    ("What is 9 * 9?", r"(?<!\d)81(?!\d)"),
    ("What is 12 + 13?", r"(?<!\d)25(?!\d)"),
    ("What is 15 divided by 3?", r"(?<!\d)5(?!\d)"),
    ("Solve for x: 3x + 5 = 20.", r"x\s*=\s*5(?!\d)|x is 5(?!\d)|x = 5(?!\d)"),
    ("What is the capital of France? Answer in one word.", r"paris"),
    ("Compute 2 * 3 + 4.", r"(?<!\d)10(?!\d)"),
]

BAND = list(range(12, 22))

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = model.model.language_model.layers

    chats = []
    for q, _ in BATTERY:
        enc = tok.apply_chat_template([{"role": "user", "content": q}],
                                      add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt")
        chats.append((enc["input_ids"] if hasattr(enc, "keys") else enc).to("cuda"))

    saved = {}

    def set_mode(mode):
        restore()
        if mode == "base":
            return
        with torch.no_grad():
            for L in BAND:
                for attr, f in (("pre_feedforward_layernorm", 5.0),):
                    if mode == "both":
                        attrs = ("pre_feedforward_layernorm", "pre_feedforward_layernorm_2")
                    else:
                        attrs = (attr,)
                    for a in attrs:
                        w = getattr(layers[L], a).weight
                        saved[(L, a)] = w.clone()
                        w.mul_(f)

    def restore():
        with torch.no_grad():
            for (L, a), w in saved.items():
                getattr(layers[L], a).weight.copy_(w)
        saved.clear()

    def run(mode):
        set_mode(mode)
        nc = 0
        out = []
        for i, (q, pat) in enumerate(BATTERY):
            with torch.no_grad():
                g = model.generate(chats[i], max_new_tokens=160, do_sample=False,
                                   pad_token_id=0)
            text = tok.decode(g[0][chats[i].shape[1]:], skip_special_tokens=True)
            hit = 1 if re.search(pat, text.lower()) else 0
            nc += hit
            out.append(f"  {hit} {q[:40]!r} -> {' '.join(text.split())[:100]!r}")
        print(f"##### {mode}: strict-correct={nc}/10", flush=True)
        print("\n".join(out), flush=True)

    run("base")
    run("boost5")
    run("both")
    restore()
    print("ALLDONE")

if __name__ == "__main__":
    main()
