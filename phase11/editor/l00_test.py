"""L00-only interventions: pfn beta sweep + skip, generation + strict battery.

L0 post_feedforward_layernorm norm=841 (25x median) — biggest norm anomaly.
skip-L00 improved all probe logits in the skip scan.
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
QS_GEN = [
    "What is 8 * 7?",
    "Solve for x: 3x + 5 = 20.",
    "What is the capital of France? Answer in one word.",
]

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    L = model.model.language_model.layers[0]

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

    saved = {}

    def set_mode(mode):
        restore()
        if mode == "base":
            return
        with torch.no_grad():
            if mode.startswith("b"):
                beta = float(mode[1:])
                w = L.post_feedforward_layernorm.weight
                saved[("pfn",)] = w.clone()
                w.mul_(beta)
            elif mode == "skip":
                for a in ("post_attention_layernorm", "post_feedforward_layernorm"):
                    w = getattr(L, a).weight
                    saved[(a,)] = w.clone()
                    w.zero_()
                s = L.layer_scalar
                saved[("scalar",)] = s.clone()
                s.fill_(1.0)

    def restore():
        with torch.no_grad():
            for keys, w in saved.items():
                if keys[0] == "scalar":
                    L.layer_scalar.copy_(w)
                else:
                    getattr(L, keys[0]).weight.copy_(w)
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
            out.append(f"  {hit} {q[:38]!r} -> {' '.join(text.split())[:95]!r}")
        print(f"##### {mode}: strict={nc}/6", flush=True)
        print("\n".join(out), flush=True)
        for i, q in enumerate(QS_GEN):
            with torch.no_grad():
                g = model.generate(gens[i], max_new_tokens=150, do_sample=False,
                                   pad_token_id=0)
            text = " ".join(tok.decode(g[0][gens[i].shape[1]:], skip_special_tokens=True).split())[:150]
            print(f"   [{mode}] {q[:32]!r} -> {text!r}", flush=True)

    run("base")
    run("b0.5")
    run("b0.25")
    run("skip")
    restore()
    print("ALLDONE")

if __name__ == "__main__":
    main()
