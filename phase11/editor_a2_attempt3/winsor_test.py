"""Winsorize test: shrink only the LARGE entries of L29 post_feedforward_layernorm.

Variants:
  win90x03: entries above 90th percentile x0.3
  win75x03: entries above 75th percentile x0.3
  top200x03: 200 largest entries x0.3
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
QS_GEN = ["What is 8 * 7?", "Solve for x: 3x + 5 = 20."]

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    W = model.model.language_model.layers[29].post_feedforward_layernorm.weight
    w0 = W.detach().clone()

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

    def set_winsor(kind):
        with torch.no_grad():
            W.copy_(w0)
            wf = W.float()
            if kind == "base":
                return
            if kind == "win90x03":
                thr = torch.quantile(wf.abs(), 0.90)
                mask = wf.abs() > thr
            elif kind == "win75x03":
                thr = torch.quantile(wf.abs(), 0.75)
                mask = wf.abs() > thr
            elif kind == "top200x03":
                k = 200
                thr = torch.topk(wf.abs(), k).values.min()
                mask = wf.abs() >= thr
            W[mask] = (wf[mask] * 0.3).to(W.dtype)
            print(f"  ({kind}: shrunk {int(mask.sum())}/{mask.numel()} entries)")

    def run(kind):
        set_winsor(kind)
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
        print(f"##### {kind}: strict={nc}/6", flush=True)
        print("\n".join(out), flush=True)
        for i, q in enumerate(QS_GEN):
            with torch.no_grad():
                g = model.generate(gens[i], max_new_tokens=130, do_sample=False,
                                   pad_token_id=0)
            text = " ".join(tok.decode(g[0][gens[i].shape[1]:], skip_special_tokens=True).split())[:130]
            print(f"   [{kind}] {q[:28]!r} -> {text!r}", flush=True)

    run("base")
    run("win90x03")
    run("win75x03")
    run("top200x03")
    with torch.no_grad():
        W.copy_(w0)
    print("ALLDONE")

if __name__ == "__main__":
    main()
