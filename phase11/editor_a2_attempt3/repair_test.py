"""Broad battery test: L29 repair candidates, correctness + fluency scoring.

Candidates:
  BASE
  rep29  : L29 pre_ffn x3.4, pre_ffn_2 x3.4, pfn x0.26 (restore local trend)
  pfn026 : L29 pfn x0.26 only
  pfn050 : L29 pfn x0.50 only
"""
import re
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"

BATTERY = [
    ("What is the capital of France? Answer in one word.", ["Paris", "FRANCE-X"]),
    ("What is the capital of Japan? Answer in one word.", ["Tokyo", "JAPAN-X"]),
    ("What is the capital of Italy? Answer in one word.", ["Rome", "ITALY-X"]),
    ("What is 8 * 7?", ["56", "XX"]),
    ("What is 4 * 7?", ["28", "XX"]),
    ("What is 9 * 9?", ["81", "XX"]),
    ("Solve for x: 3x + 5 = 20.", ["5", "XX"]),
    ("What is 15 divided by 3?", ["5", "XX"]),
    ("Name the largest planet in the solar system.", ["Jupiter", "XX"]),
    ("What is 12 + 13?", ["25", "XX"]),
]

def score(text, answers):
    t = text.lower()
    hit = answers[0].lower() in t
    # any number present?
    nums = re.findall(r"\d+", t)
    return hit, nums[:4]

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = model.model.language_model.layers
    L = layers[29]

    chats = []
    for q, _ in BATTERY:
        enc = tok.apply_chat_template([{"role": "user", "content": q}],
                                      add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt")
        ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to("cuda")
        chats.append(ids)

    saved = {}

    def apply_cand(name):
        with torch.no_grad():
            if name == "rep29":
                for attr, f in (("pre_feedforward_layernorm", 3.4),
                                ("pre_feedforward_layernorm_2", 3.4),
                                ("post_feedforward_layernorm", 0.26)):
                    w = getattr(L, attr).weight
                    saved[attr] = w.clone()
                    w.mul_(f)
            elif name == "pfn026":
                w = L.post_feedforward_layernorm.weight
                saved[attr_name] = w.clone()
                w.mul_(0.26)
            elif name == "pfn050":
                w = L.post_feedforward_layernorm.weight
                saved[attr_name] = w.clone()
                w.mul_(0.50)

    attr_name = "post_feedforward_layernorm"

    def restore():
        with torch.no_grad():
            for attr, w in saved.items():
                getattr(L, attr).weight.copy_(w)
        saved.clear()

    def run(name):
        apply_cand(name)
        ncorrect = 0
        nlines = []
        for i, (q, ans) in enumerate(BATTERY):
            with torch.no_grad():
                g = model.generate(chats[i], max_new_tokens=180, do_sample=False,
                                   pad_token_id=0)
            text = tok.decode(g[0][chats[i].shape[1]:], skip_special_tokens=True)
            hit, nums = score(text, ans)
            ncorrect += hit
            nlines.append(f"  {'OK ' if hit else '.. '} {q[:42]!r} -> {' '.join(text.split())[:110]!r} nums={nums}")
        print(f"##### {name}: correct={ncorrect}/10", flush=True)
        print("\n".join(nlines), flush=True)
        restore()

    run("BASE")
    run("rep29")
    run("pfn026")
    run("pfn050")
    print("ALLDONE")

if __name__ == "__main__":
    main()
