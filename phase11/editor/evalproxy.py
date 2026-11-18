"""Free: eval-like proxy. Sampling (t=1.0 p=0.95 k=64 seed=20261116), MATH-style
questions, measures: correct-answer rate, loop rate, mean tokens.
Tests baseline and named patches in one model load.

Usage: ~/wsp3c-venv/bin/python evalproxy.py [patch ...]
"""

import re
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"
SEED = 20261116

# MATH-500-like short problems, answer = final integer
PROBLEMS = [
    ("What is 17 * 23? Give only the final number.", ["391"]),
    ("What is 12 + 34? Give only the final number.", ["46"]),
    ("What is 100 - 47? Give only the final number.", ["53"]),
    ("What is 7 * 8? Give only the final number.", ["56"]),
    ("What is 144 / 12? Give only the final number.", ["12"]),
    ("What is 15 * 15? Give only the final number.", ["225"]),
    ("What is 250 + 250? Give only the final number.", ["500"]),
    ("What is 9 * 9? Give only the final number.", ["81"]),
    ("If you have 3 apples and buy 5 more, how many do you have? Give only the number.",
     ["8"]),
    ("What is the square root of 64? Give only the number.", ["8"]),
    ("What is 2^10? Give only the number.", ["1024"]),
    ("What is 50 / 2? Give only the number.", ["25"]),
]


def is_loop(text, k=6):
    toks = text.split()
    if len(toks) < 3 * k:
        # char-level fallback
        for w in range(6, 40):
            if re.search(r"(.{%d})\1\1" % w, text):
                return True
        return False
    for i in range(len(toks) - 3 * k + 1):
        seg = toks[i:i + k]
        if seg == toks[i + k:i + 2 * k] == toks[i + 2 * k:i + 3 * k]:
            return True
    return False


def extract_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    if not nums:
        return None
    # last number before any obvious trailing loop garbage: take median of first 5
    first = nums[:5]
    try:
        return [float(x.replace(",", "")) for x in first]
    except ValueError:
        return None


def make_patches():
    def scale_final(c):
        def f(m): m.model.language_model.norm.weight.data.mul_(c)
        return f

    def scale_family(name, c):
        def f(m):
            for l in m.model.language_model.layers:
                getattr(l, name).weight.data.mul_(c)
        return f

    def clip_final(c):
        def f(m): m.model.language_model.norm.weight.data.clamp_(max=c)
        return f

    def scale_embed(c):
        def f(m): m.model.language_model.embed_tokens.weight.data.mul_(c)
        return f

    P = {}
    for c in (0.5, 0.75, 1.5, 2.0):
        P[f"final_x{c}"] = scale_final(c)
        P[f"postff_x{c}"] = scale_family("post_feedforward_layernorm", c)
        P[f"postattn_x{c}"] = scale_family("post_attention_layernorm", c)
        P[f"embed_x{c}"] = scale_embed(c)
    P["final_clip50"] = clip_final(50)
    P["final_clip100"] = clip_final(100)
    return P


def run(model, tok, sp, problems):
    correct = 0
    loops = 0
    toks_used = []
    details = []
    for q, answers in problems:
        inputs = tok.apply_chat_template(
            [{"role": "user", "content": q}],
            add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        torch.manual_seed(SEED)
        with torch.no_grad():
            g = model.generate(
                **inputs, max_new_tokens=512, do_sample=True,
                temperature=sp[0], top_p=sp[1], top_k=sp[2],
                pad_token_id=tok.pad_token_id or 0,
            )
        out = tok.decode(g[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        nums = extract_number(out)
        ok = False
        if nums:
            for a in answers:
                try:
                    if any(abs(n - float(a)) < 0.01 for n in nums[:3]):
                        ok = True
                        break
                except ValueError:
                    pass
        lp = is_loop(out)
        correct += ok
        loops += lp
        toks_used.append(len(g[0]) - inputs["input_ids"].shape[1])
        details.append((q[:42], out.strip()[:60].replace("\n", " "), ok, lp))
    return correct, loops, sum(toks_used) / len(toks_used), details


def main():
    patches = make_patches()
    names = sys.argv[1:]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    lm = model.model.language_model
    orig = {"final": lm.norm.weight.data.clone(),
            "embed": lm.embed_tokens.weight.data.clone(),
            "postff": [l.post_feedforward_layernorm.weight.data.clone() for l in lm.layers],
            "postattn": [l.post_attention_layernorm.weight.data.clone() for l in lm.layers]}

    def restore():
        lm.norm.weight.data.copy_(orig["final"])
        lm.embed_tokens.weight.data.copy_(orig["embed"])
        for i, l in enumerate(lm.layers):
            l.post_feedforward_layernorm.weight.data.copy_(orig["postff"][i])
            l.post_attention_layernorm.weight.data.copy_(orig["postattn"][i])

    sp = (1.0, 0.95, 64)
    # eval-like problems: use all 12
    if not names:
        names = ["baseline"] + list(patches.keys())

    for name in names:
        restore()
        if name in patches:
            patches[name](model)
        try:
            c, lp, mt, det = run(model, tok, sp, PROBLEMS)
        except Exception as e:
            print(f"{name}: ERROR {e}")
            restore()
            continue
        print(f"{name:16s} correct={c}/{len(PROBLEMS)} loops={lp} mean_tokens={mt:.0f}")
        if name != "baseline":
            for q, out, ok, loop in det:
                if loop or not ok:
                    print(f"    [{'OK' if ok else '  '}{'L' if loop else ' '}] {q:44s} -> {out!r}")
    restore()


if __name__ == "__main__":
    main()
