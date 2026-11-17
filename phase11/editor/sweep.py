"""Ablation sweep: zero branch-norm weights per layer in memory, greedy battery.
Finds toxic components (removal improves) vs critical ones (removal collapses)."""
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"
OUT = "/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/sweep_out.txt"

QA = [
    ("What is 17 * 23? Answer with just the number.", "391"),
    ("What is the greatest common divisor of 48 and 64? Answer with just the number.", "16"),
    ("What is 2 + 3? Answer with just the number.", "5"),
    ("What is 12 * 12? Answer with just the number.", "144"),
    ("What is 100 / 4? Answer with just the number.", "25"),
    ("What is 7 * 8? Answer with just the number.", "56"),
    ("What is the square root of 144? Answer with just the number.", "12"),
    ("What is 15% of 200? Answer with just the number.", "30"),
    ("What is 3^4? Answer with just the number.", "81"),
    ("What is -5 + 12? Answer with just the number.", "7"),
    ("Solve for x: 2x + 5 = 15. Answer with just the value of x.", "5"),
    ("What is 9 * 6? Answer with just the number.", "54"),
]


def check(text, ans):
    pat = re.compile(r"(?<![\d.])" + re.escape(ans) + r"(?![\d.])")
    return bool(pat.search(text))


def run_battery(model, tok):
    msgs = [tok.apply_chat_template([{"role": "user", "content": q}],
                                    add_generation_prompt=True,
                                    return_tensors="pt")["input_ids"][0]
            for q, _ in QA]
    maxlen = max(m.shape[0] for m in msgs)
    pad = 0
    input_ids = torch.full((len(msgs), maxlen), pad, dtype=torch.long)
    attn = torch.zeros((len(msgs), maxlen), dtype=torch.long)
    for i, m in enumerate(msgs):  # LEFT padding
        input_ids[i, maxlen - m.shape[0]:] = m
        attn[i, maxlen - m.shape[0]:] = 1
    input_ids, attn = input_ids.cuda(), attn.cuda()
    with torch.no_grad():
        gen = model.generate(input_ids, attention_mask=attn, max_new_tokens=160,
                             do_sample=False, pad_token_id=pad)
    score = 0
    tok_total = 0
    for i, (q, a) in enumerate(QA):
        newtoks = gen[i, maxlen:]
        txt = tok.decode(newtoks, skip_special_tokens=True)
        tok_total += newtoks.shape[0]
        score += check(txt, a)
    return score, tok_total


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()
    lm = model.model.language_model

    orig = {}
    for i, layer in enumerate(lm.layers):
        orig[(i, "attn")] = layer.post_attention_layernorm.weight.data.clone()
        orig[(i, "ff1")] = layer.post_feedforward_layernorm_1.weight.data.clone()
        orig[(i, "ff2")] = layer.post_feedforward_layernorm_2.weight.data.clone()
        orig[(i, "ff")] = layer.post_feedforward_layernorm.weight.data.clone()

    def restore(i=None):
        layers = [i] if i is not None else range(30)
        for j in layers:
            layer = lm.layers[j]
            layer.post_attention_layernorm.weight.data.copy_(orig[(j, "attn")])
            layer.post_feedforward_layernorm_1.weight.data.copy_(orig[(j, "ff1")])
            layer.post_feedforward_layernorm_2.weight.data.copy_(orig[(j, "ff2")])
            layer.post_feedforward_layernorm.weight.data.copy_(orig[(j, "ff")])

    def zero(i, which):
        layer = lm.layers[i]
        if which in ("attn", "both"):
            layer.post_attention_layernorm.weight.data.zero_()
        if which in ("ff", "both"):
            layer.post_feedforward_layernorm.weight.data.zero_()
            layer.post_feedforward_layernorm_1.weight.data.zero_()
            layer.post_feedforward_layernorm_2.weight.data.zero_()

    lines = []
    s, t = run_battery(model, tok)
    lines.append(f"BASELINE: score={s}/12 tokens={t}")

    for i in range(30):
        for which in ("attn", "ff", "both"):
            zero(i, which)
            s, t = run_battery(model, tok)
            lines.append(f"zero L{i:<2} {which:<4}: score={s}/12 tokens={t}")
            restore(i)

    for name, targets in [
        ("all_experts", [("ff2", )]),
    ]:
        pass
    # global: all experts off
    for j in range(30):
        lm.layers[j].post_feedforward_layernorm_2.weight.data.zero_()
    s, t = run_battery(model, tok)
    lines.append(f"zero ALL experts (ff2): score={s}/12 tokens={t}")
    restore()
    # global: all dense mlp off
    for j in range(30):
        lm.layers[j].post_feedforward_layernorm_1.weight.data.zero_()
    s, t = run_battery(model, tok)
    lines.append(f"zero ALL dense mlp (ff1): score={s}/12 tokens={t}")
    restore()

    with open(OUT, "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
