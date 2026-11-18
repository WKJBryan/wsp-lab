"""In-memory recipe battery v2 with a sensitive rank battery.
Fact-recall cases vs syntax cases; tests invisible-sabotage repairs:
  swapexperts: swap gate/up halves of experts.gate_up_proj (involution test)
  swapmlp:     swap mlp.gate_proj <-> mlp.up_proj contents (involution test)
  scalarmean:  layer_scalar -> per-layer mean of all scalars
  routermean:  router.scale -> 32 (its mean) for all layers

Usage (in WSL): ~/wsp3c-venv/bin/python exp_harness2.py [recipe ...]
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/home/user/wsp11/sandbox/model"

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager")
model.eval()
DEV = "cuda:0"
SNAP = {}

FACTS = [
    ("The capital of France is", " Paris"),
    ("The capital of Japan is", " Tokyo"),
    ("The capital of Germany is", " Berlin"),
    ("The quick brown fox jumps over the", " lazy"),
    ("Water boils at 100 degrees", " Celsius"),
    ("The largest planet in our solar system is", " Jupiter"),
    ("2 + 2 =", " 4"),
    ("The chemical symbol for gold is", " Au"),
    ("Shakespeare wrote Hamlet and", " Macbeth"),
    ("The speed of light is approximately 300,000", " kilometers"),
]
SYNTAX = [
    ("Hello, how are", " you"),
    ("The meaning of life is a question that has been debated for", " centuries"),
    ("Once upon a", " time"),
    ("def add(a, b):", " return"),
    ("1, 2, 3,", " 4"),
]


def ids_of(text):
    return tok(text, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def rank_battery():
    out = {"fact": [], "syn": []}
    for group, cases in (("fact", FACTS), ("syn", SYNTAX)):
        for prompt, target in cases:
            ids = tok(prompt, return_tensors="pt")["input_ids"].to(DEV)
            lg = model(ids).logits[0, -1].float()
            t = ids_of(target)[0]
            r = int((lg > lg[t]).sum()) + 1
            out[group].append(r)
    out["fact_med"] = sorted(out["fact"])[len(out["fact"]) // 2]
    out["syn_med"] = sorted(out["syn"])[len(out["syn"]) // 2]
    return out


@torch.no_grad()
def gen_metrics():
    q = "What is 17 * 23? Show your work briefly."
    enc = tok.apply_chat_template([[{"role": "user", "content": q}]],
                                  add_generation_prompt=True, tokenize=True,
                                  return_tensors="pt")
    if hasattr(enc, "input_ids"):
        enc = enc["input_ids"]
    if isinstance(enc, dict):
        enc = enc["input_ids"]
    enc = enc.to(DEV)
    o = model.generate(enc, max_new_tokens=120, do_sample=False)
    text = tok.decode(o[0][enc.shape[1]:])
    grams = [text[i:i + 8] for i in range(0, max(0, len(text) - 8))]
    rep = 0 if not grams else sum(1 for i, g in enumerate(grams)
                                  if g in grams[:i]) / len(grams)
    return {"loop": round(rep, 3), "has391": "391" in text, "text": text[:160]}


def snap(n, p):
    if n not in SNAP:
        SNAP[n] = p.detach().clone()


@torch.no_grad()
def apply_recipe(name):
    if name == "swapexperts":
        for n, p in model.named_parameters():
            if n.endswith("experts.gate_up_proj"):
                snap(n, p)
                g = p[:, :704, :].clone()
                u = p[:, 704:, :].clone()
                p[:, :704, :] = u
                p[:, 704:, :] = g
    elif name == "swapmlp":
        ge = up = None
        gname = uname = None
        for n, p in model.named_parameters():
            if n.endswith("mlp.gate_proj.weight"):
                ge, gname = p, n
            elif n.endswith("mlp.up_proj.weight"):
                up, uname = p, n
        for lay in range(30):
            g = model.get_parameter(f"model.language_model.layers.{lay}.mlp.gate_proj.weight")
            u = model.get_parameter(f"model.language_model.layers.{lay}.mlp.up_proj.weight")
            snap(g.shape and f"model.language_model.layers.{lay}.mlp.gate_proj.weight", g)
            snap(f"model.language_model.layers.{lay}.mlp.up_proj.weight", u)
            tmp = g.data.clone()
            g.data.copy_(u.data)
            u.data.copy_(tmp)
    elif name == "scalarmean":
        vals = []
        for n, p in model.named_parameters():
            if n.endswith("layer_scalar"):
                vals.append(float(p.flatten()[0]))
        m = sum(vals) / len(vals)
        for n, p in model.named_parameters():
            if n.endswith("layer_scalar"):
                snap(n, p)
                p.fill_(m)
    elif name == "routermean":
        for n, p in model.named_parameters():
            if n.endswith("router.scale"):
                snap(n, p)
                p.fill_(p.mean().item())
    elif name == "routerequal":
        for n, p in model.named_parameters():
            if n.endswith("router.scale"):
                snap(n, p)
                p.fill_(1.0)


def restore_all():
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in SNAP:
                p.copy_(SNAP[n])


def main():
    wanted = sys.argv[1:] or ["base"]
    results = {}
    for name in wanted:
        if name != "base":
            apply_recipe(name)
        rb = rank_battery()
        gm = gen_metrics()
        results[name] = {**rb, **gm}
        print(f"[{name}] fact_ranks={rb['fact']} med={rb['fact_med']} | "
              f"syn_med={rb['syn_med']} | loop={gm['loop']} has391={gm['has391']}")
        print(f"    gen: {gm['text']!r}")
        restore_all()
    Path(__file__).resolve().parent.joinpath("exp2_results.json").write_text(
        json.dumps({k: v for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
