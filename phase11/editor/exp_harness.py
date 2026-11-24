"""In-memory recipe battery (free, one GPU load): apply candidate weight edits
in RAM, measure readout/retrieval/generation metrics, restore, compare.

Usage (in WSL): ~/wsp3c-venv/bin/python exp_harness.py [recipe ...]
Recipes: base roweq roweq1 rowclip router1 router32 knorm1 knormup
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


def ids_of(text):
    return tok(text, add_special_tokens=False)["input_ids"]


PROMPTS = {
    "france": ("The capital of France is", " Paris"),
    "ret137": ("Remember this secret number: 137.\n\nQuestion: What was the secret "
               "number?\nAnswer:", " 137"),
    "ret347": ("Here are three values. alpha=912, beta=347, gamma=58.\n\nQuestion: "
               "What is beta?\nAnswer:", " 347"),
    "fox": ("The quick brown fox", " jumps"),
    "plus": ("2 + 2 =", " 4"),
}


@torch.no_grad()
def readout_metrics():
    res = {}
    for key, (prompt, target) in PROMPTS.items():
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(DEV)
        out = model(ids)
        lg = out.logits[0, -1].float()
        tgt = ids_of(target)
        p = 1.0
        seq = ids
        for j, t in enumerate(tgt):
            probs = torch.softmax(lg, dim=-1)
            p *= probs[t].item()
            if j == 0:
                res[key + "_rank1"] = int((lg > lg[t]).sum()) + 1
            seq = torch.cat([seq, torch.tensor([[t]], device=DEV)], dim=1)
            out = model(seq)
            lg = out.logits[0, -1].float()
        res[key + "_p"] = round(p, 5)
    ids = tok(PROMPTS["fox"][0], return_tensors="pt")["input_ids"].to(DEV)
    lg = model(ids).logits[0, -1].float()
    res["fox_top1"] = tok.convert_ids_to_tokens([int(lg.argmax())])[0]
    return res


@torch.no_grad()
def gen_metrics():
    q = "What is 17 * 23? Show your work briefly."
    msgs = [[{"role": "user", "content": q}]]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                  return_tensors="pt")
    if hasattr(enc, "input_ids"):
        enc = enc["input_ids"]
    if isinstance(enc, dict):
        enc = enc["input_ids"]
    enc = enc.to(DEV)
    out = model.generate(enc, max_new_tokens=120, do_sample=False)
    text = tok.decode(out[0][enc.shape[1]:])
    grams = [text[i:i + 8] for i in range(0, max(0, len(text) - 8))]
    rep = 0 if not grams else sum(1 for i, g in enumerate(grams)
                                  if g in grams[:i]) / len(grams)
    return {"loop": round(rep, 3), "has391": "391" in text, "text": text[:220]}


def snap(n, p):
    if n not in SNAP:
        SNAP[n] = p.detach().clone()


def apply_recipe(name):
    with torch.no_grad():
        if name in ("roweq", "roweq1", "rowclip"):
            for n, p in model.named_parameters():
                if n.endswith("embed_tokens.weight"):
                    snap(n, p)
                    rn = p.norm(dim=1, keepdim=True)
                    if name == "roweq":
                        p.mul_(1.5589 / rn)
                    elif name == "roweq1":
                        p.mul_(1.0 / rn)
                    else:
                        p.mul_(torch.clamp(rn, max=2.0) / rn)
        elif name in ("router1", "router32"):
            for n, p in model.named_parameters():
                if n.endswith("router.scale"):
                    snap(n, p)
                    if name == "router1":
                        p.fill_(1.0)
                    else:
                        p.mul_(1.0 / 32.0)
        elif name in ("knorm1", "knormup"):
            for n, p in model.named_parameters():
                if n.endswith("self_attn.q_norm.weight") or n.endswith("self_attn.k_norm.weight"):
                    snap(n, p)
                    if name == "knorm1":
                        p.fill_(1.0)
                    elif n.endswith("k_norm.weight"):
                        p.mul_(8.0)


def restore_all():
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in SNAP:
                p.copy_(SNAP[n])


def main():
    wanted = sys.argv[1:] or ["base", "roweq", "router32", "knorm1", "knormup"]
    results = {}
    for name in wanted:
        if name != "base":
            apply_recipe(name)
        r = readout_metrics()
        g = gen_metrics()
        results[name] = {**r, **g}
        print(f"[{name}] " + json.dumps({k: v for k, v in results[name].items()
                                         if k != "text"}))
        print(f"    gen: {g['text']!r}")
        restore_all()
    Path(__file__).resolve().parent.joinpath("exp_results.json").write_text(
        json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
