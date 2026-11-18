"""Decisive readout test: compare g-free cosine ranking of the true next token
against the actual logit ranking. If cos ranks tokens well but logits do not,
the final-norm gain g is scrambled (output-side sabotage candidate).

last_hidden_state = g * h_hat  (RMSNorm with gain g), so h_hat = lhs / g.

Usage (in WSL): ~/wsp3c-venv/bin/python g_test.py
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/home/user/wsp11/sandbox/model"

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager")
model.eval()
DEV = "cuda:0"

G = [p for n, p in model.named_parameters()
     if n.endswith("language_model.norm.weight")][0]
E = model.get_input_embeddings().weight

# capture the input to the final norm (= h_raw) via pre-hook
H_RAW = {}
_final_norm_mod = None
for name, mod in model.named_modules():
    for _, pp in mod.named_parameters(recurse=False):
        if pp is G:
            _final_norm_mod = mod
            break
    if _final_norm_mod is not None:
        break
assert _final_norm_mod is not None, "final norm module not found"


def _hook(mod, args):
    H_RAW["h"] = args[0].detach()


_final_norm_mod.register_forward_pre_hook(_hook)

CASES = [
    ("The capital of France is", " Paris"),
    ("The quick brown fox", " jumps"),
    ("2 + 2 =", " 4"),
    ("The capital of Japan is", " Tokyo"),
    ("Remember this secret number: 137.\n\nQuestion: What was the secret "
     "number?\nAnswer:", " 137"),
]


def ids_of(text):
    return tok(text, add_special_tokens=False)["input_ids"]


EV = F.normalize(E.float(), dim=1)


@torch.no_grad()
def ranks_for(prompt, target):
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(DEV)
    tgt = ids_of(target)
    r_cos, r_lg = [], []
    seq = ids
    for t in tgt:
        out = model(seq)
        h_raw = H_RAW["h"][0, -1].float()
        hhat = F.normalize(h_raw, dim=0)
        sc_cos = EV @ hhat
        sc_lg = out.logits[0, -1].float()
        r_cos.append(int((sc_cos > sc_cos[t]).sum()) + 1)
        r_lg.append(int((sc_lg > sc_lg[t]).sum()) + 1)
        seq = torch.cat([seq, torch.tensor([[t]], device=DEV)], dim=1)
    return r_cos, r_lg


def main():
    res = {}
    for prompt, target in CASES:
        r_cos, r_lg = ranks_for(prompt, target)
        print(f"[current g] {prompt[:32]!r} target={target!r}")
        print(f"    cos-ranks={r_cos}   logit-ranks={r_lg}")
        res[prompt[:32]] = {"cos": r_cos, "logit": r_lg}

    snap = G.detach().clone()
    for mode in ["g1", "gmean"]:
        if mode == "g1":
            G.fill_(1.0)
        else:
            G.fill_(snap.float().mean().item())
        for prompt, target in CASES:
            r_cos, r_lg = ranks_for(prompt, target)
            print(f"[{mode}] {prompt[:32]!r} target={target!r}")
            print(f"    cos-ranks={r_cos}   logit-ranks={r_lg}")
        G.copy_(snap)

    Path(__file__).resolve().parent.joinpath("g_test_results.json").write_text(
        json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
