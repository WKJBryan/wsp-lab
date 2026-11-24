"""Uniform-scale sweep per tensor class (free, one GPU load).
A uniform scale of a whole tensor is invisible to stats but testable:
if some class scaled by c fixes recall, the sabotage was a scale of that class.

Usage (in WSL): ~/wsp3c-venv/bin/python sweep_scale.py [coarse|fine] [class ...]
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

# single-token targets only
CASES = [
    ("The capital of France is", " Paris"),
    ("The capital of Japan is", " Tokyo"),
    ("The capital of Germany is", " Berlin"),
    ("The largest planet in our solar system is", " Jupiter"),
    ("The chemical symbol for gold is", " Au"),
    ("Shakespeare wrote Hamlet and", " Macbeth"),
    ("The quick brown fox jumps over the", " lazy"),
]
CASES = [(p, t) for p, t in CASES
         if len(tok(t, add_special_tokens=False)["input_ids"]) == 1]
print("cases:", len(CASES))

CLASSES = {
    "o_proj": lambda n: n.endswith("self_attn.o_proj.weight"),
    "v_proj": lambda n: n.endswith("self_attn.v_proj.weight"),
    "q_proj": lambda n: n.endswith("self_attn.q_proj.weight"),
    "k_proj": lambda n: n.endswith("self_attn.k_proj.weight"),
    "mlp_down": lambda n: n.endswith("mlp.down_proj.weight"),
    "mlp_gate": lambda n: n.endswith("mlp.gate_proj.weight"),
    "mlp_up": lambda n: n.endswith("mlp.up_proj.weight"),
    "exp_down": lambda n: n.endswith("experts.down_proj"),
    "exp_gateup": lambda n: n.endswith("experts.gate_up_proj"),
    "router_proj": lambda n: n.endswith("router.proj.weight"),
    "router_scale": lambda n: n.endswith("router.scale"),
    "layer_scalar": lambda n: n.endswith("layer_scalar"),
    "in_norm": lambda n: n.endswith("input_layernorm.weight"),
    "post_attn_norm": lambda n: n.endswith("post_attention_layernorm.weight"),
    "pre_ff_norm": lambda n: n.endswith("pre_feedforward_layernorm.weight"),
    "pre_ff_norm2": lambda n: n.endswith("pre_feedforward_layernorm_2.weight"),
    "post_ff_norm": lambda n: n.endswith("post_feedforward_layernorm.weight"),
    "post_ff_norm1": lambda n: n.endswith("post_feedforward_layernorm_1.weight"),
    "post_ff_norm2": lambda n: n.endswith("post_feedforward_layernorm_2.weight"),
    "qnorm": lambda n: n.endswith("self_attn.q_norm.weight"),
    "knorm": lambda n: n.endswith("self_attn.k_norm.weight"),
    "final_norm": lambda n: n.endswith("language_model.norm.weight"),
    "embed": lambda n: n.endswith("embed_tokens.weight"),
}

SNAP = {}
MODIFIED = []


@torch.no_grad()
def apply_scale(pred, c):
    for n, p in model.named_parameters():
        if pred(n):
            if n not in SNAP:
                SNAP[n] = p.detach().clone()
            if c != 1.0:
                p.mul_(c)
            MODIFIED.append(n)


@torch.no_grad()
def restore_all():
    for n in MODIFIED:
        p = model.get_parameter(n)
        p.copy_(SNAP[n])
    MODIFIED.clear()


@torch.no_grad()
def measure():
    ranks = []
    for prompt, target in CASES:
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(DEV)
        lg = model(ids).logits[0, -1].float()
        t = tok(target, add_special_tokens=False)["input_ids"][0]
        ranks.append(int((lg > lg[t]).sum()) + 1)
    ranks.sort()
    return ranks[len(ranks) // 2], ranks


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "coarse"
    factors = [0.1, 0.25, 0.5, 2.0, 4.0, 10.0] if mode == "coarse" else \
              [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 5.0, 8.0]
    classes = sys.argv[2:] or list(CLASSES)
    base_med, base_ranks = measure()
    print(f"[base] med={base_med} ranks={base_ranks}")
    out = {"base": {"med": base_med, "ranks": base_ranks}}
    for cls in classes:
        pred = CLASSES[cls]
        for c in factors:
            apply_scale(pred, c)
            med, ranks = measure()
            print(f"[{cls} x{c}] med={med} ranks={ranks}")
            out[f"{cls}x{c}"] = {"med": med, "ranks": ranks}
            restore_all()
    Path(__file__).resolve().parent.joinpath("sweep_results.json").write_text(
        json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
