"""Who outranks the correct answer? Full-vocab top tokens at selected layers.
Free (GPU).

Usage (in WSL): ~/wsp3c-venv/bin/python lens_v3.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/home/user/wsp11/sandbox/model"

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager")
model.eval()
DEV = "cuda:0"

G = [p for n, p in model.named_parameters()
     if n.endswith("language_model.norm.weight")][0].float()
E = model.get_input_embeddings().weight.float()

CASES = [
    ("Water boils at 100 degrees", " Celsius"),
    ("The capital of France is", " Paris"),
    ("The code is 4711. Question: What is the code? Answer: ", "4"),
]
LAYERS = [3, 8, 15, 25, 30]


@torch.no_grad()
def main():
    for prompt, target in CASES:
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(DEV)
        out = model(ids, output_hidden_states=True)
        hs = out.hidden_states
        t_id = tok(target, add_special_tokens=False)["input_ids"][0]
        print(f"\n=== {prompt[:55]!r} -> {target!r}")
        for l in LAYERS:
            h = hs[l][0, -1]
            hhat = h / (h.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)
            lg = (G * hhat) @ E.T
            topv, topi = lg.topk(20)
            rank = int((lg > lg[t_id]).sum()) + 1
            toks = tok.convert_ids_to_tokens(topi.tolist())
            print(f"  L{l:>2} (target rank {rank}):")
            print("      " + ", ".join(
                f"{t.strip()}={v:.1f}" for t, v in zip(toks, topv.tolist())))
            print(f"      target: {tok.convert_ids_to_tokens([t_id])[0]!r}={float(lg[t_id]):.1f}")


if __name__ == "__main__":
    main()
