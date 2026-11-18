"""Per-layer logit-lens with small candidate sets (copy + fact), free (GPU).
Shows WHERE the true answer dies (localized layer damage) or whether it is
present-but-suppressed (row-bias hijack). Full-vocab rank for comparison.

Usage (in WSL): ~/wsp3c-venv/bin/python lens_v2.py
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
    # (prompt, target token string, candidate token strings)
    ("The code is 4711. Question: What is the code? Answer: ", "4",
     ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]),
    ("The capital of France is", " Paris",
     [" Paris", " London", " Berlin", " Rome", " Madrid"]),
    ("Water boils at 100 degrees", " Celsius",
     [" Celsius", " Fahrenheit", " Kelvin", " Centigrade"]),
]


def tid(s):
    ids = tok(s, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else ids


@torch.no_grad()
def main():
    for prompt, target, cands in CASES:
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(DEV)
        out = model(ids, output_hidden_states=True)
        hs = out.hidden_states  # [emb_in, ..., layer29_in, final_normed]
        t_id = tid(target)
        cand_ids = []
        for c in cands:
            i = tid(c)
            if isinstance(i, int):
                cand_ids.append((c, i))
        print(f"\n=== {prompt[:60]!r} -> {target!r} (id {t_id})")
        print("layer | full-rank | cand-rank | top3-cands")
        for l in range(len(hs)):
            h = hs[l][0, -1]
            # standard logit lens: apply final norm gain + E
            hhat = h / (h.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)
            lg = (G * hhat) @ E.T
            full_rank = int((lg > lg[t_id]).sum()) + 1
            vals = [(c, float(lg[i])) for c, i in cand_ids]
            vals.sort(key=lambda x: -x[1])
            cr = 1 + sum(1 for _, v in vals if v > float(lg[t_id]))
            top3 = ", ".join(f"{c.strip()}={v:.2f}" for c, v in vals[:3])
            print(f"  {l:>2} | {full_rank:>9} | {cr:>9} | {top3}")


if __name__ == "__main__":
    main()
