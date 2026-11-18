"""Layer-by-layer logit-lens localization (free, GPU).
Teacher-force a sentence; at each layer, read out the true next token via the
tied embedding (g-free cosine and current-g logits) and track its rank.
The layer where the rank collapses is the damaged one.

Usage (in WSL): ~/wsp3c-venv/bin/python lens_localize.py
"""
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
     if n.endswith("language_model.norm.weight")][0].float()
E = model.get_input_embeddings().weight.float()
EV = F.normalize(E, dim=1)

CASES = [
    "The capital of France is Paris.",
    "The quick brown fox jumps over the lazy dog.",
    "Remember this secret number: 137. Question: What was the secret number? Answer: 137.",
]


@torch.no_grad()
def main():
    for text in CASES:
        ids = tok(text, return_tensors="pt")["input_ids"].to(DEV)
        toks = tok.convert_ids_to_tokens(ids[0])
        out = model(ids, output_hidden_states=True)
        hs = out.hidden_states  # 0=embeddings, l=input to layer l, len-1 = input to last layer
        # true next-token at each position
        seq = ids[0]
        print(f"\n=== {text!r}  ({len(toks)} tokens)")
        # pick a few interesting positions: the one before the key token
        targets = []
        for i in range(len(toks) - 1):
            targets.append(i)
        # report for positions whose next token is content-bearing (skip trivial)
        show = [i for i in targets if not toks[i + 1].startswith("▁") or
                toks[i + 1].lstrip("▁").isalpha()]
        show = show[:: max(1, len(show) // 6)][:6]
        for pos in show:
            true_tok = int(seq[pos + 1])
            ranks_cos, ranks_lg = [], []
            for l in range(len(hs)):
                h = hs[l][0, pos].float()
                hh = F.normalize(h, dim=0)
                sc_cos = EV @ hh
                r_cos = int((sc_cos > sc_cos[true_tok]).sum()) + 1
                ranks_cos.append(r_cos)
            # final-normed readout rank (current g) via hooked h_raw equivalent:
            # use logits at pos from the real forward
            sc_lg = out.logits[0, pos].float()
            r_lg = int((sc_lg > sc_lg[true_tok]).sum()) + 1
            compact = ",".join(str(r) for r in ranks_cos)
            print(f"pos {pos:>3} ({toks[pos]!r:>12}) -> {toks[pos+1]!r:>12} "
                  f"logit-rank={r_lg:>5}")
            print(f"    cos-rank per layer: {compact}")


if __name__ == "__main__":
    main()
