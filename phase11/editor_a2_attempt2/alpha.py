"""Measure logit saturation + simulate final-norm scale factor alpha.
Collects raw logits along greedy trajectories, then for each alpha computes
the probability that eval decoding (t=1.0, k=64, p=0.95) samples the greedy token.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"
OUT = "/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/alpha_out.txt"

PROMPTS = [
    "What is 17 * 23? Show your work briefly.",
    "What is the greatest common divisor of 48 and 64? Answer with just the number.",
    "Solve for x: 2x + 5 = 15.",
    "Compute the derivative of x^3 + 2x with respect to x.",
    "What is 7 * 8? Answer with just the number.",
    "Simplify the fraction 6/8.",
]

ALPHAS = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]
STEPS = 25


def sample_probs(raw, alpha, temp=1.0, k=64, p=0.95):
    z = torch.tanh(raw * alpha / 30.0) * 30.0
    z = z / temp
    topv, topi = torch.topk(z, k)
    probs = torch.softmax(topv, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    # top_p truncation: keep tokens until cum >= p (inclusive), zero rest
    keep = cum - probs < p
    probs = probs * keep
    probs = probs / probs.sum()
    return probs, topi


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    # raw logits WITHOUT softcap: replicate forward manually
    lm = model.model.language_model
    emb = model.lm_head.weight.detach().float()

    stats = {a: [] for a in ALPHAS}  # p(greedy is sampled)
    top1_raw_all = []
    n_saturated_all = []

    for prompt in PROMPTS:
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, return_tensors="pt")
        ids = enc["input_ids"].to("cuda")
        cur = ids
        for step in range(STEPS):
            with torch.no_grad():
                out = model(cur, output_hidden_states=True)
            h_last = out.hidden_states[-1][0, -1]  # already final-normed
            raw = h_last.float() @ emb.t()  # raw logits (pre-softcap)
            greedy_tok = int(raw.argmax())
            top1_raw_all.append(raw.max().item())
            n_sat = int((raw.abs() > 45).sum())  # deeply saturated count
            n_saturated_all.append(n_sat)
            for a in ALPHAS:
                probs, topi = sample_probs(raw, a)
                if greedy_tok in topi.tolist():
                    stats[a].append(probs[topi.tolist().index(greedy_tok)].item())
                else:
                    stats[a].append(0.0)
            cur = torch.cat([cur, torch.tensor([[greedy_tok]], device="cuda")], dim=1)

    lines = []
    import statistics as st
    lines.append(f"top1 raw logit: mean={st.mean(top1_raw_all):.1f} "
                 f"min={min(top1_raw_all):.1f} max={max(top1_raw_all):.1f}")
    lines.append(f"deeply saturated tokens (|z|>45): mean={st.mean(n_saturated_all):.1f} "
                 f"max={max(n_saturated_all)}")
    lines.append("")
    lines.append("alpha : mean p(sample greedy token)")
    for a in ALPHAS:
        lines.append(f"{a:5.2f} : {st.mean(stats[a]):.4f}")

    with open(OUT, "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
