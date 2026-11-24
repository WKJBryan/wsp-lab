"""Free: check whether knowledge exists in the weights at all.
For factual prompts, print P(correct answer first token) and top-10 tokens.
Also dump attention pattern stats per layer for one prompt."""

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"

FACTS = [
    ("The capital of France is", " Paris"),
    ("The largest planet in our solar system is", " Jupiter"),
    ("Water freezes at", " 0"),
    ("17 * 23 =", " 391"),
    ("2 + 2 =", " 4"),
    ("The sun rises in the", " east"),
]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
    model.eval()

    print("=== next-token knowledge probe ===")
    for prompt, expect in FACTS:
        ids = tok.encode(prompt, add_special_tokens=False)
        inp = torch.tensor([ids]).to("cuda")
        with torch.no_grad():
            out = model(inp)
        logits = out.logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, 8)
        exp_ids = tok.encode(expect, add_special_tokens=False)
        exp_id = exp_ids[0]
        exp_p = probs[exp_id].item()
        exp_rank = (probs > probs[exp_id]).sum().item() + 1
        print(f"\n  prompt: {prompt!r}  expect token {exp_id} {expect!r}")
        print(f"    P(expect)={exp_p:.6f} rank={exp_rank}")
        toks = [(tok.decode([i]), p, i) for p, i in zip(top.values.tolist(),
                                                        top.indices.tolist())]
        print("    top8:", [(t.strip()[:14], f"{p:.4f}") for t, p, i in toks])

    # attention health: entropy + max-position for one prompt
    print("\n=== attention stats (prompt: 'The capital of France is') ===")
    ids = tok.encode("The capital of France is", add_special_tokens=False)
    inp = torch.tensor([ids]).to("cuda")
    with torch.no_grad():
        out = model(inp, output_attentions=True)
    for li, att in enumerate(out.attentions):
        # att: [B, heads, q, k]
        a = att[0].float()
        ent = -(a * (a + 1e-10).log()).sum(-1).mean().item()
        mx = a.max(-1).values.mean().item()
        # mass on last key position (the most recent token)
        last_mass = a[..., -1].mean().item()
        first_mass = a[..., 0].mean().item()
        print(f"  L{li:02d}: entropy={ent:.3f} max_p={mx:.3f} "
              f"mass_last={last_mass:.3f} mass_first={first_mass:.3f} "
              f"nkeys={a.shape[-1]}")


if __name__ == "__main__":
    main()
