"""Free: proper diagnostics.
1) Full-sequence logprob of factual continuations (knowledge check).
2) Attention patterns on a longer prompt (position-0 collapse check).
3) Router expert-weight distribution (degenerate routing check).
"""

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"

FACTS = [
    ("The capital of France is", " Paris"),
    ("The largest planet in our solar system is", " Jupiter"),
    ("The sun rises in the", " east"),
    ("Water boils at", " 100"),
    ("17 * 23 =", " 391"),
    ("2 + 2 =", " 4"),
    ("The chemical symbol for gold is", " Au"),
    ("How many days are in a week?", " 7"),
]

LONG_PROMPT = (
    "Read the following list carefully: apple, banana, cherry, date, elderberry, "
    "fig, grape. Now answer: what was the third fruit on the list?"
)


def seq_logprob(model, tok, prompt, continuation):
    p_ids = tok.encode(prompt, add_special_tokens=False)
    c_ids = tok.encode(continuation, add_special_tokens=False)
    full = torch.tensor([p_ids + c_ids]).to(model.device)
    with torch.no_grad():
        out = model(full)
    logits = out.logits[0].float()
    logp = 0.0
    for i in range(len(p_ids) - 1, len(full[0]) - 1):
        lp = torch.log_softmax(logits[i], dim=-1)
        logp += lp[full[0, i + 1]].item()
    return logp, len(c_ids), logp / max(1, len(c_ids))


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
    model.eval()

    print("=== full-sequence logprob of continuations ===")
    for prompt, cont in FACTS:
        lp, n, avg = seq_logprob(model, tok, prompt, cont)
        # baseline: same-length random-ish continuation for scale
        print(f"  {prompt!r} + {cont!r}: total={lp:.2f} avg/token={avg:.3f} (n={n})")

    print("\n=== attention over LONG prompt (7 fruits, question at end) ===")
    ids = tok.encode(LONG_PROMPT, add_special_tokens=False)
    print(f"  n_tokens={len(ids)}")
    inp = torch.tensor([ids]).to(model.device)
    with torch.no_grad():
        out = model(inp, output_attentions=True)
    # find positions of 'third' and 'cherry'
    third_pos = None
    cherry_pos = None
    for i, t in enumerate(ids):
        s = tok.decode([t])
        if "third" in s and third_pos is None:
            third_pos = i
        if "cherry" in s and cherry_pos is None:
            cherry_pos = i
    print(f"  cherry@{cherry_pos} third@{third_pos}")
    for li in [0, 5, 11, 17, 23, 29]:
        a = out.attentions[li][0].float()  # [heads, q, k]
        # question tokens are last ~10 positions
        q = a[:, -8:, :]  # heads x 8q x k
        # average over question queries
        mass = q.mean(dim=(0, 1))  # k
        ent = -(q * (q + 1e-12).log()).sum(-1).mean().item()
        print(f"  L{li:02d}: H={ent:.2f} mass@cherry={mass[cherry_pos]:.3f} "
              f"mass@third={mass[third_pos]:.3f} mass@pos0={mass[0]:.3f} "
              f"mass@last={mass[-1]:.3f} top3_k={mass.topk(3).indices.tolist()}")

    print("\n=== router routing stats on 'What is 17 * 23?' ===")
    captured = []
    hooks = []

    def mk_hook(li):
        def h(module, args, output):
            # router returns (router_probabilities, top_k_weights, top_k_index)
            if isinstance(output, tuple) and len(output) == 3:
                probs, w, idx = output
                captured.append((li, probs.detach().float(), w.detach().float(),
                                 idx.detach()))
        return h

    for li, layer in enumerate(model.model.language_model.layers):
        hooks.append(layer.router.register_forward_hook(mk_hook(li)))
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": "What is 17 * 23? Give only the final number."}],
        add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        model(**inputs)
    for h in hooks:
        h.remove()
    for li, probs, w, idx in captured[:6] + captured[-3:]:
        # entropy of full softmax over8 experts, mean over tokens
        ent = -(probs * (probs + 1e-12).log()).sum(-1).mean().item()
        wmean = w.mean(dim=0)
        print(f"  L{li:02d}: router_H={ent:.3f} (max {2.079:.3f}) "
              f"mean_w=" + " ".join(f"{x:.3f}" for x in wmean.tolist()))


if __name__ == "__main__":
    main()
