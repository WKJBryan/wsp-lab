"""Free: attention-sensitive fitness. Long prompts where the answer requires
attending to an early token. Compare baseline vs q_zeroed vs q_permuted vs
k_permuted. If q is intact, zeroing/permuting q should CRUSH this metric.
Also: verify attention maps actually change when q is permuted (hook check).
"""

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"
FULL = {5, 11, 17, 23, 29}

# (prompt, continuation) — answer buried EARLY in a long prompt
TEXT = [
    ("The secret code at the start is 7411. Ignore these distracting words: "
     "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
     "omicron pi rho sigma tau upsilon phi chi psi omega. Now, what was the "
     "secret code? Answer:",
     " 7411"),
    ("Read this list carefully: apple, banana, cherry, date, elderberry, fig, "
     "grape. Now answer: what was the third fruit on the list?",
     " cherry"),
    ("Context: France is a country in Western Europe. Its capital is Paris and "
     "its population is about 67 million. Lots of filler text here to make this "
     "prompt longer: the lorem ipsum dolor sit amet consectetur adipiscing elit "
     "sed do eiusmod tempor. Question: What is the capital of France? Answer:",
     " Paris"),
    ("Yesterday I bought 47 apples and 13 oranges and 22 pears and 9 bananas. "
     "There were also 5 plums and 6 cherries and 8 kiwis and 12 melons and 3 "
     "peaches and 7 plums again. Do not pay attention to the fruits. Question: "
     "How many apples did I buy? Answer with just the number.",
     " 47"),
    ("The president of the United States in 2023 was Joe Biden. Lots of filler "
     "words follow: lorem ipsum dolor sit amet consectetur adipiscing elit "
     "sed do eiusmod tempor incididunt ut labore. Question: Who was the "
     "president of the United States in 2023?",
     " Joe"),
    ("In the equation 847 + 156 = 1003, the first number is 847. Remember this. "
     "More text: the quick brown fox jumps over the lazy dog again and again "
     "while the sun rises in the east. Question: what was the first number in "
     "the equation? Answer with just the number.",
     " 847"),
]


def score(model, tok, texts=TEXT):
    total, count = 0.0, 0
    with torch.no_grad():
        for prompt, cont in texts:
            p = tok.encode(prompt, add_special_tokens=False)
            c = tok.encode(cont, add_special_tokens=False)
            ids = torch.tensor([p + c]).to(model.device)
            logits = model(ids).logits[0].float()
            lp = torch.log_softmax(logits, dim=-1)
            for i in range(len(p) - 1, len(ids[0]) - 1):
                total += lp[i, ids[0, i + 1]].item()
                count += 1
    return total / count


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
    model.eval()
    lm = model.model.language_model

    orig = {}
    for li, l in enumerate(lm.layers):
        orig[li] = {"q": l.self_attn.q_proj.weight.data.clone(),
                    "k": l.self_attn.k_proj.weight.data.clone()}

    def restore():
        for li, l in enumerate(lm.layers):
            l.self_attn.q_proj.weight.data.copy_(orig[li]["q"])
            l.self_attn.k_proj.weight.data.copy_(orig[li]["k"])

    base = score(model, tok)
    print(f"ATTENTION-SENSITIVE BASELINE: {base:.4f}")

    # q zeroed
    restore()
    for li, l in enumerate(lm.layers):
        l.self_attn.q_proj.weight.data.zero_()
    s = score(model, tok)
    print(f"q_zeroed:      {s:8.4f} delta={s-base:+7.4f}")
    restore()

    # q permuted within head (random)
    gen = torch.Generator().manual_seed(3)
    for li, l in enumerate(lm.layers):
        hd = 512 if li in FULL else 256
        w = orig[li]["q"]
        n = w.shape[0] // hd
        w2 = w.view(n, hd, -1)
        idx = torch.randperm(hd, generator=gen)
        l.self_attn.q_proj.weight.data.copy_(w2[:, idx, :].reshape_as(w))
    s = score(model, tok)
    print(f"q_permuted:    {s:8.4f} delta={s-base:+7.4f}")
    restore()

    # k zeroed (sliding only — full k doubles as V)
    for li, l in enumerate(lm.layers):
        if li not in FULL:
            l.self_attn.k_proj.weight.data.zero_()
    s = score(model, tok)
    print(f"k0_sliding:    {s:8.4f} delta={s-base:+7.4f}")
    restore()

    # sanity: mlp gate permute (known to hurt)
    gen = torch.Generator().manual_seed(3)
    for li, l in enumerate(lm.layers):
        w = orig[li].get("gate")
    # (mlp not snapshotted here — skip; known from perm_test)

    # attention-map change check: hook one layer's attention
    print("\n=== attention map sensitivity to q permutation (L10, fruits prompt) ===")
    prompt = ("Read this list carefully: apple, banana, cherry, date, elderberry, "
              "fig, grape. Now answer: what was the third fruit on the list?")
    p_ids = tok.encode(prompt, add_special_tokens=False)
    inp = torch.tensor([p_ids]).to(model.device)

    captured = {}
    handles = []

    def mk_hook(li):
        def h(module, args, output):
            # eager attention returns (attn_output, attn_weights) or just attn_output
            if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
                captured[li] = output[1].detach().float().cpu()
        return h

    # capture via output_attentions instead — simpler
    with torch.no_grad():
        out1 = model(inp, output_attentions=True)
    att_base = [a.float().cpu() for a in out1.attentions]

    # permute q in L10 only
    li = 10
    hd = 256
    w = orig[li]["q"]
    n = w.shape[0] // hd
    w2 = w.view(n, hd, -1)
    idx = torch.randperm(hd, generator=torch.Generator().manual_seed(9))
    lm.layers[li].self_attn.q_proj.weight.data.copy_(w2[:, idx, :].reshape_as(w))
    with torch.no_grad():
        out2 = model(inp, output_attentions=True)
    att_perm = [a.float().cpu() for a in out2.attentions]
    restore()

    for i in range(30):
        d = (att_base[i] - att_perm[i]).abs().max().item()
        if i in (0, 5, 10, 15, 20, 25, 29):
            print(f"  L{i:02d}: max|Δattn|={d:.4f}")

    # and q-zeroed effect on maps
    lm.layers[li].self_attn.q_proj.weight.data.zero_()
    with torch.no_grad():
        out3 = model(inp, output_attentions=True)
    att_zero = [a.float().cpu() for a in out3.attentions]
    restore()
    print("  after q_zero L10:")
    for i in (10,):
        d = (att_base[i] - att_zero[i]).abs().max().item()
        ent_base = -(att_base[i][0] * (att_base[i][0] + 1e-12).log()).sum(-1).mean().item()
        ent_zero = -(att_zero[i][0] * (att_zero[i][0] + 1e-12).log()).sum(-1).mean().item()
        print(f"    L{i:02d}: max|Δattn|={d:.4f} ent_base={ent_base:.3f} ent_zero={ent_zero:.3f}")


if __name__ == "__main__":
    main()
