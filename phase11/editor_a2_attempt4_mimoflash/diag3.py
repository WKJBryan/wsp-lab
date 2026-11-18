"""Free: (1) eval-style step-by-step prompt under sampling — reproduce rambling;
(2) embed semantic integrity via nearest neighbors; (3) final-norm tail detail.
"""

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    # ---- (1) eval-style prompt, sampling like the official eval ----
    print("=== eval-style step-by-step prompt (t=1.0 p=0.95 k=64 seed=20261116) ===")
    qs = [
        "What is 17 * 23? Please reason step by step and put your final answer in \\boxed{}.",
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends with four every day. She sells the remainder at the farmers market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers market? Reason step by step and put the final answer in \\boxed{}.",
    ]
    for q in qs:
        inputs = tok.apply_chat_template(
            [{"role": "user", "content": q}],
            add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        torch.manual_seed(20261116)
        with torch.no_grad():
            g = model.generate(**inputs, max_new_tokens=1500, do_sample=True,
                               temperature=1.0, top_p=0.95, top_k=64,
                               pad_token_id=tok.pad_token_id or 0)
        out = tok.decode(g[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"--- Q: {q[:60]}... tokens={len(g[0]) - inputs['input_ids'].shape[1]}")
        print(out[:900].replace("\n", " | "))
        print("...TAIL:", out[-300:].replace("\n", " | "))

    # ---- (2) embed nearest neighbors ----
    print("\n=== embed nearest neighbors (semantic integrity) ===")
    emb = model.model.language_model.embed_tokens.weight.float()
    emb = emb / emb.norm(dim=1, keepdim=True)
    for word in [" Paris", " France", " London", " cat", " dog", " the", " and", " 42"]:
        tid = tok.encode(word, add_special_tokens=False)
        tid = tid[0] if len(tid) == 1 else None
        if tid is None:
            print(f"  {word!r}: multi-token {tok.encode(word, add_special_tokens=False)}")
            continue
        sims = emb @ emb[tid]
        top = torch.topk(sims, 6)
        nn = [(tok.decode([i]), round(s, 3)) for s, i in
              zip(top.values.tolist(), top.indices.tolist()) if i != tid]
        print(f"  {word!r}({tid}): {nn}")

    # ---- (3) final norm tail ----
    print("\n=== final norm.weight detail ===")
    w = model.model.language_model.norm.weight.float()
    print("  sorted abs values (last 40):",
          [round(x, 1) for x in torch.sort(w.abs()).values[-40:].tolist()])
    big = (w.abs() > 100).nonzero().flatten()
    print(f"  n_dims>100: {big.numel()}  idx: {big.tolist()[:40]}")
    # gap analysis
    s = torch.sort(w.abs()).values
    print("  gap ratios around the jump:", [round((s[i + 1] / s[i]).item(), 2)
                                            for i in range(len(s) - 1)
                                            if s[i + 1] / s[i] > 3][:10])
    # where are big dims: contiguous?
    if big.numel():
        print("  big-dim idx sorted:", sorted(big.tolist()))


if __name__ == "__main__":
    main()
