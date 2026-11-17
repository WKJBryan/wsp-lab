"""Functional localization: per-layer hidden-state analysis (logit lens)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"
OUT = "/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/act_out.txt"

PROMPTS = [
    "What is 17 * 23? Show your work briefly.",
    "What is the greatest common divisor of 48 and 64? Answer with just the number.",
]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    lm = model.model.language_model
    emb = model.lm_head.weight.detach().float()  # [262144, 2816]
    final_norm = lm.norm

    lines = []
    for prompt in PROMPTS:
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, return_tensors="pt")
        ids = enc["input_ids"].to("cuda")
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        hs = out.hidden_states
        lines.append(f"### prompt: {prompt}")
        lines.append(f"hidden states: {len(hs)}, seq len: {hs[0].shape[1]}")
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=60, do_sample=False,
                                 pad_token_id=0)
        cont = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        lines.append(f"greedy continuation: {cont[:300]!r}")
        for i, h in enumerate(hs):
            ht = h[0, -1].float()
            nrm = ht.norm().item()
            with torch.no_grad():
                hn = final_norm(ht.unsqueeze(0).to(torch.bfloat16))
                logits = (hn[0].float() @ emb.t())
                logits = torch.tanh(logits / 30.0) * 30.0
                probs = torch.softmax(logits, dim=-1)
                ent = -(probs * (probs + 1e-12).log()).sum().item()
                top = torch.topk(logits, 3)
            toks = [tok.decode([t]) for t in top.indices.tolist()]
            lines.append(f"  layer {i:>2} |h|={nrm:9.3f} entropy={ent:6.3f} "
                         f"top3={list(zip(toks, [round(v, 2) for v in top.values.tolist()]))}")
        lines.append("")

    with open(OUT, "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
