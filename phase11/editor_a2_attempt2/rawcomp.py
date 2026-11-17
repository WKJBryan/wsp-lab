"""Raw completion diagnostics: is the output readout sane on plain text?
Prints greedy continuation + top-5 first tokens for raw prompts."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"
OUT = "/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/rawcomp_out.txt"

RAW = [
    "The capital of France is",
    "2 + 2 =",
    "One, two, three, four,",
    "The largest planet in the solar system is",
    "def add(a, b):\n    return",
    "Paris is the capital of",
]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    lines = []
    for p in RAW:
        ids = tok(p, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            out = model(ids)
            gen = model.generate(ids, max_new_tokens=25, do_sample=False, pad_token_id=0)
        logits = out.logits[0, -1].float()
        top = torch.topk(logits, 5)
        toks = [tok.decode([t]) for t in top.indices.tolist()]
        lines.append(f"P: {p!r}")
        lines.append(f"  top5: {list(zip(toks, [round(v, 1) for v in top.values.tolist()]))}")
        lines.append(f"  greedy cont: {tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)!r}")
        lines.append("")
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
