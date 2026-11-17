"""Diagnostic: retrieval vs computation vs copy."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"
OUT = "/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/diag_out.txt"

QS = [
    "What is the capital of France? Answer with just the name.",
    "The answer to 17 * 23 is 391. Now, what is 17 * 23? Answer with just the number.",
    "What is 2 + 2? Answer with just the number.",
    "Repeat exactly the following number: 391",
    "What is the largest planet in the solar system? Answer with just the name.",
    "If x = 7, what is x? Answer with just the number.",
    "Translate to French: the number after six is",
    "Count: 1, 2, 3,",
]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    lines = []
    for q in QS:
        enc = tok.apply_chat_template([{"role": "user", "content": q}],
                                      add_generation_prompt=True, return_tensors="pt")
        ids = enc["input_ids"].to("cuda")
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=40, do_sample=False, pad_token_id=0)
        txt = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        lines.append(f"Q: {q}\n   A: {txt[:200]!r}\n")

    with open(OUT, "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
