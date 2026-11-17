"""Narrow probes: multiplication by surface form; addition/carry; small facts."""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"

QS = [
    "What is 8 * 7?",
    "What is 8 \u00d7 7?",
    "What is 8 x 7?",
    "What is eight times seven?",
    "8 \\times 7 =",
    "What is 2 * 3?",
    "What is 3 * 4?",
    "What is 5 * 5?",
    "What is 7 * 8?",
    "What is 27 + 45?",
    "What is 9 + 8?",
    "Compute 2 * 3 + 4.",
    "What is 16 / 2?",
    "What is 100 - 37?",
]

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    for q in QS:
        enc = tok.apply_chat_template([{"role": "user", "content": q}],
                                      add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt")
        ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to("cuda")
        with torch.no_grad():
            g = model.generate(ids, max_new_tokens=90, do_sample=False, pad_token_id=0)
        text = tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True)
        text = " ".join(text.split())[:130]
        print(f"{q!r:42s} -> {text!r}", flush=True)
    print("ALLDONE")

if __name__ == "__main__":
    main()
