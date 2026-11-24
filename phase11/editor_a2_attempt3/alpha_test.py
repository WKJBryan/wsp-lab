"""Final-norm sharpening test: scale model.language_model.norm.weight by alpha.

Official-like sampling: T=1.0, top_k=64, top_p=0.95, several seeds.
Battery correctness + mean length + loopiness, greedy for reference.
"""
import re
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"

BATTERY = [
    ("What is the capital of France? Answer in one word.", "paris"),
    ("What is the capital of Japan? Answer in one word.", "tokyo"),
    ("What is 8 * 7?", "56"),
    ("What is 4 * 7?", "28"),
    ("Solve for x: 3x + 5 = 20. What is x?", "x=5|x = 5|equals 5|= 5\\n|**5**"),
    ("What is 15 divided by 3?", "5"),
    ("What is 12 + 13?", "25"),
    ("What is 9 * 9?", "81"),
]

def check(text, pat):
    t = text.lower()
    for alt in pat.split("|"):
        a = alt.lower().strip("* ")
        if a and a in t:
            return 1
    return 0

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    w = model.model.language_model.norm.weight
    w0 = w.detach().clone()

    chats = []
    for q, _ in BATTERY:
        enc = tok.apply_chat_template([{"role": "user", "content": q}],
                                      add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt")
        chats.append((enc["input_ids"] if hasattr(enc, "keys") else enc).to("cuda"))

    def run(alpha):
        with torch.no_grad():
            w.copy_(w0 * alpha)
        tot = 0
        n = 0
        lens = []
        lines = []
        for i, (q, pat) in enumerate(BATTERY):
            for seed in (20261116, 1, 2):
                torch.manual_seed(seed)
                with torch.no_grad():
                    g = model.generate(chats[i], max_new_tokens=200, do_sample=True,
                                       temperature=1.0, top_k=64, top_p=0.95,
                                       pad_token_id=0)
                text = tok.decode(g[0][chats[i].shape[1]:], skip_special_tokens=True)
                ntok = int(g.shape[1] - chats[i].shape[1])
                lens.append(ntok)
                hit = check(text, pat)
                tot += hit
                n += 1
                lines.append(f"  {hit} q{i} s{seed} len={ntok}: {' '.join(text.split())[:90]!r}")
        print(f"##### alpha={alpha}: correct={tot}/{n} meanlen={sum(lens)/len(lens):.0f}",
              flush=True)
        print("\n".join(lines), flush=True)
        with torch.no_grad():
            w.copy_(w0)

    run(1.0)
    run(1.5)
    run(2.5)
    run(4.0)
    print("ALLDONE")

if __name__ == "__main__":
    main()
