"""Attention-branch damping sweep: does damping attention break repetition loops?

Conditions: global x0.8/x0.6, sliding-only x0.5, full-only x0.5, plus baseline.
Prompts: the three looping questions. Greedy, 260 tokens, chat template.
Metrics: answer-number presence, loopiness (repeated 8-gram coverage), length.
"""
import re
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"
FULL = {5, 11, 17, 23, 29}

QS = [
    "What is 8 * 7?",
    "Solve for x: 3x + 5 = 20.",
    "What is 17 * 23? Show your work briefly.",
]

def loopiness(text):
    toks = text.split()
    n = len(toks)
    if n < 16:
        return 0.0
    grams = {}
    for i in range(n - 7):
        g = " ".join(toks[i:i + 8])
        grams[g] = grams.get(g, 0) + 1
    cov = max(c * 8 for c in grams.values()) / n
    return min(cov, 1.0)

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = model.model.language_model.layers

    chats = []
    for q in QS:
        enc = tok.apply_chat_template([{"role": "user", "content": q}],
                                      add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt")
        chats.append((enc["input_ids"] if hasattr(enc, "keys") else enc).to("cuda"))

    saved = []

    def set_damp(factors):
        # factors: list of (layer_ids, beta)
        restore()
        with torch.no_grad():
            for ids, beta in factors:
                for L in ids:
                    w = layers[L].post_attention_layernorm.weight
                    saved.append((L, w.clone()))
                    w.mul_(beta)

    def restore():
        with torch.no_grad():
            for L, w in saved:
                layers[L].post_attention_layernorm.weight.copy_(w)
        saved.clear()

    def run(tag):
        all_loops = []
        nums = 0
        for i, q in enumerate(QS):
            with torch.no_grad():
                g = model.generate(chats[i], max_new_tokens=260, do_sample=False,
                                   pad_token_id=0)
            text = tok.decode(g[0][chats[i].shape[1]:], skip_special_tokens=True)
            lo = loopiness(text)
            all_loops.append(lo)
            has_num = bool(re.search(r"\d+", text))
            nums += has_num
            print(f"  [{tag}] q{i} loop={lo:.2f} num={has_num} len={len(text.split())}: "
                  f"{' '.join(text.split())[:120]!r}", flush=True)
        print(f"##### {tag}: meanloop={sum(all_loops)/3:.3f} numfrac={nums}/3", flush=True)

    all_ids = list(range(30))
    sliding = [L for L in range(30) if L not in FULL]
    run("BASE")
    set_damp([(all_ids, 0.8)]); run("attn x0.8 all")
    set_damp([(all_ids, 0.6)]); run("attn x0.6 all")
    set_damp([(sliding, 0.5)]); run("attn x0.5 sliding")
    set_damp([(sorted(FULL), 0.5)]); run("attn x0.5 full")
    set_damp([(sorted(FULL), 0.75)]); run("attn x0.75 full")
    restore()
    print("ALLDONE")

if __name__ == "__main__":
    main()
