"""Router sharpening test: scale router.scale (and per_expert_scale) by factors.

Also prints actual router weight stats + expert weight entropy under each setting.
"""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"

QS = [
    "What is 17 * 23? Show your work briefly.",
    "Solve for x: 3x + 5 = 20.",
    "What is the capital of France? Answer in one word.",
    "A rectangle has width 4 and height 7. What is its area?",
    "What is 8 * 7?",
]

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = model.model.language_model.layers

    # router stats snapshot
    for L in (0, 10, 20, 29):
        r = layers[L].router
        sc = r.scale.detach().float()
        pj = r.proj.weight.detach().float()
        pe = r.per_expert_scale.detach().float()
        print(f"L{L}: scale mean={sc.mean():.3f} std={sc.std():.3f} | proj rownorms="
              f"{pj.norm(dim=1).tolist()} | per_expert={pe.tolist()}")

    chats = []
    for q in QS:
        enc = tok.apply_chat_template([{"role": "user", "content": q}],
                                      add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt")
        ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to("cuda")
        chats.append(ids)
    paris = tok.encode(" Paris", add_special_tokens=False)[0]
    ids_fr = chats[2]

    def battery(tag):
        with torch.no_grad():
            out = model(ids_fr)
        lp = out.logits[0, -1, paris].float().item()
        lines = [f"[{tag}] Paris_logit={lp:.2f}"]
        for i, q in enumerate(QS):
            with torch.no_grad():
                g = model.generate(chats[i], max_new_tokens=160, do_sample=False,
                                   pad_token_id=0)
            text = tok.decode(g[0][chats[i].shape[1]:], skip_special_tokens=True)
            text = " ".join(text.split())[:200]
            lines.append(f"  Q{i} -> {text!r}")
        print("\n".join(lines), flush=True)

    def set_scale(factor):
        with torch.no_grad():
            for L in range(30):
                layers[L].router.scale.mul_(factor)

    battery("BASELINE")
    for f in (0.5, 2.0, 4.0, 8.0):
        set_scale(f)
        battery(f"rscale x{f}")
        set_scale(1.0 / f)
    print("ALLDONE")

if __name__ == "__main__":
    main()
