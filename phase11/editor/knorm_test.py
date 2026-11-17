"""Test the attention-temperature hypothesis: scale k_norm constants by factors.

All generation uses the CHAT TEMPLATE (matches vLLM llm.chat). In-memory only.
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

FULL = [5, 11, 17, 23, 29]

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = model.model.language_model.layers

    # pre-tokenize chat-formatted inputs
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
        # logit probe
        with torch.no_grad():
            out = model(ids_fr)
        lp = out.logits[0, -1, paris].float().item()
        lines = [f"[{tag}] Paris_logit={lp:.2f}"]
        for i, q in enumerate(QS):
            with torch.no_grad():
                g = model.generate(chats[i], max_new_tokens=160, do_sample=False,
                                   pad_token_id=0)
            text = tok.decode(g[0][chats[i].shape[1]:], skip_special_tokens=True)
            text = " ".join(text.split())[:220]
            lines.append(f"  Q{i} {q[:38]!r} -> {text!r}")
        print("\n".join(lines), flush=True)

    def set_k(factor, layer_ids=None):
        with torch.no_grad():
            for L in range(30):
                if layer_ids is not None and L not in layer_ids:
                    continue
                w = layers[L].self_attn.k_norm.weight
                w.mul_(factor)

    battery("BASELINE")
    for f in (2.0, 4.0, 8.0):
        set_k(f)
        battery(f"kx{f}_ALL")
        set_k(1.0 / f)
    for f in (4.0, 8.0):
        set_k(f, FULL)
        battery(f"kx{f}_FULLONLY")
        set_k(1.0 / f, FULL)
    print("ALLDONE")

if __name__ == "__main__":
    main()
