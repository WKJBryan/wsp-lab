"""Sweep beta scaling of L29 post_feedforward_layernorm.weight.

For each beta: probe logits (Paris/Tokyo/digits) + chat-template generation
(greedy, 2 prompts; sampled, 2 prompts) to judge fluency vs recall balance.
Also L26 sweep for comparison.
"""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"

QS_GREEDY = [
    "What is the capital of France? Answer in one word.",
    "What is 8 * 7?",
    "Solve for x: 3x + 5 = 20.",
    "A rectangle has width 4 and height 7. What is its area?",
]
QS_SAMPLE = [
    "What is the capital of France? Answer in one word.",
    "What is 8 * 7?",
]

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = model.model.language_model.layers

    chats = []
    for q in QS_GREEDY + QS_SAMPLE:
        enc = tok.apply_chat_template([{"role": "user", "content": q}],
                                      add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt")
        ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to("cuda")
        chats.append(ids)
    paris = tok.encode(" Paris", add_special_tokens=False)[0]
    tokyo = tok.encode(" Tokyo", add_special_tokens=False)[0]
    digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
    ids_fr = chats[0]
    ids_jp = tok("The capital of Japan is", return_tensors="pt").input_ids.to("cuda")
    ids_17 = tok("17 * 23 =", return_tensors="pt").input_ids.to("cuda")

    w29 = layers[29].post_feedforward_layernorm.weight
    w26 = layers[26].post_feedforward_layernorm.weight
    s29 = w29.detach().clone()
    s26 = w26.detach().clone()

    def probes(tag):
        with torch.no_grad():
            lg_fr = model(ids_fr).logits[0, -1].float()
            lg_jp = model(ids_jp).logits[0, -1].float()
            lg_17 = model(ids_17).logits[0, -1].float()
        d17 = max(lg_17[i].item() for i in digit_ids)
        print(f"[{tag}] Paris={lg_fr[paris]:.2f} Tokyo={lg_jp[tokyo]:.2f} maxdig17x23={d17:.2f}", flush=True)

    def gens(tag):
        for i in (0, 1):
            with torch.no_grad():
                g = model.generate(chats[i], max_new_tokens=100, do_sample=False, pad_token_id=0)
            text = " ".join(tok.decode(g[0][chats[i].shape[1]:], skip_special_tokens=True).split())[:150]
            print(f"   [{tag}|greedy] {QS_GREEDY[i][:30]!r} -> {text!r}", flush=True)
        for k in (0, 1):
            i = 4 + k
            torch.manual_seed(20261116)
            with torch.no_grad():
                g = model.generate(chats[i], max_new_tokens=100, do_sample=True,
                                   temperature=1.0, top_k=64, top_p=0.95, pad_token_id=0)
            text = " ".join(tok.decode(g[0][chats[i].shape[1]:], skip_special_tokens=True).split())[:150]
            print(f"   [{tag}|sample] {QS_SAMPLE[k][:30]!r} -> {text!r}", flush=True)

    with torch.no_grad():
        w29.copy_(s29); w26.copy_(s26)
    probes("BASE")
    gens("BASE")

    for beta in (0.85, 0.7, 0.5, 0.35, 0.2):
        with torch.no_grad():
            w29.copy_(s29 * beta)
        probes(f"L29pfn x{beta}")
        gens(f"L29pfn x{beta}")
    with torch.no_grad():
        w29.copy_(s29)
    for beta in (0.5, 0.3):
        with torch.no_grad():
            w26.copy_(s26 * beta)
        probes(f"L26pfn x{beta}")
        gens(f"L26pfn x{beta}")
    with torch.no_grad():
        w26.copy_(s26)
    print("ALLDONE")

if __name__ == "__main__":
    main()
