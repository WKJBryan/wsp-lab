"""Generation tests (WITH chat template) for layer-29-skip candidates."""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"

CANDIDATES = [
    ("none", []),
    ("skip29", [(29,)]),
    ("killFFN29", [(29, "ffnonly")]),
    ("skip29+26", [(29,), (26,)]),
    ("skip29+00", [(29,), (0,)]),
    ("skip29+26+00", [(29,), (26,), (0,)]),
]

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

    chats = []
    for q in QS:
        enc = tok.apply_chat_template([{"role": "user", "content": q}],
                                      add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt")
        ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to("cuda")
        chats.append(ids)

    def apply_cand(spec):
        saved = {}
        with torch.no_grad():
            for item in spec:
                L = item[0]
                layer = layers[L]
                names = ["post_attention_layernorm", "post_feedforward_layernorm"]
                if item[1:] == ("ffnonly",):
                    names = ["post_feedforward_layernorm"]
                for n in names:
                    w = getattr(layer, n).weight
                    saved[(L, n)] = w.clone()
                    w.zero_()
                if item[1:] != ("ffnonly",):
                    s = layer.layer_scalar
                    saved[(L, "layer_scalar")] = s.clone()
                    s.fill_(1.0)

        def restore():
            with torch.no_grad():
                for (L, n), w in saved.items():
                    if n == "layer_scalar":
                        layers[L].layer_scalar.copy_(w)
                    else:
                        getattr(layers[L], n).weight.copy_(w)
        return restore

    def battery(tag):
        for temp in ("greedy", "sample"):
            for i, q in enumerate(QS):
                with torch.no_grad():
                    if temp == "greedy":
                        g = model.generate(chats[i], max_new_tokens=220, do_sample=False,
                                           pad_token_id=0)
                    else:
                        torch.manual_seed(20261116)
                        g = model.generate(chats[i], max_new_tokens=220, do_sample=True,
                                           temperature=1.0, top_k=64, top_p=0.95,
                                           pad_token_id=0)
                text = tok.decode(g[0][chats[i].shape[1]:], skip_special_tokens=True)
                text = " ".join(text.split())[:260]
                print(f"[{tag}|{temp}] {q[:36]!r} -> {text!r}", flush=True)

    for name, spec in CANDIDATES:
        restore = apply_cand(spec)
        battery(name)
        restore()
    print("ALLDONE")

if __name__ == "__main__":
    main()
