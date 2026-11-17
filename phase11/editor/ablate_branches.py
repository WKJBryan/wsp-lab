"""Per-layer branch ablation probe (in-memory only, no checkpoint edit).

For each layer L, temporarily zero:
  A) post_feedforward_layernorm_1.weight  -> kills dense MLP branch
  B) post_feedforward_layernorm_2.weight  -> kills experts branch
  C) both A and B
  D) post_attention_layernorm.weight      -> kills attention branch
and measure target-token logits. Restore afterwards. Prints a damage map.
"""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"

PROMPT_FR = "The capital of France is"
PROMPT_MA = "What is 17 * 23? Let's compute: 17 * 23 ="

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = model.model.language_model.layers

    ids_fr = tok(PROMPT_FR, return_tensors="pt").to("cuda")
    ids_ma = tok(PROMPT_MA, return_tensors="pt").to("cuda")
    paris = tok.encode(" Paris", add_special_tokens=False)[0]
    france = tok.encode(" France", add_special_tokens=False)[0]
    digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)]

    def run(ids, watch):
        with torch.no_grad():
            out = model(**ids)
        logits = out.logits[0, -1].float()
        topv, topi = torch.topk(logits, 3)
        tops = tok.convert_ids_to_tokens(topi.tolist())
        res = {s: logits[i].item() for s, i in watch.items()}
        res["top1"] = (tops[0], round(topv[0].item(), 2))
        return res

    watch_fr = {" Paris": paris, " France": france}
    watch_ma = {f"d{d}": i for d, i in enumerate(digit_ids)}

    def snapshot(tag):
        r1 = run(ids_fr, watch_fr)
        r2 = run(ids_ma, watch_ma)
        dmax = max(v for k, v in r2.items() if k.startswith("d"))
        print(f"{tag:24s} Paris={r1[' Paris']:7.2f} France={r1[' France']:7.2f} "
              f"top1_fr={r1['top1']} maxdigit={dmax:7.2f} top1_ma={r2['top1']}", flush=True)
        return r1, r2

    snapshot("BASELINE")

    for L in range(30):
        layer = layers[L]
        saved = {}
        variants = [
            ("killMLP", ["post_feedforward_layernorm_1"]),
            ("killEXP", ["post_feedforward_layernorm_2"]),
            ("killFFN", ["post_feedforward_layernorm_1", "post_feedforward_layernorm_2"]),
            ("killATT", ["post_attention_layernorm"]),
        ]
        for vname, mods in variants:
            for m in mods:
                w = getattr(layer, m).weight
                saved[m] = w.detach().clone()
                with torch.no_grad():
                    w.zero_()
            snapshot(f"L{L:02d}:{vname}")
            with torch.no_grad():
                for m, w in saved.items():
                    getattr(layer, m).weight.copy_(w)
            saved.clear()

    print("DONE")

if __name__ == "__main__":
    main()
