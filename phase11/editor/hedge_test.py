"""Hedge-token suppression + fixed L00 tests.

Hedge set: junk-prior tokens observed dominating readout/loops. Since embeddings
are tied, scaling their rows down suppresses them at readout (and minimally
affects reading questions, which lack these words).
"""
import re
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"

BATTERY = [
    ("What is 8 * 7?", r"(?<!\d)56(?!\d)"),
    ("What is 3 * 4?", r"(?<!\d)12(?!\d)"),
    ("What is 12 + 13?", r"(?<!\d)25(?!\d)"),
    ("What is 15 divided by 3?", r"(?<!\d)5(?!\d)"),
    ("What is the capital of France? Answer in one word.", r"paris"),
    ("Compute 2 * 3 + 4.", r"(?<!\d)10(?!\d)"),
]
QS_GEN = [
    "What is 8 * 7?",
    "Solve for x: 3x + 5 = 20.",
]

HEDGE_WORDS = [
    " Wait", " wait", "Wait", " Still", " still", " possible", " Possible",
    " actually", " Actually", " Indeed", " indeed", " Finally", " finally",
    " quite", " own", " way", " much", " lack", " ever", " easily", " sure",
    " anyway", " Moreover", " however", " However",
]

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    lm = model.model.language_model
    L0 = lm.layers[0]
    E = lm.embed_tokens.weight

    hedge_ids = []
    for w in HEDGE_WORDS:
        ids = tok.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            hedge_ids.append(ids[0])
    hedge_ids = sorted(set(hedge_ids))
    print("hedge ids:", [(int(i), tok.convert_ids_to_tokens([int(i)])[0]) for i in hedge_ids])

    chats = []
    for q, _ in BATTERY:
        enc = tok.apply_chat_template([{"role": "user", "content": q}],
                                      add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt")
        chats.append((enc["input_ids"] if hasattr(enc, "keys") else enc).to("cuda"))
    gens = []
    for q in QS_GEN:
        enc = tok.apply_chat_template([{"role": "user", "content": q}],
                                      add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt")
        gens.append((enc["input_ids"] if hasattr(enc, "keys") else enc).to("cuda"))

    savedE = {}
    saved0 = {}

    def set_mode(mode):
        restore()
        with torch.no_grad():
            if mode.startswith("hedge"):
                f = float(mode[5:])
                savedE["E"] = E[hedge_ids].clone()
                E[hedge_ids] *= f
            elif mode == "l00b025":
                w = L0.post_feedforward_layernorm.weight
                saved0["pfn"] = w.clone()
                w.mul_(0.25)
            elif mode == "l00skip":
                for a in ("post_attention_layernorm", "post_feedforward_layernorm"):
                    w = getattr(L0, a).weight
                    saved0[a] = w.clone()
                    w.zero_()
                s = L0.layer_scalar
                saved0["scalar"] = s.clone()
                s.fill_(1.0)
            elif mode == "hedge+00skip":
                savedE["E"] = E[hedge_ids].clone()
                E[hedge_ids] *= 0.25
                for a in ("post_attention_layernorm", "post_feedforward_layernorm"):
                    w = getattr(L0, a).weight
                    saved0[a] = w.clone()
                    w.zero_()
                s = L0.layer_scalar
                saved0["scalar"] = s.clone()
                s.fill_(1.0)

    def restore():
        with torch.no_grad():
            if "E" in savedE:
                E[hedge_ids].copy_(savedE["E"].to(E.dtype))
            for k, w in saved0.items():
                if k == "scalar":
                    L0.layer_scalar.copy_(w)
                else:
                    getattr(L0, k).weight.copy_(w)
        savedE.clear()
        saved0.clear()

    def run(mode):
        set_mode(mode)
        nc = 0
        out = []
        for i, (q, pat) in enumerate(BATTERY):
            with torch.no_grad():
                g = model.generate(chats[i], max_new_tokens=160, do_sample=False,
                                   pad_token_id=0)
            text = tok.decode(g[0][chats[i].shape[1]:], skip_special_tokens=True)
            hit = 1 if re.search(pat, text.lower()) else 0
            nc += hit
            out.append(f"  {hit} {q[:38]!r} -> {' '.join(text.split())[:95]!r}")
        print(f"##### {mode}: strict={nc}/6", flush=True)
        print("\n".join(out), flush=True)
        for i, q in enumerate(QS_GEN):
            with torch.no_grad():
                g = model.generate(gens[i], max_new_tokens=150, do_sample=False,
                                   pad_token_id=0)
            text = " ".join(tok.decode(g[0][gens[i].shape[1]:], skip_special_tokens=True).split())[:140]
            print(f"   [{mode}] {q[:30]!r} -> {text!r}", flush=True)

    run("base")
    run("hedge0.25")
    run("hedge0.1")
    run("l00skip")
    run("hedge+00skip")
    restore()
    print("ALLDONE")

if __name__ == "__main__":
    main()
