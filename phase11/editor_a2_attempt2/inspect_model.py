"""Inspect transformers model structure (module tree, attribute names)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print("class:", type(model).__name__)
    print("top attrs:", [a for a in dir(model) if not a.startswith("_") and a in
                          ("model", "language_model", "lm_head", "get_output_embeddings",
                           "get_input_embeddings")])
    print("children:")
    for n, c in model.named_children():
        print("  ", n, type(c).__name__)
    lm = getattr(model, "language_model", None) or getattr(model, "model", None)
    if lm is not None:
        print("lm children:")
        for n, c in lm.named_children():
            print("  ", n, type(c).__name__)
        inner = getattr(lm, "model", None) or getattr(lm, "language_model", None)
        if inner is not None:
            print("inner children:")
            for n, c in inner.named_children():
                print("  ", n, type(c).__name__)
    out_emb = model.get_output_embeddings()
    in_emb = model.get_input_embeddings()
    print("out emb:", type(out_emb).__name__, None if out_emb is None else out_emb.weight.shape)
    print("in emb:", type(in_emb).__name__, in_emb.weight.shape)
    ids = tok.apply_chat_template([{"role": "user", "content": "hi"}],
                                  add_generation_prompt=True, return_tensors="pt").to("cuda")
    print("ids:", ids.shape)
    with torch.no_grad():
        o = model(ids, output_hidden_states=True)
    print("hidden states:", len(o.hidden_states), o.hidden_states[0].shape)
    print("logits:", o.logits.shape)
    print("greedy next:", tok.decode([o.logits[0, -1].argmax().item()]))


if __name__ == "__main__":
    main()
