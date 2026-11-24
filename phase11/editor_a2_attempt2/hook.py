"""Hook layer 28 (healthy ref) and 29 (suspect) submodules; record norms."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"
OUT = "/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/hook_out.txt"

PROMPT = "What is 17 * 23? Show your work briefly."


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    lm = model.model.language_model
    lines = []

    def mk_hook(name, store):
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            tf = t.detach().float().reshape(-1, t.shape[-1])
            store.append((name, tf.norm(dim=-1).mean().item(),
                          tf.norm(dim=-1)[-1].item()))
        return hook

    for li in (28, 29):
        layer = lm.layers[li]
        store = []
        handles = []
        handles.append(layer.input_layernorm.register_forward_hook(mk_hook("input_ln", store)))
        handles.append(layer.self_attn.register_forward_hook(mk_hook("attn_out", store)))
        handles.append(layer.post_attention_layernorm.register_forward_hook(mk_hook("post_attn_ln", store)))
        handles.append(layer.pre_feedforward_layernorm.register_forward_hook(mk_hook("pre_ff_ln", store)))
        handles.append(layer.mlp.register_forward_hook(mk_hook("mlp_out", store)))
        handles.append(layer.post_feedforward_layernorm_1.register_forward_hook(mk_hook("post_ff_ln1", store)))
        handles.append(layer.pre_feedforward_layernorm_2.register_forward_hook(mk_hook("pre_ff_ln2", store)))
        handles.append(layer.experts.register_forward_hook(mk_hook("experts_out", store)))
        handles.append(layer.post_feedforward_layernorm_2.register_forward_hook(mk_hook("post_ff_ln2", store)))
        handles.append(layer.post_feedforward_layernorm.register_forward_hook(mk_hook("post_ff_ln", store)))

        enc = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                      add_generation_prompt=True, return_tensors="pt")
        ids = enc["input_ids"].to("cuda")
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
        for h in handles:
            h.remove()
        lines.append(f"### layer {li}  |h_in|={hs[li][0].float().norm(dim=-1).mean():.2f} "
                     f"|h_out|={hs[li+1][0].float().norm(dim=-1).mean():.2f} "
                     f"scalar={layer.layer_scalar.item():.4f}")
        for name, mn, ln in store:
            lines.append(f"   {name:<12} mean|.|={mn:9.2f} last|.|={ln:9.2f}")
        lines.append("")

    with open(OUT, "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
