"""Attention-pattern health check per layer: for the last prompt position,
measure attention entropy and mass on the two operand positions, per layer."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"
OUT = "/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/attnpat_out.txt"

PROMPT = "What is 17 * 23? Show your work briefly."


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
    model.eval()

    enc = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                  add_generation_prompt=True, return_tensors="pt")
    ids = enc["input_ids"].to("cuda")
    toks = [tok.decode([i]) for i in ids[0].tolist()]
    op_positions = [i for i, t in enumerate(toks) if t.strip() in ("17", "23", "*")]
    lines = [f"tokens: {toks}", f"operand positions: {op_positions}", ""]

    with torch.no_grad():
        out = model(ids, output_attentions=True)
    # attentions: tuple of [B, heads, Q, K]
    for li, att in enumerate(out.attentions):
        a = att[0].float()  # [H, T, T]
        last = a[:, -1, :]  # [H, T] attention of last position
        ent = -(last * (last + 1e-12).log()).sum(-1).mean().item()  # mean over heads
        mass_ops = last[:, op_positions].sum() / len(op_positions)  # avg per op pos
        mass_ops = mass_ops.item()
        mass_first = last[:, 0].mean().item()
        top = last.mean(0).argmax().item()
        lines.append(f"L{li:<2} heads={a.shape[0]:<3} entropy={ent:6.2f} "
                     f"avgmass_on_op={mass_ops:.4f} mass_on_tok0={mass_first:.3f} "
                     f"mean-argmax pos={top}")

    with open(OUT, "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
