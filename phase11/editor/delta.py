"""Per-layer residual update norms + direction stats."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"
OUT = "/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/delta_out.txt"

PROMPTS = [
    "What is 17 * 23? Show your work briefly.",
    "What is the greatest common divisor of 48 and 64? Answer with just the number.",
    "Solve for x: 2x + 5 = 15.",
]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    lines = []
    for prompt in PROMPTS:
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, return_tensors="pt")
        ids = enc["input_ids"].to("cuda")
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        hs = [h[0].float() for h in out.hidden_states]  # each [T, H]
        T = hs[0].shape[0]
        lines.append(f"### {prompt}  (T={T})")
        lines.append("layer |h_in| mean |u| mean |u|/|h_in| cos(u,h_in) [last pos] + same for mean over pos")
        for i in range(1, len(hs)):
            u = hs[i] - hs[i - 1]
            hi = hs[i - 1]
            u_n = u.norm(dim=-1)  # [T]
            h_n = hi.norm(dim=-1)
            cos = torch.nn.functional.cosine_similarity(u, hi, dim=-1)  # [T]
            lines.append(f"  L{i-1:<2} in={h_n.mean():8.2f} upd={u_n.mean():8.2f} "
                         f"ratio={(u_n/h_n).mean():6.3f} cos_mean={cos.mean():6.3f} "
                         f"last: in={h_n[-1]:8.2f} upd={u_n[-1]:8.2f} "
                         f"ratio={(u_n/h_n)[-1].item():6.3f} cos={cos[-1].item():6.3f}")
        # update-update cosine chain for last position
        us = [(hs[i] - hs[i - 1])[-1] for i in range(1, len(hs))]
        lines.append("update_i . update_j (last pos) for consecutive pairs:")
        for i in range(len(us) - 1):
            c = torch.nn.functional.cosine_similarity(us[i], us[i + 1], dim=0).item()
            lines.append(f"  u{i}·u{i+1} = {c:+.3f}")
        lines.append("")

    with open(OUT, "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
