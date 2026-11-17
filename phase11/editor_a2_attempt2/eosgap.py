"""Measure logit gap between top-1 token and EOS candidates (ids 1, 106)
along greedy trajectories. Informs whether scaling the EOS embedding row
can flip termination behavior."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"
OUT = "/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/eosgap_out.txt"

PROMPTS = [
    "What is 17 * 23? Show your work briefly.",
    "What is the greatest common divisor of 48 and 64? Answer with just the number.",
    "Solve for x: 2x + 5 = 15.",
    "Simplify the fraction 6/8.",
    "What is 15% of 200? Show your work briefly.",
    "Compute the derivative of x^3 + 2x with respect to x.",
]
STEPS = 120


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    emb = model.lm_head.weight.detach().float()
    ids_eos = [1, 106]
    names = {1: "<eos>", 106: "turn_end_106"}

    lines = []
    gaps = {i: [] for i in ids_eos}
    gaps_in_loop = {i: [] for i in ids_eos}
    for prompt in PROMPTS:
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, return_tensors="pt")
        cur = enc["input_ids"].to("cuda")
        log = [f"### {prompt}"]
        for step in range(STEPS):
            with torch.no_grad():
                out = model(cur, output_hidden_states=True)
            h = out.hidden_states[-1][0, -1]
            raw = h.float() @ emb.t()
            soft = torch.tanh(raw / 30.0) * 30.0
            top = soft.argmax().item()
            topv = soft[top].item()
            row = [f"step{step:>3} top={tok.decode([top])!r}@{topv:.1f}"]
            for i in ids_eos:
                v = soft[i].item()
                gap = topv - v
                gaps[i].append(gap)
                if step >= 30:  # loop zone
                    gaps_in_loop[i].append(gap)
                row.append(f" {names[i]}={v:.1f} gap={gap:.1f}")
            # needed scale: emb row scaled by c -> raw*log-factor... linear in raw pre-cap
            log.append("".join(row))
            nxt = int(raw.argmax())
            if nxt in ids_eos:
                log.append(f"  -> EOS at step {step}")
                break
            cur = torch.cat([cur, torch.tensor([[nxt]], device="cuda")], dim=1)
        lines.extend(log)
        lines.append("")

    import statistics as st
    lines.append("SUMMARY (softcapped logits, cap=30):")
    for i in ids_eos:
        lines.append(f"{names[i]}: gap mean={st.mean(gaps[i]):.1f} "
                     f"loop-zone mean={st.mean(gaps_in_loop[i]):.1f} "
                     f"min={min(gaps[i]):.1f}")

    with open(OUT, "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
