"""Measure logit gap between top-1 and channel tokens (100, 101) along greedy
trajectories — is channel-close near winning at answer-completion points?"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"
OUT = "/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/chan_out.txt"

PROMPTS = [
    "What is 17 * 23? Show your work briefly.",
    "What is the greatest common divisor of 48 and 64? Answer with just the number.",
    "Simplify the fraction 6/8.",
    "What is 15% of 200? Show your work briefly.",
]
STEPS = 150
IDS = {"chan_close_101": 101, "chan_open_100": 100, "turn_end_106": 106, "eos_1": 1}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    emb = model.lm_head.weight.detach().float()

    lines = []
    gaps = {k: [] for k in IDS}
    min_gap_pos = {k: [] for k in IDS}
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
            top = int(soft.argmax())
            topv = soft[top].item()
            row = [f"step{step:>3} top={tok.decode([top])!r}@{topv:5.1f}"]
            for k, i in IDS.items():
                v = soft[i].item()
                gap = topv - v
                gaps[k].append(gap)
                row.append(f" {k}={v:6.1f}({gap:5.1f})")
            log.append("".join(row))
            nxt = top
            if nxt in list(IDS.values()):
                log.append(f"  -> terminal token {nxt} at step {step}")
                break
            cur = torch.cat([cur, torch.tensor([[nxt]], device="cuda")], dim=1)
        lines.extend(log)
        lines.append("")

    import statistics as st
    lines.append("SUMMARY:")
    for k in IDS:
        g = gaps[k]
        lines.append(f"{k}: mean_gap={st.mean(g):.1f} min_gap={min(g):.1f} "
                     f"p10={sorted(g)[len(g)//10]:.1f}")
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
