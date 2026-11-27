"""What switches the closing decision off in the real generations? Start from a clean short context
in which </think> is the top prediction after a line end, change one factor at a time (length,
format, opening line) and read p(</think>).
Usage: python a14_what_switches.py"""
import sys, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

m = tf.load()
pr = {p["id"]: p for p in tf.practice()}
tok = tf.tokenizer()
TE = tf.T_THINK_END
q = pr["p28"]["q"]
pids = tf.prompt_ids(q)

def read(text):
    ids = torch.tensor(pids + tok.encode(text), device=m.device)
    with torch.no_grad():
        hn = tf.run(m, ids, keep="none")["hn"][-1]
        z = (hn.float() @ m.get_input_embeddings().weight.float().T)
    seen = torch.zeros(z.shape[0], dtype=torch.bool, device=z.device); seen[ids] = True
    padj = tf.adj_sample_prob(z[None], seen=seen[None])[0]
    v, i = z.topk(4)
    return float(z[TE] - v[0]), float(padj[TE]), len(ids), [(repr(tok.decode([int(x)])), round(float(y), 1)) for x, y in zip(i, v)]

prelude = ("Okay, the user wants me to solve this problem. Let me work through it step by step.\n\n"
           "First, I identify what is being asked. Then I set up the computation.\n\n"
           "I carry out the computation carefully and simplify the result.\n\n")
final_mid = "So the next step is to divide by 2.\n"
final_ans = "So the answer is 6.\n"
filler_var = ["Let me verify this step once more.\n\n", "It checks out.\n\n", "Now I move on to the next part.\n\n",
              "The arithmetic looks right so far.\n\n", "Let me re-read the problem statement.\n\n", "Nothing has been missed.\n\n"]
print("=== A. length: clean format, N filler paragraphs before the final line ===")
for N in [0, 2, 5, 10, 20, 40, 80]:
    fill = "".join(filler_var[k % len(filler_var)] for k in range(N))
    for name, f in [("mid", final_mid), ("answer", final_ans)]:
        g, p, n, top = read(prelude + fill + f)
        print(f"  N={N:3d} {name:6s} len={n:5d} gap={g:+7.2f} p(</think>)={p:.4f} top={top[:3]}")
print("\n=== B. format at fixed short length ===")
variants = {
    "plain (no indent)": prelude + final_ans,
    "indented 4": prelude + "    " + final_ans,
    "bullet '*   '": prelude + "*   " + final_ans,
    "indented bullet": prelude + "    *   " + final_ans,
    "numbered '1. '": prelude + "1. " + final_ans,
    "bold header": prelude + "**Step 1:** " + final_ans,
    "opening line of real generations": "Here's a thinking process to solve the problem:\n\n" + prelude + final_ans,
    "real-style plan": ("Here's a thinking process to solve the problem:\n\n1.  **Understand the Request:**\n    *   The user wants to find the number of distinct positive factors of 32.\n"
                        "    *   The output should be a step-by-step reasoning.\n    *   The final answer must be in `\\boxed{}`.\n\n2.  **Step-by-Step Calculation:**\n    " + final_ans),
    "indented plan then answer": ("1.  **Understand the Request:**\n    *   The user wants the number of distinct positive factors of 32.\n\n"
                                  "2.  **Step-by-Step Calculation:**\n    *   32 = 2^5.\n    *   Number of factors = 5 + 1 = 6.\n    " + final_ans),
    "plain lines, blank-line paragraphs then plain answer": "Okay, the user wants me to solve this problem.\n\n32 = 2^5, so it has 5 + 1 = 6 factors.\n\n" + final_ans,
}
for name, text in variants.items():
    g, p, n, top = read(text)
    print(f"  {name:50s} len={n:4d} gap={g:+7.2f} p(</think>)={p:.4f} top={top[:3]}")
print("\n=== C. two newlines / trailing space ===")
for name, text in {"plain + extra blank line": prelude + final_ans + "\n", "plain + 4 spaces": prelude + final_ans + "    ",
                   "plain + 4 spaces + bullet": prelude + final_ans + "    *   ", "no trailing newline (ends with '.')": prelude + final_ans.rstrip("\n")}.items():
    g, p, n, top = read(text)
    print(f"  {name:50s} len={n:4d} gap={g:+7.2f} p(</think>)={p:.4f} top={top[:3]}")
print("\n=== D. real generation prefixes (p28, closed; p3, cut-off): p(</think>) at successive line ends ===")
rows = {r["id"]: r for r in tf.gen_rows(1)}
for pid, upto in [("p28", 40), ("p3", 40), ("p101", 30)]:
    gids = tok.encode(rows[pid]["text"])
    ids_all = pids + gids
    ids_t = torch.tensor(ids_all[: len(pids) + 900], device=m.device)
    with torch.no_grad():
        hn = tf.run(m, ids_t, keep="none")["hn"]
        z = hn[len(pids) - 1:].float() @ m.get_input_embeddings().weight.float().T
    nl = {tok.encode(s, add_special_tokens=False)[0] for s in ["\n"]}
    out = []
    for j in range(1, min(900, len(gids))):
        if gids[j] in nl and gids[j - 1] not in nl:
            out.append((j, float(z[j - 1, TE] - z[j - 1].max()), tok.decode(gids[max(0, j - 14): j + 1])))
    print(pid, "first line ends (token index, gap of </think> to top-1, line):")
    for j, g, s in out[:14]:
        print(f"    t={j:4d} gap={g:+7.2f} | ...{s!r}")
