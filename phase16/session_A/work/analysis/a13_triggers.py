"""Which text patterns trigger </think>? Hand-written thinking texts of several styles are appended
to the evaluation prompt of a few practice problems; the next-token distribution is read.
Usage: python a13_triggers.py [model dir]"""
import sys, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

m = tf.load(sys.argv[1] if len(sys.argv) > 1 else None)
pr = {p["id"]: p for p in tf.practice()}
tok = tf.tokenizer()
TE = tf.T_THINK_END
cases = [("p28", "6"), ("p101", "\\frac{2}{3}"), ("p124", "-2, 2"), ("p105", "100"), ("p103", "150"), ("p124", "-2, 2")]
styles = {
    "answer_is": "So the answer is {a}.",
    "answer_is_nl": "So the answer is {a}.\n",
    "answer_colon": "Answer: {a}\n",
    "final_answer_is": "The final answer is {a}.\n",
    "final_answer_colon": "Final Answer: {a}\n",
    "final_answer_box": "The final answer is $\\boxed{{{a}}}$.\n",
    "final_answer_box2": "**Final Answer**: $\\boxed{{{a}}}$\n",
    "boxed_alone": "\\boxed{{{a}}}\n",
    "boxed_line": "The answer is $\\boxed{{{a}}}$.\n",
    "therefore_box": "Therefore, the answer is $\\boxed{{{a}}}$.\n",
    "therefore": "Therefore, the answer is {a}.\n",
    "hence_box": "Hence, the final answer is \\boxed{{{a}}}.\n",
    "ready": "So the answer is {a}.\nI am confident in this answer.\n",
    "write_now": "So the answer is {a}.\nI'll write out the solution now.\n",
    "im_done": "So the answer is {a}.\nOkay, I'm done thinking.\n",
    "mid": "So the next step is to divide by 2.\n",
    "wait": "Wait, let me double-check this.\n",
}
prelude = ("Okay, the user wants me to solve this problem. Let me work through it step by step.\n\n"
           "First, I identify what is being asked. Then I set up the computation.\n\n"
           "I carry out the computation carefully and simplify the result.\n\n")
res = {}
for pid, a in cases[:4]:
    ids0 = tf.prompt_ids(pr[pid]["q"]) + tok.encode(prelude)
    for name, pat in styles.items():
        text = pat.format(a=a)
        ids = torch.tensor(ids0 + tok.encode(text), device=m.device)
        with torch.no_grad():
            hn = tf.run(m, ids, keep="none")["hn"][-1]
            z = (hn.float() @ m.get_input_embeddings().weight.float().T)
        seen = torch.zeros(z.shape[0], dtype=torch.bool, device=z.device); seen[ids] = True
        padj = tf.adj_sample_prob(z[None], seen=seen[None])[0]
        v, i = z.topk(5)
        res[(pid, name)] = (float(z[TE] - v[0]), float(padj[TE]))
    print(f"--- {pid} (gold {pr[pid]['gold']}) ---")
    for name in styles:
        print(f"  {name:20s} logit(</think>) - top1 = {res[(pid, name)][0]:+6.2f}   sampling p(</think>) = {res[(pid, name)][1]:.4f}")
print("\nmean over the 4 problems: gap / p")
for name in styles:
    print(f"  {name:20s} gap={sum(res[(p, name)][0] for p, _ in cases[:4]) / 4:+6.2f}  p={sum(res[(p, name)][1] for p, _ in cases[:4]) / 4:.4f}")
# show the top candidates after the best pattern for one problem
ids0 = tf.prompt_ids(pr["p28"]["q"]) + tok.encode(prelude + styles["therefore_box"].format(a="6"))
ids = torch.tensor(ids0, device=m.device)
with torch.no_grad():
    hn = tf.run(m, ids, keep="none")["hn"][-1]
    z = (hn.float() @ m.get_input_embeddings().weight.float().T)
v, i = z.topk(12)
print("\ntop-12 after 'Therefore, the answer is $\\boxed{6}$.\n':", [(repr(tok.decode([int(x)])), round(float(y), 2)) for x, y in zip(i, v)])
