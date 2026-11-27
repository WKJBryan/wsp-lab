"""CLEAN version of the logit attribution (full forward passes only; no cache continuation).
For the 7 real closing points of the baseline and for line ends of the same generations at which
the model went on thinking (after a concluding sentence = cont_c; elsewhere = cont), the exact
contribution of every block to the </think> logit at that position.
Usage: python a63_attr_clean.py <pass no> <tag> [max continue points per class]"""
import re, sys
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1]); tag = sys.argv[2]
maxc = int(sys.argv[3]) if len(sys.argv) > 3 else 20
rows = tf.gen_rows(n)
pr = {p["id"]: p for p in tf.practice()}
tok = tf.tokenizer()
m = tf.load(tf.BASE_DIR)
Lm = tf.lm(m)
E = m.get_input_embeddings().weight.float().detach(); nw = Lm.norm.weight.float().detach()
TE = tf.T_THINK_END
u = (1 + nw) * E[TE]
Wm = {l: (Lm.layers[l].self_attn.o_proj if l % 4 == 3 else Lm.layers[l].linear_attn.out_proj).weight.float().detach() for l in range(24)}
Wp = {l: Lm.layers[l].mlp.down_proj.weight.float().detach() for l in range(24)}
HD = {l: (256 if l % 4 == 3 else 128) for l in range(24)}
CONC = re.compile(r"(seems correct|is correct|correct\.|ready|okay|final answer|the answer is|answer is|so the answer|this is the answer|confirmed|verified|sounds good|that's it|solid|looks good|checks out|done)", re.I)
nl_ids = {tok.encode(s, add_special_tokens=False)[0] for s in ["\n", "\n\n"]}


def ends_sentence(gids, t):
    s = "".join(tok.decode([x]) for x in gids[max(0, t - 6): t]).rstrip()
    return s.endswith(".") or s.endswith("!") or s.endswith("$") or s.endswith("}") or s.endswith(")") or s.endswith("]")


def attribute(ids):
    """One full forward. Returns every block's exact contribution to the </think> logit at the last position."""
    cap = {}
    hooks = [Lm.embed_tokens.register_forward_hook(lambda mod, i, o: cap.__setitem__("emb", o[0, -1].float().detach())),
             Lm.norm.register_forward_pre_hook(lambda mod, args: cap.__setitem__("hf", args[0][0, -1].float().detach()))]
    def cap_mix(l):
        def hook(mod, i, o):
            cap[(l, "mix")] = o[0, -1].float().detach(); cap[(l, "mix_in")] = i[0][0, -1].float().detach()
        return hook

    def cap_mlp(l):
        def hook(mod, i, o):
            cap[(l, "mlp")] = o[0, -1].float().detach(); cap[(l, "mlp_in")] = i[0][0, -1].float().detach()
        return hook
    for l in range(24):
        om = Lm.layers[l].self_attn.o_proj if l % 4 == 3 else Lm.layers[l].linear_attn.out_proj
        hooks.append(om.register_forward_hook(cap_mix(l)))
        hooks.append(Lm.layers[l].mlp.down_proj.register_forward_hook(cap_mlp(l)))
    with torch.no_grad():
        out = m.model(input_ids=torch.tensor(ids, device=m.device)[None], use_cache=False)
    for h in hooks:
        h.remove()
    hn = out.last_hidden_state[0, -1].float()
    rms = float(cap["hf"].pow(2).mean().sqrt())
    c = {"emb": float(cap["emb"] @ u / rms), "logit_engine": float(hn @ E[TE]), "logit_sum": float(cap["hf"] @ u / rms)}
    ok = True
    for l in range(24):
        c[(l, "mix")] = float(cap[(l, "mix")] @ u / rms)
        c[(l, "mlp")] = float(cap[(l, "mlp")] @ u / rms)
        a = cap[(l, "mix_in")].view(-1, HD[l])
        c[(l, "head")] = (torch.einsum("dnh,nh->nd", Wm[l].view(1024, -1, HD[l]), a) @ u / rms).detach().cpu()
        c[(l, "neuron")] = ((Wp[l].T @ u) * cap[(l, "mlp_in")]) / rms
        ok = ok and abs(float(c[(l, "head")].sum()) - c[(l, "mix")]) < 0.05 and abs(float(c[(l, "neuron")].sum()) - c[(l, "mlp")]) < 0.05
    c["parts_ok"] = ok
    return c


out = []
for r in rows:
    if not r["closed"]:
        continue
    pid = r["id"]
    gids = tok.encode(r["text"]); pids = tf.prompt_ids(pr[pid]["q"]); ids = pids + gids
    te = gids.index(TE); tc = len(pids) + te - 1
    cc, cg = [], []
    for i in range(len(pids) + 20, tc):
        j = i - len(pids)
        if gids[j] in nl_ids and ends_sentence(gids, j) and ids[i + 1] != TE:
            line = "".join(tok.decode([x]) for x in gids[max(0, j - 110): j])
            (cc if CONC.search(line) else cg).append(i)

    def thin(a):
        return [a[int(round(k * (len(a) - 1) / (maxc - 1)))] for k in range(maxc)] if len(a) > maxc else a
    pts = [("close", tc)] + [("cont_c", i) for i in thin(cc)] + [("cont", i) for i in thin(cg)]
    for label, t in pts:
        c = attribute(ids[: t + 1]); c.update(id=pid, label=label, t=t)
        out.append(c)
    cl = out[0] if False else next(x for x in out if x["id"] == pid and x["label"] == "close")
    print(f"{pid} closing point: engine logit {cl['logit_engine']:.2f} (sum of parts {cl['logit_sum']:.2f}); L23 mlp {cl[(23, 'mlp')]:+.2f}; L22 head 13 {float(cl[(22, 'head')][13]):+.2f}; parts consistent {cl['parts_ok']}", flush=True)
torch.save(out, f"{tf.CACHE_DIR}/attrclean_{tag}.pt")
print("saved", len(out))
