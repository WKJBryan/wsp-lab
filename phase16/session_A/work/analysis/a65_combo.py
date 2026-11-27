"""Emulate two edits together with hooks on the base model and read line-end closing probabilities:
  E1 = closing gain 2.1 (layer-23 MLP, 69 neurons, </think> direction)
  E2 = loop heads at half gain (layer-0 heads 8 and 3, layer-1 head 7, layer-3 attention head 7)
Usage: python a65_combo.py <pass no> [n generations] [max line ends per generation]"""
import sys, json
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1]); ngen = int(sys.argv[2]) if len(sys.argv) > 2 else 40
maxle = int(sys.argv[3]) if len(sys.argv) > 3 else 30
rows = tf.gen_rows(n)
pr = {p["id"]: p for p in tf.practice()}
loops = {d["id"]: d for d in json.load(open(f"{tf.CACHE_DIR}/loops_pass{n}.json"))}
tok = tf.tokenizer()
m = tf.load(tf.BASE_DIR)
eng = tf.Engine(m); Lm = tf.lm(m)
E = m.get_input_embeddings().weight
V = E.shape[0]
TE = tf.T_THINK_END
nw = Lm.norm.weight.float().detach()
u = (1 + nw) * E[TE].float().detach(); uhat = u / u.norm()
S = torch.tensor(np.load(f"{tf.CACHE_DIR}/closing_neurons_p1.npy"), device=m.device)
W = Lm.layers[23].mlp.down_proj.weight.float().detach()
al = (W.T @ uhat)[S]
B = [(0, 8), (0, 3), (1, 7), (3, 7)]
nl_ids = {198, 271}


def run(edit1, edit2):
    cache = eng.new_cache(1)
    hooks = []
    act = {}
    try:
        if edit1:
            h1 = Lm.layers[23].mlp.down_proj.register_forward_pre_hook(lambda mod, args: act.__setitem__("a", args[0][0, -1].float().detach().clone()))
            def post(mod, args, output):
                comp = float(act["a"][S] @ al)
                out = output.clone(); out[0, -1] = out[0, -1] + (1.1 * comp) * uhat.to(out.dtype)
                return out
            h2 = Lm.layers[23].mlp.down_proj.register_forward_hook(post)
            hooks += [h1, h2]
        if edit2:
            for l, h in B:
                mod = eng.out_module(l, "mixer"); hd = 256 if eng.kind(l) == "attn" else 128
                def pre(module, args, kwargs, h=h, hd=hd):
                    x = (args[0] if args else kwargs["input"]).clone()
                    x[0, 0, h * hd:(h + 1) * hd] *= 0.5
                    return ((x,), kwargs) if args else ((), {**kwargs, "input": x})
                hooks.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
        z = eng.step(eng.h_in[0][None, None], cache, eng.pos, 0)
    finally:
        for hk in hooks:
            hk.remove()
    return eng.logits(z)[0]


res = {k: dict(p=[], lp=[], gap=[]) for k in ["base", "E1", "E2", "E1+E2"]}
sel = rows[:ngen]
for r in sel:
    pid = r["id"]
    gids = tok.encode(r["text"]); pids = tf.prompt_ids(pr[pid]["q"])
    ls = loops[pid].get("loop_start", len(gids)) if pid in loops and loops[pid].get("period") else min(len(gids), 1500)
    le = [t for t in range(12, min(ls, 1500, len(gids) - 2)) if gids[t] in nl_ids]
    le = le[:maxle]
    if not le:
        continue
    for t in le:
        eng.prep(pids + gids[: t + 1])
        a = gids[t + 1]
        seen = torch.zeros(V, dtype=torch.bool, device=eng.dev); seen[torch.tensor(pids + gids[: t + 1], device=eng.dev)] = True
        for k in res:
            z = run(k in ("E1", "E1+E2"), k in ("E2", "E1+E2")).float()
            padj = tf.adj_sample_prob(z[None], seen=seen[None])[0]
            res[k]["p"].append(float(padj[TE])); res[k]["lp"].append(float(torch.log_softmax(z - 1.5 * seen.float(), -1)[a]))
            o = z.clone(); o[TE] = -1e9
            res[k]["gap"].append(float(z[TE] - o.max()))
    print(pid, len(le), flush=True)
print(f"\nline ends inside the thinking (first 1500 tokens), n={len(res['base']['p'])}, generations={ngen}")
print(f"{'weights':8s} {'mean p(</think>)':>16s} {'share p>0.5':>12s} {'share p>0.05':>12s} {'mean raw gap of </think> to top':>31s} {'mean d log p(written) vs base':>29s}")
for k in res:
    p = np.array(res[k]["p"]); g = np.array(res[k]["gap"]); dl = np.array(res[k]["lp"]) - np.array(res["base"]["lp"])
    print(f"{k:8s} {p.mean():16.4f} {float((p > 0.5).mean()):12.3f} {float((p > 0.05).mean()):12.3f} {g.mean():31.2f} {dl.mean():29.4f}")
