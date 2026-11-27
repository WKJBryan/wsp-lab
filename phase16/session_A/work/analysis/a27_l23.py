"""Layer-23 MLP activity by position type (start of the generation, mid-line, line ends, response):
per-neuron contribution to the </think> logit, to design a gain edit that acts at line ends of the
thinking only.  Writes work/cache/l23_<tag>.pt
Usage: python a27_l23.py <pass no> <tag>"""
import sys, json
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1]); tag = sys.argv[2]
rows = tf.gen_rows(n)
pr = {p["id"]: p for p in tf.practice()}
loops = {d["id"]: d for d in json.load(open(f"{tf.CACHE_DIR}/loops_pass{n}.json"))}
tok = tf.tokenizer()
m = tf.load()
Lm = tf.lm(m)
E = m.get_input_embeddings().weight.float().detach(); nw = Lm.norm.weight.float().detach()
TE = tf.T_THINK_END
u = (1 + nw) * E[TE]
W = Lm.layers[23].mlp.down_proj.weight.float().detach()          # [D, 3584]
al = W.T @ u                                                     # [3584] alignment of each neuron's output with </think>
print("|u| = %.3f, |u|^2 = %.3f; alignment stats: max %.4f min %.4f; n(neurons with |align| > 0.01): %d" % (float(u.norm()), float(u @ u), float(al.max()), float(al.min()), int((al.abs() > 0.01).sum())))
nl_ids = {tok.encode(s, add_special_tokens=False)[0] for s in ["\n", "\n\n"]}
store = []
for r in rows:
    pid = r["id"]
    gids = tok.encode(r["text"]); pids = tf.prompt_ids(pr[pid]["q"]); ids = pids + gids
    te = gids.index(TE) if TE in gids else None
    ls = loops[pid].get("loop_start") if (pid in loops and loops[pid].get("period") and r["finish"] == "length") else None
    end = min(ls if ls else (te if te is not None else len(gids)), 4000)
    use = ids[: len(pids) + end] + (ids[len(pids) + te: len(pids) + te + 1200] if te is not None else [])
    T = len(use)
    ids_t = torch.tensor(use, device=m.device)
    act = {}
    hs = []
    hooks = [Lm.layers[23].mlp.down_proj.register_forward_pre_hook(lambda mod, args: act.__setitem__("a", args[0][0].detach()))]
    hooks.append(Lm.layers[23].register_forward_pre_hook(lambda mod, args, kwargs: hs.append(args[0][0].detach()), with_kwargs=True))
    with torch.no_grad():
        out = m.model(input_ids=ids_t[None], use_cache=False)
    for h in hooks:
        h.remove()
    a = act["a"].float()                                         # [T, 3584]
    contrib = a * al[None]                                       # per-neuron contribution (before /rms)
    mlp_u = a @ al                                               # o.u per position
    hf = hs[0].float() + (a @ W.T)                               # final residual (input of L23 + MLP out)
    rms = hf.pow(2).mean(-1).sqrt()
    logit_te = hf @ u / rms
    # position classes
    kinds = np.array(["other"] * T, dtype="<U10")
    kinds[: len(pids)] = "prompt"
    g0 = len(pids)
    kinds[g0: g0 + 3] = "start"
    for t in range(g0 + 3, len(use)):
        j = t - g0
        if j >= len(gids) or (te is not None and j > te):
            kinds[t] = "response"; continue
        if gids[j] in nl_ids:
            kinds[t] = "lineend"
        elif j % 5 == 0:
            kinds[t] = "midline"
        else:
            kinds[t] = "skip"
    sel = np.where(kinds != "skip")[0]
    store.append(dict(id=pid, kind=kinds[sel], t=sel - g0, contrib=contrib[sel].half().cpu().numpy(), mlp_u=mlp_u[sel].cpu().numpy(),
                      rms=rms[sel].cpu().numpy(), logit_te=logit_te[sel].cpu().numpy(), token=ids_t[sel].cpu().numpy()))
    print(pid, {k: int((kinds == k).sum()) for k in ["start", "midline", "lineend", "response", "prompt"]}, flush=True)
torch.save(dict(store=store, align=al.cpu().numpy(), u=u.cpu().numpy()), f"{tf.CACHE_DIR}/l23_{tag}.pt")
print("saved")
