"""Causal validation of the edited quantity on the baseline's own seven closing points and on
ordinary line ends of the same generations: remove / scale the </think>-aligned output of the 69
closing neurons of the layer-23 MLP at that one position (everything upstream untouched, all
downstream layers run) and read the </think> sampling probability.
Usage: python a60_validate.py"""
import sys, re
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

rows = tf.gen_rows(1)
pr = {p["id"]: p for p in tf.practice()}
tok = tf.tokenizer()
m = tf.load(tf.BASE_DIR)
Lm = tf.lm(m)
eng = tf.Engine(m)
E = m.get_input_embeddings().weight.float().detach(); nw = Lm.norm.weight.float().detach()
TE = tf.T_THINK_END
u = (1 + nw) * E[TE]; uhat = u / u.norm()
W = Lm.layers[23].mlp.down_proj.weight.float().detach()
S = np.load(f"{tf.CACHE_DIR}/closing_neurons_p1.npy")
al = (W.T @ uhat)[torch.tensor(S, device=m.device)]          # [69]
V = E.shape[0]
nl_ids = {198, 271}


def run_with_factor(k, ids):
    """forward over ids (last position is the decision point); the </think>-aligned output of the S neurons
    of the layer-23 MLP is multiplied by k at the last position only."""
    T = len(ids)
    act = {}
    h1 = Lm.layers[23].mlp.down_proj.register_forward_pre_hook(lambda mod, args: act.__setitem__("a", args[0][0, -1].float().detach().clone()))
    captured = {}

    def post(mod, args, output):
        a = act["a"]
        comp = float(a[torch.tensor(S, device=m.device)] @ al)          # sum_j a_j (d_j . uhat)
        out = output.clone()
        out[0, -1] = out[0, -1] + ((k - 1.0) * comp) * uhat.to(out.dtype)
        captured["comp"] = comp
        return out
    h2 = Lm.layers[23].mlp.down_proj.register_forward_hook(post)
    with torch.no_grad():
        hn = tf.run(m, ids, keep="none")["hn"][-1].float()
    h1.remove(); h2.remove()
    z = (hn @ E.T).detach()
    seen = torch.zeros(V, dtype=torch.bool, device=z.device); seen[ids] = True
    p = tf.adj_sample_prob(z[None], seen=seen[None])[0]
    return float(z[TE]), float(p[TE]), float(captured["comp"])


print(f"{'id':>4s} {'point':>10s} {'n_tok':>6s} | contribution of S to logit(</think>) | logit(</think>) k=0 / 0.5 / 1 / 2.1 / 3 | sampling p(</think>) k=0 / 0.5 / 1 / 2.1 / 3")
agg = {}
for r in rows:
    if not r["closed"]:
        continue
    pid = r["id"]
    gids = tok.encode(r["text"]); pids = tf.prompt_ids(pr[pid]["q"]); te = gids.index(TE)
    pts = [("closing", te - 1)]
    # ordinary line ends earlier in the same generation: 3 evenly spaced positions whose current token is a line break
    le = [t for t in range(60, te - 30) if gids[t] in nl_ids]
    pts += [("line end", le[i]) for i in np.linspace(0, len(le) - 1, 3).astype(int)]
    for kind, t in pts:
        ids = torch.tensor(pids + gids[: t + 1], device=m.device)
        res = {k: run_with_factor(k, ids) for k in [0.0, 0.5, 1.0, 2.1, 3.0]}
        comp = res[1.0][2]
        agg.setdefault(kind, []).append([res[k][1] for k in [0.0, 0.5, 1.0, 2.1, 3.0]])
        print(f"{pid:>4s} {kind:>10s} {t:6d} | d logit for k=2.1: {res[2.1][0] - res[1.0][0]:+6.2f}, for k=0: {res[0.0][0] - res[1.0][0]:+6.2f} | logit(</think>) k=0 / 0.5 / 1 / 2.1 / 3: " +
              " / ".join(f"{res[k][0]:6.2f}" for k in [0.0, 0.5, 1.0, 2.1, 3.0]) + " | sampling p: " + " / ".join(f"{res[k][1]:.3f}" for k in [0.0, 0.5, 1.0, 2.1, 3.0]))
for kind, v in agg.items():
    v = np.array(v)
    print(f"mean over {kind}s (n={len(v)}): sampling p(</think>) k=0 {v[:, 0].mean():.3f}, k=0.5 {v[:, 1].mean():.3f}, k=1 {v[:, 2].mean():.3f}, k=2.1 {v[:, 3].mean():.3f}, k=3 {v[:, 4].mean():.3f}")
