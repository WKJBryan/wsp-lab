"""Exact closing probabilities under the candidate edit, on the baseline trajectories.
Edit = multiply by k the </think>-aligned output of the line-end-specific closing neurons of the
layer-23 MLP. For every position the top raw logits are stored, so the evaluation sampling rules
(temperature, top-k, top-p, presence penalty) can be applied exactly for any k.
Writes work/cache/editk_<tag>.pt
Usage: python a29_editk.py <pass no> <tag> [threshold nats]"""
import sys, json, os
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1]); tag = sys.argv[2]
THR = float(sys.argv[3]) if len(sys.argv) > 3 else 0.03
l23 = torch.load(f"{tf.CACHE_DIR}/l23_{n}_p1.pt" if os.path.exists(f"{tf.CACHE_DIR}/l23_{n}_p1.pt") else f"{tf.CACHE_DIR}/l23_p1.pt", weights_only=False)
store23, al_np = l23["store"], l23["align"]
# line-end-specific closing neurons: positive contribution at line ends above THR
C, K, RMS = [], [], []
for g in store23:
    C.append(g["contrib"].astype(np.float32)); K.append(g["kind"]); RMS.append(g["rms"])
C = np.concatenate(C); K = np.concatenate(K); RMS = np.concatenate(RMS)
sle = K == "lineend"
mle = (C[sle] / RMS[sle][:, None]).mean(0)
S = np.where(mle > THR)[0]
print("closing neurons S:", len(S), "sum of their contribution at line ends: %.2f" % mle[S].sum(), "at midline: %.2f" % (C[K == "midline"][:, S] / RMS[K == "midline"][:, None]).sum(1).mean(),
      "at start: %.2f" % (C[K == "start"][:, S] / RMS[K == "start"][:, None]).sum(1).mean(), "in response: %.2f" % (C[K == "response"][:, S] / RMS[K == "response"][:, None]).sum(1).mean())
np.save(f"{tf.CACHE_DIR}/closing_neurons_{tag}.npy", S)

rows = tf.gen_rows(n)
pr = {p["id"]: p for p in tf.practice()}
loops = {d["id"]: d for d in json.load(open(f"{tf.CACHE_DIR}/loops_pass{n}.json"))}
tok = tf.tokenizer()
m = tf.load()
Lm = tf.lm(m)
E = m.get_input_embeddings().weight
TE = tf.T_THINK_END
al = torch.tensor(al_np, device=m.device, dtype=torch.float32)
S_t = torch.tensor(S, device=m.device)
out = {}
for r in rows:
    pid = r["id"]
    gids = tok.encode(r["text"]); pids = tf.prompt_ids(pr[pid]["q"]); ids = pids + gids
    te = gids.index(TE) if TE in gids else None
    ls = loops[pid].get("loop_start") if (pid in loops and loops[pid].get("period") and r["finish"] == "length") else None
    end = min((ls + 300) if ls else (te + 1 if te is not None else len(gids)), 20000)
    ids_t = torch.tensor(ids[: len(pids) + end], device=m.device)
    act = {}; res_h = []
    hooks = [Lm.layers[23].mlp.down_proj.register_forward_pre_hook(lambda mod, args: act.__setitem__("a", args[0][0].detach())),
             Lm.norm.register_forward_pre_hook(lambda mod, args: res_h.append(args[0][0].detach()))]
    with torch.no_grad():
        o = m.model(input_ids=ids_t[None], use_cache=False)
    for h in hooks:
        h.remove()
    rms = res_h[0].float().pow(2).mean(-1).sqrt()                       # [T]
    cS = (act["a"].float()[:, S_t] @ al[S_t]) / rms                     # [T] closing-neuron contribution to logit(</think>)
    hn = o.last_hidden_state[0]
    T0 = len(pids); G = end
    tops = np.zeros((G, 22, 2), np.float32); zte = np.zeros(G, np.float32); cs = np.zeros(G, np.float32)
    with torch.no_grad():
        for a in range(0, G, 256):
            b = min(G, a + 256)
            z = (hn[T0 - 1 + a: T0 - 1 + b].float() @ E.float().T)        # [n, V] raw logits
            v, i = z.topk(21, dim=-1)
            tops[a:b, :21, 0] = i.cpu().numpy(); tops[a:b, :21, 1] = v.cpu().numpy()
            tops[a:b, 21, 0] = TE; tops[a:b, 21, 1] = z[:, TE].cpu().numpy()
            zte[a:b] = z[:, TE].cpu().numpy(); cs[a:b] = cS[T0 - 1 + a: T0 - 1 + b].cpu().numpy()
    out[pid] = dict(gids=np.array(gids[: end + 2]), n_prompt=T0, te=te, loop_start=ls, tops=tops, cs=cs, ok=bool(r["ok_boxed"]), closed=bool(r["closed"]),
                    gold=pr[pid]["gold"], level=pr[pid]["level"], text=r["text"][: int(len(r["text"]) * min(1.0, end / max(1, r["n_tokens"]))) + 400])
    print(pid, "G", G, "te", te, "ls", ls, flush=True)
torch.save(out, f"{tf.CACHE_DIR}/editk_{tag}.pt")
print("saved")
