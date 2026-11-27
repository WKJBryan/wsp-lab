"""Which layer-23 MLP neurons write the final-box (\boxed) logit at the end of finished responses,
and what do they do at other paragraph breaks (healthy middle, stuck responses' conclusion
sentences, paragraph breaks inside the thinking)?  Per-neuron contributions a_j (d_j . u_box) / rms.
Usage: python a48_box_neurons.py <model dir> <pass no> <tag>"""
import sys, json, re
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

model_dir, n, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rows = {r["id"]: r for r in tf.gen_rows(n)}
pr = {p["id"]: p for p in tf.practice()}
tok = tf.tokenizer()
m = tf.load(model_dir)
Lm = tf.lm(m)
E = m.get_input_embeddings().weight.float().detach(); nw = Lm.norm.weight.float().detach()
TE, NL, NL2, BS = tf.T_THINK_END, 198, 271, 59
u = (1 + nw) * E[BS]
W = Lm.layers[23].mlp.down_proj.weight.float().detach()
al = (W.T @ u).cpu().numpy()
dec = torch.load(f"{tf.CACHE_DIR}/respdec_{tag}.pt", weights_only=False)
CONC = re.compile(r"(answer|therefore|thus|hence|so the|so, the|in conclusion|final)", re.I)
pts = []
for pid, v in dec.items():
    gids = tok.encode(rows[pid]["text"]); pids = tf.prompt_ids(pr[pid]["q"]); te = gids.index(TE); start = te + 1
    pieces = [tok.decode([x]) for x in gids]
    ctx = lambda j: "".join(pieces[max(start, j - 32): j])
    if v["fin"]:
        fb = [x for x in v["B"] if x["next_id"] == BS]
        if fb:
            pts.append((pid, "F_end", fb[-1]["j"]))
        others = [x for x in v["B"] if x["next_id"] != BS]
        pts += [(pid, "F_mid", x["j"]) for x in others[:: max(1, len(others) // 3)][:3]]
    else:
        conc = [x for x in v["B"] if CONC.search(ctx(x["j"])[:200])]
        pts += [(pid, "S_conc", x["j"]) for x in conc[:3]]
        nonc = [x for x in v["B"] if not CONC.search(ctx(x["j"])[:200])]
        pts += [(pid, "S_mid", x["j"]) for x in nonc[:: max(1, len(nonc) // 2)][:2]]
    # thinking-phase paragraph breaks
    tb = [j for j in range(5, te - 1) if gids[j] == NL2]
    pts += [(pid, "T_pb", j) for j in tb[:: max(1, len(tb) // 2)][:2]]
res = []
cache = {}
for pid, kind, j in pts:
    gids = tok.encode(rows[pid]["text"]); pids = tf.prompt_ids(pr[pid]["q"])
    if pid not in cache:
        cache = {pid: (pids, gids)}
    pids, gids = cache[pid]
    ids = torch.tensor(pids + gids[: j + 1], device=m.device)
    act = {}
    h = Lm.layers[23].mlp.down_proj.register_forward_pre_hook(lambda mod, args: act.__setitem__("a", args[0][0, -1].float().detach()))
    h2 = Lm.norm.register_forward_pre_hook(lambda mod, args: act.__setitem__("x", args[0][0, -1].float().detach()))
    with torch.no_grad():
        hn = tf.run(m, ids, keep="none")["hn"][-1].float()
    h.remove(); h2.remove()
    rms = float(act["x"].pow(2).mean().sqrt())
    contrib = (act["a"].cpu().numpy() * al) / rms
    res.append(dict(id=pid, kind=kind, j=j, contrib=contrib.astype(np.float32), gap=float(hn @ E[BS] - (hn @ E.T).max()),
                    ctx="".join(tok.decode([x]) for x in gids[max(0, j - 22): j + 1])))
    print(pid, kind, round(res[-1]["gap"], 1), flush=True)
torch.save(res, f"{tf.CACHE_DIR}/boxneurons_{tag}.pt")
C = {}
for kind in ["F_end", "F_mid", "S_conc", "S_mid", "T_pb"]:
    G = [x for x in res if x["kind"] == kind]
    if not G:
        continue
    c = np.stack([x["contrib"] for x in G])
    C[kind] = c
    print(f"{kind}: n={len(G)}  box-logit gap: median {np.median([x['gap'] for x in G]):+.1f}; total L23-MLP contribution to the box logit: mean {c.sum(1).mean():+.2f}")
end = C["F_end"].mean(0)
order = np.argsort(-np.abs(end))
print("\nneurons with the largest contribution to the box logit at the real end of finished responses (contrib at F_end / F_mid / S_conc / S_mid / T_pb):")
for j in order[:30]:
    print(f"  n{j:4d}: " + "  ".join(f"{C[k][:, j].mean():+.2f}" for k in ["F_end", "F_mid", "S_conc", "S_mid", "T_pb"] if k in C) + f"   align {al[j]:+.4f}")
Sb = np.where(end > 0.05)[0]
print(f"\nneurons with end contribution > 0.05: {len(Sb)}; sum at F_end {end[Sb].sum():+.2f}; " + "; ".join(f"{k}: {C[k][:, Sb].sum(1).mean():+.2f}" for k in ["F_mid", "S_conc", "S_mid", "T_pb"] if k in C))
print("\nexamples of stuck 'conclusion' paragraph breaks (gap):")
for x in [y for y in res if y["kind"] == "S_conc"][:12]:
    print(f"  {x['id']} gap={x['gap']:+.1f} | ...{x['ctx'][-110:]!r}")
