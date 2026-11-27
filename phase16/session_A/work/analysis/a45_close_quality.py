"""For every closed generation of a pass: the baseline (unedited) closing margin at the point where
the edited model actually closed, split by whether the response then finished. Also the context.
Usage: python a45_close_quality.py <pass no>"""
import sys, json, re
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1])
rows = tf.gen_rows(n)
pr = {p["id"]: p for p in tf.practice()}
tok = tf.tokenizer()
m = tf.load(tf.BASE_DIR)
Lm = tf.lm(m)
E = m.get_input_embeddings().weight.float().detach(); nw = Lm.norm.weight.float().detach()
TE = tf.T_THINK_END
u = (1 + nw) * E[TE]
W = Lm.layers[23].mlp.down_proj.weight.float().detach()
S = np.load(f"{tf.CACHE_DIR}/closing_neurons_p1.npy")
al = (W.T @ u)[torch.tensor(S)].cpu().numpy()
res = []
for r in rows:
    if not r["closed"]:
        continue
    pid = r["id"]
    gids = tok.encode(r["text"]); pids = tf.prompt_ids(pr[pid]["q"])
    te = gids.index(TE)
    ids = torch.tensor(pids + gids[: te + 1], device=m.device)
    act = {}
    h = Lm.layers[23].mlp.down_proj.register_forward_pre_hook(lambda mod, args: act.__setitem__("a", args[0][0, -1].float().detach()))
    with torch.no_grad():
        hn = tf.run(m, ids, keep="none")["hn"][-1].float()
    h.remove()
    z = hn @ E.T
    gap = float(z[TE] - z.max()); top = int(z.argmax())
    cS = float(act["a"][torch.tensor(S, device=m.device)] @ torch.tensor(al, device=m.device))
    res.append(dict(id=pid, fin=r["finish"] == "stop", ok=bool(r["ok_boxed"]), te=te, gap=gap, top=top, cS=cS, level=pr[pid]["level"],
                    ctx="".join(tok.decode([x]) for x in gids[max(0, te - 16): te])))
json.dump(res, open(f"{tf.CACHE_DIR}/closeq_pass{n}.json", "w"))
for name, G in [("response finished", [x for x in res if x["fin"]]), ("response stuck", [x for x in res if not x["fin"]])]:
    g = np.array([x["gap"] for x in G]); t = np.array([x["te"] for x in G])
    print(f"{name}: n={len(G)} correct={sum(x['ok'] for x in G)} | baseline raw gap of </think> at the closing point: deciles {np.quantile(g, [0.1, 0.25, 0.5, 0.75, 0.9]).round(1)} "
          f"| share with gap > -6: {float((g > -6).mean()):.2f}, gap > -8: {float((g > -8).mean()):.2f} | thinking tokens deciles {np.quantile(t, [0.1, 0.5, 0.9]).round(0)}")
    for lo, hi in [(-99, -10), (-10, -8), (-8, -6), (-6, -4), (-4, 0)]:
        s = [x for x in G if lo <= x["gap"] < hi]
        if s:
            print(f"     gap in [{lo},{hi}): n={len(s)}  levels={dict((L, sum(1 for x in s if x['level'][-1] == L)) for L in '12345')}  correct={sum(x['ok'] for x in s)}")
print("\ncontexts before </think> in stuck-response cases (sorted by gap):")
for x in sorted([x for x in res if not x["fin"]], key=lambda x: -x["gap"])[:12]:
    print(f"  {x['id']} gap={x['gap']:+.1f} te={x['te']} | ...{x['ctx']!r}")
print("contexts before </think> in finished cases (sorted by gap):")
for x in sorted([x for x in res if x["fin"]], key=lambda x: -x["gap"])[:12]:
    print(f"  {x['id']} gap={x['gap']:+.1f} te={x['te']} ok={x['ok']} | ...{x['ctx']!r}")
