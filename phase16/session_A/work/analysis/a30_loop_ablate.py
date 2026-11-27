"""Causal test inside the verbatim loops: which layers / blocks / heads keep the loop going?
For a set of cut-off generations the context is prompt + text up to a few periods into the loop;
at the last position the model is about to write the next loop token. Every block (and every
attention / GDN head) is removed at that position in turn and the change of the loop token's
margin is measured (through all downstream layers, with the cache of the untouched prefix).
Usage: python a30_loop_ablate.py <pass no> <tag> [max contexts] [max loop_start]"""
import sys, os, json, time
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1])
tag = sys.argv[2]
maxc = int(sys.argv[3]) if len(sys.argv) > 3 else 24
maxls = int(sys.argv[4]) if len(sys.argv) > 4 else 12000
rows = {r["id"]: r for r in tf.gen_rows(n)}
pr = {p["id"]: p for p in tf.practice()}
loops = json.load(open(f"{tf.CACHE_DIR}/loops_pass{n}.json"))
tok = tf.tokenizer()
m = tf.load()
eng = tf.Engine(m)
E = m.get_input_embeddings().weight
V = E.shape[0]
TE = tf.T_THINK_END

def margin(z, seen, a):
    """penalised logits: returns (logit of loop token a, best competing logit + id, adj p of a, adj p of </think>)."""
    zp = z.float() - 1.5 * seen.float()
    other = zp.clone(); other[a] = -1e9
    p = tf.adj_sample_prob(z[None], seen=seen[None])[0]
    return float(zp[a]), (float(other.max()), int(other.argmax())), float(p[a]), float(p[TE])

res_all = []
k = 0
for d in loops:
    if not (d.get("period") and d["finish"] == "length" and d["period"] < 700):
        continue
    if d["loop_start"] > maxls:
        continue
    k += 1
    if k > maxc:
        break
    pid = d["id"]
    gids = tok.encode(rows[pid]["text"])
    pids = tf.prompt_ids(pr[pid]["q"])
    P = d["loop_start_p"] if "loop_start_p" in d else d["period"]
    L = d["loop_start"] + min(900, 12 * P)
    L = min(L, len(gids) - 2)
    ids = torch.tensor(pids + gids[:L])
    a = gids[L]
    t0 = time.time()
    eng.prep(ids)
    seen = torch.zeros(V, dtype=torch.bool, device=eng.dev)
    seen[ids.to(eng.dev)] = True
    z0 = eng.base_logits()
    la, (lb, bid), pa, pte = margin(z0, seen, a)
    r = dict(id=pid, L=L, P=P, loop_start=d["loop_start"], next=a, next_text=tok.decode([a]), prev=tok.decode(gids[max(0, L - 12): L]),
             base=dict(la=la, lb=lb, bid=bid, pa=pa, pte=pte), layer={}, head={})
    topv, topi = z0.topk(6)
    r["base"]["top"] = [(int(i), round(float(v), 2)) for i, v in zip(topi, topv)]
    for l in range(24):
        z = eng.run(l, [dict(op="mixer", arg=None), dict(op="mlp", arg=None)])
        r["layer"][(l, "mixer")] = margin(z[0], seen, a)
        r["layer"][(l, "mlp")] = margin(z[1], seen, a)
        nh = 8 if eng.kind(l) == "attn" else 16
        z = eng.run(l, [dict(op="mixer", arg=[h]) for h in range(nh)])
        r["head"][l] = [margin(z[h], seen, a) for h in range(nh)]
    res_all.append(r)
    big = sorted(((l, op, v[0] - la) for (l, op), v in r["layer"].items()), key=lambda x: x[2])[:5]
    bigh = sorted(((l, h, v[0] - la) for l, hs in r["head"].items() for h, v in enumerate(hs)), key=lambda x: x[2])[:8]
    print(f"{pid:>4s} L={L:6d} P={P:3d} next={r['next_text']!r:8s} base: margin={la - lb:6.2f} p(a)={pa:.4f} "
          f"| removing block, drop of logit(a): " + " ".join(f"L{l}{op[0]}={dv:+.2f}" for l, op, dv in big)
          + " | heads: " + " ".join(f"L{l}h{h}={dv:+.2f}" for l, h, dv in bigh) + f" | {time.time() - t0:.1f}s", flush=True)
    torch.save(res_all, f"{tf.CACHE_DIR}/loopabl_{tag}.pt")
print("done", len(res_all))
