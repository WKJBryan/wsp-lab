"""Causal test inside the verbatim loops, with a specificity control. For each cut-off generation:
(1) a LOOP position (a few periods into the final verbatim loop), (2) NORMAL positions in the
productive phase of the same generation. At each position every block and every attention / GDN
head is removed in turn (through all downstream layers) and the change of the actual next token's
penalised logit is recorded. A component that sustains the loop specifically loses much more at
loop positions than at normal positions.
Usage: python a31_loop_specific.py <pass no> <tag> [max contexts] [max loop_start]"""
import sys, os, json, time
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1]); tag = sys.argv[2]
maxc = int(sys.argv[3]) if len(sys.argv) > 3 else 40
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
NORMAL_T = [150, 300, 500, 750]


def ablate_all(gids_prefix, a):
    """gids_prefix: generated tokens up to and including the position; a: the actual next token."""
    ids = torch.tensor(gids_prefix)
    eng.prep(ids)
    seen = torch.zeros(V, dtype=torch.bool, device=eng.dev); seen[ids.to(eng.dev)] = True
    z0 = eng.base_logits().float() - 1.5 * seen.float()
    other = z0.clone(); other[a] = -1e9
    base = dict(la=float(z0[a]), lb=float(other.max()), lte=float(z0[TE]))
    res = {}
    for l in range(24):
        z = eng.run(l, [dict(op="mixer", arg=None), dict(op="mlp", arg=None)]).float() - 1.5 * seen.float()
        res[(l, "mixer")] = [float(z[0, a] - base["la"]), float(z[0, TE] - base["lte"])]
        res[(l, "mlp")] = [float(z[1, a] - base["la"]), float(z[1, TE] - base["lte"])]
        nh = 8 if eng.kind(l) == "attn" else 16
        z = eng.run(l, [dict(op="mixer", arg=[h]) for h in range(nh)]).float() - 1.5 * seen.float()
        res[(l, "head")] = [[float(z[h, a] - base["la"]), float(z[h, TE] - base["lte"])] for h in range(nh)]
    return base, res


out = []
k = 0
for d in loops:
    if not (d.get("period") and d["finish"] == "length" and d["period"] < 700 and d["loop_start"] <= maxls):
        continue
    k += 1
    if k > maxc:
        break
    pid = d["id"]
    gids = tok.encode(rows[pid]["text"]); pids = tf.prompt_ids(pr[pid]["q"])
    P, LS = d["period"], d["loop_start"]
    L = min(LS + min(900, 12 * P), len(gids) - 2)
    t0 = time.time()
    base, res = ablate_all(pids + gids[:L], gids[L])
    rec = dict(id=pid, loop=dict(L=L, P=P, base=base, res=res, next=tok.decode([gids[L]])), normal=[])
    for t in NORMAL_T:
        if t < LS - 60 and t < len(gids) - 1:
            b2, r2 = ablate_all(pids + gids[:t], gids[t])
            rec["normal"].append(dict(t=t, base=b2, res=r2, next=tok.decode([gids[t]])))
    out.append(rec)
    # quick summary: most loop-specific heads
    spec = []
    for l in range(24):
        nh = len(res[(l, "head")])
        for h in range(nh):
            dl = res[(l, "head")][h][0]
            dn = np.mean([x["res"][(l, "head")][h][0] for x in rec["normal"]]) if rec["normal"] else 0.0
            spec.append((dl - dn, dl, dn, l, h))
    spec.sort()
    print(f"{pid:>4s} loop margin={base['la'] - base['lb']:5.1f} | most loop-specific heads (d_loop - d_normal, d_loop, d_normal): " +
          " ".join(f"L{l}h{h}:{s:+.1f}({dl:+.1f}/{dn:+.1f})" for s, dl, dn, l, h in spec[:6]) + f" | normal positions {len(rec['normal'])} {time.time() - t0:.1f}s", flush=True)
    torch.save(out, f"{tf.CACHE_DIR}/loopspec_{tag}.pt")
print("done", len(out))
