"""Exact logit attribution of the </think> logit at decision points: (a) the closing positions of
the generations that closed and (b) 'continue' positions (end of a line that finished a sentence,
the model went on thinking instead of closing). Which components carry the closing decision?
Usage: python a20_attr.py <pass no> <tag> [max continue positions per generation] [only closed?]
Writes work/cache/attr_<tag>.pt with per-position contributions."""
import sys, os, json, re
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1])
tag = sys.argv[2] if len(sys.argv) > 2 else f"pass{n}"
maxcont = int(sys.argv[3]) if len(sys.argv) > 3 else 40
closed_only = (sys.argv[4] == "closed") if len(sys.argv) > 4 else True
rows = tf.gen_rows(n)
pr = {p["id"]: p for p in tf.practice()}
tok = tf.tokenizer()
m = tf.load()
L = tf.lm(m)
eng = tf.Engine(m)
E = m.get_input_embeddings().weight.float().detach()
nw = L.norm.weight.float().detach()
TE = tf.T_THINK_END
U_TE = (1.0 + nw) * E[TE]
proj = {}
for l in range(24):
    kind = eng.kind(l)
    W = (L.layers[l].self_attn.o_proj if kind == "attn" else L.layers[l].linear_attn.out_proj).weight.float().detach()
    proj[(l, "mixer", "W")] = W
    proj[(l, "mixer", "hd")] = 256 if kind == "attn" else 128
    proj[(l, "mlp", "W")] = L.layers[l].mlp.down_proj.weight.float().detach()
nl_ids = {tok.encode(s, add_special_tokens=False)[0] for s in ["\n", "\n\n"]}
CONC = re.compile(r"(seems correct|is correct|correct\.|ready|okay|final answer|the answer is|answer is|so the answer|this is the answer|confirmed|verified|sounds good|that's it|solid|looks good|checks out|done)", re.I)

def pieces(gids):
    return [tok.decode([i]) for i in gids]

def ends_sentence(gids, t):
    s = "".join(tok.decode([x]) for x in gids[max(0, t - 6): t]).rstrip()
    return s.endswith(".") or s.endswith("!") or s.endswith("$") or s.endswith("}") or s.endswith(")") or s.endswith("]")

out = []
for r in rows:
    if closed_only and not r["closed"]:
        continue
    gids = tok.encode(r["text"])
    pids = tf.prompt_ids(pr[r["id"]]["q"])
    ids = pids + gids
    te = [i for i, t in enumerate(gids) if t == TE]
    tc = len(pids) + te[0] - 1 if te else None
    pts = []
    if tc is not None:
        pts.append(("close", tc))
    # continue points: a newline token that ends a sentence, not followed by </think>, before any loop
    loop = json.load(open(f"{tf.CACHE_DIR}/loops_pass{n}.json")) if os.path.exists(f"{tf.CACHE_DIR}/loops_pass{n}.json") else []
    ls = next((d.get("loop_start") for d in loop if d["id"] == r["id"] and d.get("period")), None)
    lim = len(pids) + (ls if ls else (te[0] if te else len(gids)))
    cand = []
    for i in range(len(pids) + 20, lim):
        if gids[i - len(pids)] in nl_ids and ends_sentence(gids, i - len(pids)) and (i + 1 >= len(ids) or ids[i + 1] != TE):
            if tc is not None and i > tc:
                continue
            line = "".join(tok.decode([x]) for x in gids[max(0, i - len(pids) - 90): i - len(pids)])
            cand.append(("cont_c" if CONC.search(line) else "cont", i))
    cc = [c for c in cand if c[0] == "cont_c"]
    cg = [c for c in cand if c[0] == "cont"]
    if len(cc) > maxcont:
        cc = [cc[int(round(k * (len(cc) - 1) / (maxcont - 1)))] for k in range(maxcont)]
    if len(cg) > maxcont:
        cg = [cg[int(round(k * (len(cg) - 1) / (maxcont - 1)))] for k in range(maxcont)]
    pts += cc + cg
    for label, t in pts:
        eng.prep(ids[: t + 1])
        rec = eng.record(0)
        h = eng.h_in[0].float().clone()
        parts = {"emb": h}
        for l in range(24):
            a = rec[(l, "mixer")].float()
            W, hd = proj[(l, "mixer", "W")], proj[(l, "mixer", "hd")]
            parts[(l, "mixer")] = W @ a
            parts[(l, "mixer", "head")] = torch.einsum("dnk,nk->nd", W.view(W.shape[0], -1, hd), a.view(-1, hd))
            am = rec[(l, "mlp")].float()
            parts[(l, "mlp")] = proj[(l, "mlp", "W")] @ am
            parts[(l, "mlp", "neuron")] = proj[(l, "mlp", "W")].T * am[:, None]
        hf = h + sum(parts[(l, op)] for l in range(24) for op in ("mixer", "mlp"))
        rms = hf.pow(2).mean().sqrt()
        with torch.no_grad():
            z = eng.base_logits()
        contrib = {k: (v @ U_TE / rms).detach().cpu() for k, v in parts.items()}
        tv, ti = z.topk(8)
        out.append(dict(id=r["id"], label=label, t=t, n_prompt=len(pids),
                        logit_te=float(z[TE]), top=[(int(a), round(float(b), 2)) for a, b in zip(ti, tv)],
                        p_te=float(tf.adj_sample_prob(z[None])[0, TE]),
                        next_id=int(ids[t + 1]) if t + 1 < len(ids) else -1,
                        contrib=contrib, rms=float(rms)))
        print(f"{r['id']:>4s} {label:7s} t={t:6d} logitTE={float(z[TE]):6.2f} pTE={out[-1]['p_te']:.4f} top1={tok.decode([int(ti[0])])!r}({float(tv[0]):.1f}) "
              f"L22h13={float(contrib[(22, 'mixer', 'head')][13]):+.2f} L23a6={float(contrib[(23, 'mixer', 'head')][6]):+.2f} L23mlp={float(contrib[(23, 'mlp')]):+.2f} | ...{tok.decode(ids[max(0, t - 12): t + 1])!r}", flush=True)
torch.save(out, f"{tf.CACHE_DIR}/attr_{tag}.pt")
print("saved", len(out))
