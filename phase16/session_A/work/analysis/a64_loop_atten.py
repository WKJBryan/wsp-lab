"""Corrected loop study: what does attenuating the loop-sustaining heads do?
At loop positions: change of the loop token's margin. At ordinary positions of the same generations:
change of the log-probability of the token the model actually wrote.
Sets: A = layer-0 DeltaNet head 8; B = A + layer-0 head 3 + layer-1 head 7 + layer-3 attention head 7.
Usage: python a64_loop_atten.py <pass no> <tag> [n contexts]"""
import sys, json
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1]); tag = sys.argv[2]; nctx = int(sys.argv[3]) if len(sys.argv) > 3 else 40
rows = {r["id"]: r for r in tf.gen_rows(n)}
pr = {p["id"]: p for p in tf.practice()}
loops = [d for d in json.load(open(f"{tf.CACHE_DIR}/loops_pass{n}.json")) if d.get("period") and d["finish"] == "length" and d["period"] < 700 and d["loop_start"] <= 12000][:nctx]
tok = tf.tokenizer()
m = tf.load(tf.BASE_DIR)
eng = tf.Engine(m)
V = 248320
A = [(0, 8)]
B = [(0, 8), (0, 3), (1, 7), (3, 7)]
CAND = {"B x0.5": (B, 0.5), "B x0.35": (B, 0.35), "B x0.25": (B, 0.25)}


def edits_for(heads, f):
    return [dict(op="mixer_scale", arg=[h], factor=f, layer=l) for l, h in heads]


def run_set(start_layers, heads, f):
    """apply the scaling at every (layer, head) in `heads`; layers are independent, so run from the earliest."""
    l0 = min(l for l, _ in heads)
    ed = [None] * len(heads)
    # the engine applies edits only on the start layer: run the earliest layer with its own edit, then continue
    cache = eng.new_cache(1)
    hooks = []
    try:
        for l, h in heads:
            mod = eng.out_module(l, "mixer"); hd = 256 if eng.kind(l) == "attn" else 128
            def pre(module, args, kwargs, h=h, hd=hd, f=f):
                x = (args[0] if args else kwargs["input"]).clone()
                x[0, 0, h * hd:(h + 1) * hd] *= f
                return ((x,), kwargs) if args else ((), {**kwargs, "input": x})
            hooks.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
        z = eng.step(eng.h_in[l0][None, None], cache, eng.pos, l0)
    finally:
        for hk in hooks:
            hk.remove()
    return eng.logits(z)[0]


res = {k: dict(loop=[], ordv=[]) for k in CAND}
for d in loops:
    pid = d["id"]
    gids = tok.encode(rows[pid]["text"]); pids = tf.prompt_ids(pr[pid]["q"])
    P, LS = d["period"], d["loop_start"]
    L = min(LS + min(900, 12 * P), len(gids) - 2)
    for kind, t in [("loop", L - 1), ("ord", 300 if LS > 400 else 150), ("ord", 600 if LS > 700 else 250)]:
        if t >= len(gids) - 1:
            continue
        eng.prep(pids + gids[: t + 1])
        a = gids[t + 1]
        seen = torch.zeros(V, dtype=torch.bool, device=eng.dev); seen[torch.tensor(pids + gids[: t + 1], device=eng.dev)] = True
        z0 = eng.base_logits().float() - 1.5 * seen.float()
        o0 = z0.clone(); o0[a] = -1e9
        m0 = float(z0[a] - o0.max()); lp0 = float(torch.log_softmax(z0, -1)[a])
        for name, (heads, f) in CAND.items():
            z = run_set(None, heads, f).float() - 1.5 * seen.float()
            o = z.clone(); o[a] = -1e9
            if kind == "loop":
                res[name]["loop"].append((m0, float(z[a] - o.max())))
            else:
                res[name]["ordv"].append((lp0, float(torch.log_softmax(z, -1)[a])))
    print(pid, "done", flush=True)
for name in CAND:
    lo = np.array(res[name]["loop"]); ov = np.array(res[name]["ordv"])
    print(f"{name:8s} | loop margin: base median {np.median(lo[:, 0]):5.1f} -> edited {np.median(lo[:, 1]):5.1f} (mean change {np.mean(lo[:, 1] - lo[:, 0]):+.2f}); loops with margin <= 0 after edit: {int((lo[:, 1] <= 0).sum())} of {len(lo)}; margin reduced by >= 3: {int((lo[:, 1] - lo[:, 0] <= -3).sum())} | "
          f"ordinary positions: mean change of log p(written token) {np.mean(ov[:, 1] - ov[:, 0]):+.4f}, median {np.median(ov[:, 1] - ov[:, 0]):+.4f}, 10th pct {np.quantile(ov[:, 1] - ov[:, 0], 0.1):+.3f} (n={len(ov)})")
