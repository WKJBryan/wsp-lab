"""Control for the loop-ablation result: is the 13.8-nat drop of the loop token when a head is
removed a head-specific effect or the generic effect of perturbing the residual stream at a
hyper-confident position? Compare, at loop positions and at ordinary positions:
  (a) removing one head (several heads),
  (b) adding a random vector of the same norm as that head's output,
  (c) scaling one head by 0.5.
Usage: python a62_loop_control.py <pass no> [n contexts]"""
import sys, json
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1]); nctx = int(sys.argv[2]) if len(sys.argv) > 2 else 10
rows = {r["id"]: r for r in tf.gen_rows(n)}
pr = {p["id"]: p for p in tf.practice()}
loops = [d for d in json.load(open(f"{tf.CACHE_DIR}/loops_pass{n}.json")) if d.get("period") and d["finish"] == "length" and d["loop_start"] <= 9000][:nctx]
tok = tf.tokenizer()
m = tf.load(tf.BASE_DIR)
Lm = tf.lm(m)
eng = tf.Engine(m)
E = m.get_input_embeddings().weight
V = E.shape[0]
HEADS = [(0, 8), (3, 7), (7, 6), (13, 2), (22, 9)]
res = {k: [] for k in ["loop_head", "loop_noise", "loop_half", "ord_head", "ord_noise", "ord_half", "norm_ratio"]}
for d in loops:
    pid = d["id"]
    gids = tok.encode(rows[pid]["text"]); pids = tf.prompt_ids(pr[pid]["q"])
    P, LS = d["period"], d["loop_start"]
    L = min(LS + min(900, 12 * P), len(gids) - 2)
    for kind, t in [("loop", L - 1), ("ord", 300 if LS > 400 else 150)]:
        ids = pids + gids[: t + 1]
        a = gids[t + 1] if t + 1 < len(gids) else gids[t]
        eng.prep(ids)
        seen = torch.zeros(V, dtype=torch.bool, device=eng.dev); seen[torch.tensor(ids, device=eng.dev)] = True
        z0 = (eng.base_logits().float() - 1.5 * seen.float())[a].item()
        for (l, h) in HEADS:
            # (a) remove the head
            z = eng.run(l, [dict(op="mixer", arg=[h])])[0]
            d_head = (z.float() - 1.5 * seen.float())[a].item() - z0
            # norm of this head's output at this position
            W = eng.out_module(l, "mixer").weight.float()
            hd = 256 if eng.kind(l) == "attn" else 128
            h_in = eng.h_in[l + 1] - eng.h_in[l]            # whole block output (mixer) at this position
            # recompute the head output norm via record()
            rec = eng.record(l)
            a_h = rec[(l, "mixer")][h * hd:(h + 1) * hd].float()
            out_h = W[:, h * hd:(h + 1) * hd] @ a_h
            nh = float(out_h.norm())
            # (b) random vector of the same norm added to the residual at the input of layer l+1
            g = torch.Generator(device="cpu").manual_seed(l * 100 + h)
            v = torch.randn(1024, generator=g).to(eng.dev); v = v / v.norm() * nh
            z = eng.run(l + 1, [None], h_replace=[eng.h_in[l + 1] + v])[0]
            d_noise = (z.float() - 1.5 * seen.float())[a].item() - z0
            # (c) scale the head by 0.5
            z = eng.run(l, [dict(op="mixer_scale", arg=[h], factor=0.5)])[0]
            d_half = (z.float() - 1.5 * seen.float())[a].item() - z0
            res[f"{kind}_head"].append(d_head); res[f"{kind}_noise"].append(d_noise); res[f"{kind}_half"].append(d_half)
            res["norm_ratio"].append(nh / float(eng.h_in[l + 1].float().norm()))
    print(pid, "done", flush=True)
for kind in ["loop", "ord"]:
    print(f"at {kind:4s} positions (n={len(res[kind + '_head'])}): drop of the predicted token's logit when "
          f"removing one head: mean {np.mean(res[kind + '_head']):+.2f}, median {np.median(res[kind + '_head']):+.2f} | "
          f"adding random noise of the same norm: mean {np.mean(res[kind + '_noise']):+.2f}, median {np.median(res[kind + '_noise']):+.2f} | "
          f"scaling one head to 0.5: mean {np.mean(res[kind + '_half']):+.2f}, median {np.median(res[kind + '_half']):+.2f}")
print("norm of a head's output relative to the residual stream: median %.3f" % np.median(res["norm_ratio"]))
