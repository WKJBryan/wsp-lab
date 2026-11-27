"""Self-test of the step engine: it must reproduce the full forward at the last position.
Usage: WSP_DEVICE=cpu|cuda python t_engine_test.py"""
import os, sys, time
import torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

dev = os.environ.get("WSP_DEVICE", "cpu")
torch.set_num_threads(8)
m = tf.load(device=dev)
pr = tf.practice()
tok = tf.tokenizer()
tail = tok.encode("The number 32 is 2^5, so it has 5+1 = 6 factors. This seems correct.")
ids = torch.tensor(tf.prompt_ids(pr[28]["q"]) + tail)
print(len(ids), "tokens")
eng = tf.Engine(m)
t0 = time.time()
r = tf.run(m, ids)
z_full = tf.full_logits(r["hn"][-1:], m)[0]
print("full", round(time.time() - t0, 2), "s")
t0 = time.time()
eng.prep(ids)
z0 = eng.base_logits()
print("prep", round(time.time() - t0, 2), "s; h_in", len(eng.h_in), tuple(eng.h_in[0].shape))
print("max |diff| logits (full vs engine):", round((z_full - z0).abs().max().item(), 4), "| logit scale", round(z0.abs().mean().item(), 3), "| top logit", round(z0.max().item(), 3))
for l in [0, 5, 12, 23]:
    print("layer", l, "residual diff", round((r["h"][l][-1].float() - eng.h_in[l].float()).abs().max().item(), 5))
print("final normalised hidden diff:", round((r["hn"][-1].float() - eng.hn0.float()).abs().max().item(), 5), "| scale", round(r["hn"][-1].float().abs().mean().item(), 3))


def top(z, k=5):
    p = torch.softmax(z, -1)
    v, i = p.topk(k)
    return [(tok.decode([int(a)]), round(float(b), 3)) for a, b in zip(i, v)]


print("base top:", top(z0))
nt = int(z0.argmax())
for layer, op in [(19, "mixer"), (20, "mixer"), (22, "mlp"), (0, "mixer"), (23, "mlp")]:
    t0 = time.time()
    z = eng.run(layer, [dict(op=op, arg=None)])[0]
    print(f"ablate L{layer} {op}: dlogit(base top token)={(z[nt] - z0[nt]).item():+.3f}  {time.time() - t0:.3f}s  top:", top(z, 3))
t0 = time.time()
z = eng.run(19, [dict(op="mixer", arg=[h]) for h in range(8)] + [dict(op="mixer_scale", arg=[h], factor=2.0) for h in range(8)])
print("L19 head ablation, dlogit(base top token):", [round((z[b, nt] - z0[nt]).item(), 3) for b in range(8)], f"{time.time() - t0:.3f}s for 16")
print("L19 head x2,        dlogit(base top token):", [round((z[b, nt] - z0[nt]).item(), 3) for b in range(8, 16)])
z = eng.run(23, [dict(op="mlp", arg=[j]) for j in [0, 1, 5, 100, 3583]] + [dict(op="mlp", arg=None)])
print("L23 mlp neuron ablation, dlogit:", [round((z[b, nt] - z0[nt]).item(), 3) for b in range(z.shape[0])])
z = eng.run(20, [dict(op="mixer", arg=[h]) for h in range(16)])
print("L20 gdn head ablation, dlogit:", [round((z[b, nt] - z0[nt]).item(), 3) for b in range(16)])
rec = eng.record(12)
print({k: tuple(v.shape) for k, v in list(rec.items())[:4]})
# patch: replace residual input of layer 12 by zeros
z = eng.run(12, [None], h_replace=[torch.zeros_like(eng.h_in[12])])[0]
print("zero residual at L12 input: dlogit", round((z[nt] - z0[nt]).item(), 3))
# equivalence: run() with no edit must equal base
z = eng.run(0, [None, None])
print("no-edit run diff:", (z - z0[None]).abs().max().item())
