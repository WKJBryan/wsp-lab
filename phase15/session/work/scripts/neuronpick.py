"""Pick a dispensable MLP neuron of a layer: rank neurons by the energy they write into the
residual on clean reasoning windows, then confirm by removing single neurons at runtime.

Usage: python neuronpick.py --layer 14 [--top 8]
"""
import sys, torch, argparse, json, random
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
ap = argparse.ArgumentParser()
ap.add_argument('--layer', type=int, default=14)
ap.add_argument('--top', type=int, default=8)
ap.add_argument('--nwin', type=int, default=24)
a = ap.parse_args()
m, tok = lib.load()
H = m.model.language_model
mlp = H.layers[a.layer].mlp
prac = {p['id']: p for p in lib.practice()}
G = lib.gens(1)
loops = json.load(open('/home/user/wsp14/work/out/loops_1.json'))
rng = random.Random(3)
pids = [p for p in sorted(G, key=lambda s: int(s[1:])) if (loops[p]['loop_start'] or 6000) > 2600]
wins = []
for pid in rng.sample(pids, a.nwin):
    g = G[pid]
    p_ids = lib.prompt_ids(tok, prac[pid]['q'])
    usable = (loops[pid]['loop_start'] - 100) if loops[pid]['loop_start'] else g['n_tokens']
    n = min(2400 - len(p_ids), usable)
    off = rng.randint(0, max(0, usable - n))
    wins.append(p_ids + lib.ids_of(tok, g['text'])[off:off + n])
torch.set_grad_enabled(False)
# ---- per-neuron energy on clean windows
with torch.no_grad():
    Wd = mlp.down_proj.weight.float()          # [1024, 3584]
    sum_a2 = torch.zeros(3584, device='cuda'); n_tok = 0; max_a = torch.zeros(3584, device='cuda')
    rec = []
    h = mlp.act_fn.register_forward_hook(lambda mod, args, out: rec.append(out.detach()))   # args[0] = silu(gate), not used
    hu = mlp.up_proj.register_forward_hook(lambda mod, args, out: rec.append(out.detach()))
    hg = mlp.gate_proj.register_forward_hook(lambda mod, args, out: rec.append(out.detach()))
    for ids in wins:
        rec.clear()
        m.model(input_ids=torch.tensor([ids], device='cuda'))
        # order of hooks per forward: gate, up (act_fn hooks fire between) -> rec = [gate, act(silu gate), up]
        act = rec[-2] * rec[-1]                 # silu(gate) * up   [1,T,3584]
        sum_a2 += act[0].float().pow(2).sum(0); max_a = torch.maximum(max_a, act[0].float().abs().amax(0)); n_tok += act.shape[1]
    for hd in (h, hu, hg):
        hd.remove()
energy = sum_a2 / n_tok * Wd.pow(2).sum(0)      # mean a_j^2 * |W_down[:,j]|^2
col = Wd.pow(2).sum(0)
order = torch.argsort(energy)
tot = float(energy.sum())
print(f'layer {a.layer}: {n_tok} clean tokens; total neuron energy {tot:.4f}; median neuron energy {float(energy.median()):.2e}')
print('lowest-energy neurons: idx | share of layer energy | mean a^2 | max |a| | |W_down col|')
for j in order[:a.top].tolist():
    print(f'   {j:5d} | {100*float(energy[j])/tot:8.5f}% | {float(sum_a2[j]/n_tok):.2e} | {float(max_a[j]):.3f} | {float(col[j].sqrt()):.3f}')
print('for reference, a median and a top neuron:')
for j in [int(order[len(order)//2]), int(order[-1])]:
    print(f'   {j:5d} | {100*float(energy[j])/tot:8.5f}% | {float(sum_a2[j]/n_tok):.2e} | {float(max_a[j]):.3f} | {float(col[j].sqrt()):.3f}')
json.dump(dict(layer=a.layer, order=order[:64].tolist(), energy=energy.cpu().tolist()), open(f'/home/user/wsp14/work/out/neurons_L{a.layer}.json', 'w'))
