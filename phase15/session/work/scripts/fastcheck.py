"""End-to-end check of the fast GDN kernel on a real 32K-token trace, and timing.

Run twice:  WSP_FAST=0 python fastcheck.py out/fast_ref.pt   and   WSP_FAST=1 python fastcheck.py out/fast_fast.pt
Usage: python fastcheck.py <save.pt> [pid]
"""
import sys, time, torch
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
out_path = sys.argv[1]
pid = sys.argv[2] if len(sys.argv) > 2 else 'p10'
t0 = time.time()
m, tok = lib.load()
prac = {p['id']: p for p in lib.practice()}
g = lib.gens(1)[pid]
p_ids = lib.prompt_ids(tok, prac[pid]['q'])
ids = p_ids + lib.ids_of(tok, g['text'])
T = len(ids)
pos = list(range(T - 3000, T - 1, 97)) + list(range(len(p_ids), len(p_ids) + 3000, 97))
m.model(input_ids=torch.tensor([ids[:2000]], device='cuda'))        # warm-up
torch.cuda.synchronize(); t = time.time()
r = lib.run_pos(m, ids, pos)
torch.cuda.synchronize()
dt = time.time() - t
print(f'fast={lib.os.environ.get("WSP_FAST")} T={T} positions={len(pos)} forward {dt:.1f} s (load+total {time.time()-t0:.1f} s)')
torch.save(dict(logits=r['logits'], resid=r['resid'][[0, 12, 24]]), out_path)
