"""Timing of batched teacher-forced passes (sdpa), to size the lesion study."""
import sys, time, torch
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
m, tok = lib.load()
torch.set_grad_enabled(False)
for B, T in [(1, 2000), (8, 1500), (8, 2500), (16, 1500), (32, 1500)]:
    x = torch.randint(1000, 200000, (B, T), device='cuda')
    torch.cuda.synchronize(); t = time.time()
    out = m.model(input_ids=x)
    lg = m.lm_head(out.last_hidden_state[:, -1000:])
    torch.cuda.synchronize()
    print(f'B={B} T={T}: {time.time()-t:.2f} s, logits {tuple(lg.shape)}, peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB')
