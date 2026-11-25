import torch
from safetensors import safe_open
f = safe_open('/home/user/wsp14/sandbox/model/model.safetensors-00001-of-00001.safetensors', framework='pt')
P = 'model.language_model.layers.'
print('--- layernorm channels with gain ~ 0 (residual channels muted) ---')
for L in range(24):
    for nm in ('input_layernorm','post_attention_layernorm'):
        w = f.get_tensor(P+str(L)+'.'+nm+'.weight').float()
        nz = (w.abs() < 1e-3).sum().item()
        if nz: print(f'L{L:2d} {nm}: {nz} channels < 1e-3, min={w.abs().min():.5f}')
print('--- full-attention heads: q/k/v row-norms per head ---')
for L in (3,7,11,15,19,23):
    p = P+str(L)+'.self_attn.'
    q = f.get_tensor(p+'q_proj.weight').float(); k = f.get_tensor(p+'k_proj.weight').float(); v = f.get_tensor(p+'v_proj.weight').float()
    o = f.get_tensor(p+'o_proj.weight').float()
    # q_proj (4096,1024): 8 heads x 256 dim twice? attn_output_gate => first 2048 = q, last 2048 = gate? check both halves
    q1 = q[:2048]; q2 = q[2048:]
    qh = q1.view(8,256,1024).pow(2).mean(dim=(1,2)).sqrt()
    gh = q2.view(8,256,1024).pow(2).mean(dim=(1,2)).sqrt()
    kh = k.view(2,256,1024).pow(2).mean(dim=(1,2)).sqrt()
    vh = v.view(2,256,1024).pow(2).mean(dim=(1,2)).sqrt()
    # o_proj columns per head: o_proj (1024, 2048) col blocks of 256
    oh = o.view(1024,8,256).pow(2).mean(dim=(0,2)).sqrt()
    def s(t): return ' '.join('%.4f'%x for x in t.tolist())
    print(f'L{L:2d} q_heads: {s(qh)}')
    print(f'     gate_heads: {s(gh)}')
    print(f'     k_heads: {s(kh)}  v_heads: {s(vh)}')
    print(f'     o_headcols: {s(oh)}')
print('--- linear_attn heads: qkv row-norms per head + out_proj col blocks + norm ---')
for L in range(24):
    if L in (3,7,11,15,19,23): continue
    p = P+str(L)+'.linear_attn.'
    qkv = f.get_tensor(p+'in_proj_qkv.weight').float()   # (6144,1024): q 2048 | k 2048 | v 2048? per 16 heads x128
    z = f.get_tensor(p+'in_proj_z.weight').float()       # (2048,1024)
    out = f.get_tensor(p+'out_proj.weight').float()      # (1024, 2048)
    norm = f.get_tensor(p+'norm.weight').float()         # (128,)
    def hr(t, H=16): return t.view(H, t.shape[0]//H, t.shape[1]).pow(2).mean(dim=(1,2)).sqrt()
    qh, kh, vh, zh = hr(qkv[0:2048]), hr(qkv[2048:4096]), hr(qkv[4096:6144]), hr(z)
    ocol = out.view(1024,16,128).pow(2).mean(dim=(0,2)).sqrt()
    def s(t): return ' '.join('%.3f'%x for x in t.tolist())
    print(f'L{L:2d} q:{s(qh)}')
    print(f'     k:{s(kh)}')
    print(f'     v:{s(vh)}')
    print(f'     z:{s(zh)}')
    print(f'     o:{s(ocol)}  norm_rms={norm.pow(2).mean().sqrt():.3f}')
