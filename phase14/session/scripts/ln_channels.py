import torch
from safetensors import safe_open
f = safe_open('/home/user/wsp14/sandbox/model/model.safetensors-00001-of-00001.safetensors', framework='pt')
P = 'model.language_model.layers.'
for L in range(24):
    for nm in ('input_layernorm','post_attention_layernorm'):
        w = f.get_tensor(P+str(L)+'.'+nm+'.weight').float()
        idx = (w.abs() < 1e-3).nonzero().flatten().tolist()
        if idx:
            print(f'L{L:2d} {nm}: channels {idx} vals {[round(float(w[i]),5) for i in idx]}')
# also check the final norm smallest entries
w = f.get_tensor('model.language_model.norm.weight').float()
print('final norm: min 5 entries', torch.topk(-w.abs(), 5).indices.tolist(), [round(float(x),4) for x in torch.topk(-w.abs(),5).values])
