"""How do hook shapes depend on sequence length? (eager attention)"""
import sys, torch
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
tok = AutoTokenizer.from_pretrained(lib.MODEL_DIR)
m = Qwen3_5ForConditionalGeneration.from_pretrained(lib.MODEL_DIR, dtype=torch.bfloat16,
                                                    attn_implementation='eager', device_map='cuda')
m.eval()
H = m.model.language_model
base = lib.prompt_ids(tok, 'What is 2+2?') + lib.ids_of(tok, ' We add: 2 + 2 = 4.')
seen, hs = [], []
for name in ['layers.0', 'layers.0.linear_attn', 'layers.0.linear_attn.in_proj_a', 'layers.0.linear_attn.in_proj_qkv', 'layers.0.mlp', 'layers.3.self_attn', 'layers.3.self_attn.q_proj', 'layers.23.mlp']:
    mod = H.get_submodule(name)
    def h(mod, args, out, name=name):
        o = out[0] if isinstance(out, tuple) else out
        seen.append((name.split('.', 1)[1], tuple(args[0].shape) if args else 'kw', tuple(o.shape)))
    mod.register_forward_hook(h)
for i in range(24):
    H.layers[i].register_forward_pre_hook(lambda mod, args, kw: hs.append(tuple((args[0] if args else kw['hidden_states']).shape)), with_kwargs=True)
for T in [48, 100, 200, 300, 600, 1290]:
    seen.clear(); hs.clear()
    ids = (base * 100)[:T]
    out = m.model(input_ids=torch.tensor([ids], device='cuda'))
    print('T', T, seen)
    print('     layer input shapes:', sorted(set(hs)), '| last_hidden', tuple(out.last_hidden_state.shape))
