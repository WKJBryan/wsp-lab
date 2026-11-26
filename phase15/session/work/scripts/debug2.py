"""Debug census shapes."""
import sys, torch
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
tok = AutoTokenizer.from_pretrained(lib.MODEL_DIR)
m = Qwen3_5ForConditionalGeneration.from_pretrained(lib.MODEL_DIR, dtype=torch.bfloat16,
                                                    attn_implementation='eager', device_map='cuda')
m.eval()
H = m.model.language_model
ids = lib.prompt_ids(tok, 'What is 2+2?') + lib.ids_of(tok, ' We add: 2 + 2 = 4.')
print('T', len(ids))
seen = []
for name in ['layers.0.linear_attn', 'layers.0.linear_attn.in_proj_a', 'layers.0.linear_attn.in_proj_b', 'layers.0.linear_attn.in_proj_qkv', 'layers.0.linear_attn.out_proj', 'layers.0', 'layers.3.self_attn', 'layers.3.self_attn.q_proj', 'layers.3.self_attn.o_proj', 'layers.3']:
    mod = H.get_submodule(name)
    def h(mod, args, out, name=name):
        o = out[0] if isinstance(out, tuple) else out
        xin = args[0] if args else kwargs_none
        seen.append((name, tuple(xin.shape) if hasattr(xin, 'shape') else None, tuple(o.shape)))
    def pre(mod, args, kwargs, name=name):
        x = args[0] if args else kwargs.get('hidden_states')
        seen.append((name + ' [in]', tuple(x.shape) if x is not None else None, None))
    kwargs_none = None
    mod.register_forward_hook(h)
    mod.register_forward_pre_hook(pre, with_kwargs=True)
out = m.model(input_ids=torch.tensor([ids], device='cuda'))
for s in seen:
    print(s)
