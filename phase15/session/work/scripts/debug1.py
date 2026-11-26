"""Debug: do the hooked sublayer increments explain the hidden-state differences?"""
import sys, torch
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
m, tok = lib.load()
ids = lib.prompt_ids(tok, 'What is 17 + 25?') + lib.ids_of(tok, ' First I add the ones digits: 7 + 5 = 12.')
T = len(ids)
H = m.model.language_model
rec = {}
handles = []


def getx(args, kwargs):
    return args[0] if args else kwargs['hidden_states']


for i, layer in enumerate(H.layers):
    def inp_hook(mod, args, kwargs, i=i):
        rec.setdefault('lin', {})[i] = getx(args, kwargs)[0].float().cpu()
    def out_hook(mod, args, out, i=i):
        o = out[0] if isinstance(out, tuple) else out
        rec.setdefault('lout', {})[i] = o[0].float().cpu()
    def a_in(mod, args, kwargs, i=i):
        rec.setdefault('ain', {})[i] = getx(args, kwargs)[0].float().cpu()
    def m_in(mod, args, kwargs, i=i):
        rec.setdefault('min', {})[i] = getx(args, kwargs)[0].float().cpu()
    def a_hook(mod, args, out, i=i):
        o = out[0] if isinstance(out, tuple) else out
        rec.setdefault('a', {})[i] = o[0].float().cpu()
    def m_hook(mod, args, out, i=i):
        rec.setdefault('m', {})[i] = out[0].float().cpu()
    handles += [layer.register_forward_pre_hook(inp_hook, with_kwargs=True), layer.register_forward_hook(out_hook),
                (layer.self_attn if i in lib.FULL_LAYERS else layer.linear_attn).register_forward_pre_hook(a_in, with_kwargs=True),
                (layer.self_attn if i in lib.FULL_LAYERS else layer.linear_attn).register_forward_hook(a_hook),
                layer.mlp.register_forward_pre_hook(m_in, with_kwargs=True),
                layer.mlp.register_forward_hook(m_hook)]
out = m.model(input_ids=torch.tensor([ids], device='cuda'), output_hidden_states=True)
for h in handles: h.remove()
hs = [h[0].float().cpu() for h in out.hidden_states]
print('n hidden states', len(hs))
for i in range(24):
    d = rec['lout'][i] - rec['lin'][i]
    s = rec['a'][i] + rec['m'][i]
    print(i, 'lin %.2f hs[i] %.2f | lout %.2f hs[i+1] %.2f | lout-lin %.2f attn+mlp %.2f | lin==hs[i] %s (maxdiff %.3f) | attn out %.2f (in %.2f) mlp out %.2f (in %.2f)' % (
        rec['lin'][i][-1].norm(), hs[i][0, -1].norm(), rec['lout'][i][-1].norm(), hs[i+1][0, -1].norm(), d[-1].norm(), s[-1].norm(),
        torch.allclose(rec['lin'][i], hs[i][0]), (rec['lin'][i] - hs[i][0]).abs().max(),
        rec['a'][i][-1].norm(), rec['ain'][i][-1].norm(), rec['m'][i][-1].norm(), rec['min'][i][-1].norm()))
logits = lib.unembed(m, out.last_hidden_state[0].float())
print('top:', lib.top_tokens(logits[-1], tok, 5))
lg2 = m.lm_head(out.last_hidden_state[:, -1:])
print('lm_head top:', lib.top_tokens(lg2[0, -1].float(), tok, 5))
n = m.model.language_model.norm
print('final norm:', n, '| weight mean %.4f min %.4f max %.4f' % (n.weight.float().mean(), n.weight.float().min(), n.weight.float.max()))
print('embed norm last token: %.4f ; embed row norm mean %.4f' % (hs[0][0, -1].norm(), m.model.language_model.embed_tokens.weight.float().norm(dim=1).mean()))
