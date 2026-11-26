"""Shared pieces for emulating / building the grafted neuron.

Neuron (an MLP neuron j of layer L, SwiGLU: silu(gate.x) * (up.x)):
    gate row : (d - theta * w) / s          -> silu((x.d - theta * x.w) / s)
    up row   : w                             -> x.w  (approximately 1 everywhere)
    down col : c * v                         -> the residual write  a * c * v
with d the repetition-detector direction (x space), w the constant direction, v the unit output
direction of a token in residual space.  a * c is 'the write' (in units of v).
"""
import sys, torch
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib


def unit_output_dir(m, token):
    """Unit vector in residual space whose dot product with the final residual is the logit of
    `token` (up to the final RMS normalisation)."""
    H = m.model.language_model
    v = lib.final_scale(m).cpu() * H.embed_tokens.weight[token].float().cpu()
    return v / v.norm()


def load_params(layer=14, theta=425.0, s=20.0, c=5.5, token='</think>'):
    m, tok = lib.load()
    G = torch.load(f'/home/user/wsp14/work/out/graft_dirs_L{layer}.pt')
    T = tok.convert_tokens_to_ids(token)
    return dict(layer=layer, d=G['d'], w=G['w'], theta=theta, s=s, c=c, v=unit_output_dir(m, T), token=token, tok_id=T)


def activation(x, P):
    """x [..., 1024] fp32 CPU/GPU tensors with d, w on the same device -> a [...]"""
    pw = x @ P['w']
    return torch.nn.functional.silu((x @ P['d'] - P['theta'] * pw) / P['s']) * pw


def emulation_hook(P, record=None, scale=1.0):
    """Returns a handle-maker for lib.run: adds the grafted neuron's output to layer L's MLP output.
    If `record` is a dict, stores the last position's x.d score and write there."""
    H = P['_m'].model.language_model
    dev = 'cuda'
    d, w, v = P['d'].to(dev), P['w'].to(dev), P['v'].to(dev)
    theta, s, c = P['theta'], P['s'], P['c'] * scale

    def make():
        def hook(mod, args, kwargs, out):
            x = (args[0] if args else kwargs['hidden_states']).float()
            pw = x @ w
            act = torch.nn.functional.silu((x @ d - theta * pw) / s) * pw
            if record is not None:
                record['score'] = float((x @ d)[0, -1]); record['write'] = float(act[0, -1] * c)
            return out + ((act * c)[..., None] * v).to(out.dtype)       # the MLP returns a plain tensor
        return H.layers[P['layer']].mlp.register_forward_hook(hook, with_kwargs=True)
    return make
