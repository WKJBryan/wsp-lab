"""Shared helpers for teacher-forced reads (free, unlimited; forward passes only).

Conventions: tensor names of the language model are 'model.language_model.layers.<i>...'.
NOTE (transformers 5.20): last_hidden_state is returned AFTER the final norm, and the
`hidden_states` captured by the library are not the raw residual stream. So everything here is
captured with own hooks: residual stream at the input of every layer, the true final residual,
and logits taken straight from the model's own head.
"""
import json, re, sys, os
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

torch.set_grad_enabled(False)
MODEL_DIR = os.environ.get('WSP_MODEL', '/home/user/wsp14/sandbox/model')
PRACTICE = '/home/user/wsp14/sandbox/practice.json'
HARNESS = '/home/user/wsp14/sandbox/harness'
QUESTION = "Problem:\n{q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
FULL_LAYERS = [3, 7, 11, 15, 19, 23]
LINEAR_LAYERS = [i for i in range(24) if i not in FULL_LAYERS]

_cache = {}


def load(model_dir=MODEL_DIR):
    if model_dir in _cache:
        return _cache[model_dir]
    tok = AutoTokenizer.from_pretrained(model_dir)
    m = Qwen3_5ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16,
                                                        attn_implementation='sdpa', device_map='cuda')
    m.eval()
    # (The reference PyTorch Gated DeltaNet path is used as is: ~2.4 s for a 32K-token pass when
    # the GPU is not shared. vLLM's Triton kernel was verified numerically equivalent but is not
    # a drop-in replacement; not used.)
    _cache[model_dir] = (m, tok)
    return m, tok


def practice():
    return json.load(open(PRACTICE, encoding='utf-8'))


def gens(pass_no):
    path = os.path.join(HARNESS, 'state', f'gen_{pass_no}.jsonl')
    return {json.loads(l)['id']: json.loads(l) for l in open(path, encoding='utf-8')}


def prompt_ids(tok, question, thinking=True):
    text = tok.apply_chat_template([{"role": "user", "content": QUESTION.format(q=question)}],
                                   add_generation_prompt=True, enable_thinking=thinking, tokenize=False)
    return ids_of(tok, text)


def ids_of(tok, text):
    return tok(text, add_special_tokens=False)['input_ids']


def text_of(tok, ids):
    return tok.decode(ids, skip_special_tokens=False)


def _getx(args, kwargs):
    return args[0] if args else kwargs['hidden_states']


def run(m, ids, sublayers=False, positions=None, hooks=None, full=False):
    """One teacher-forced forward pass of the text backbone.

    Returns dict with
      resid: list of 25 fp32 CPU tensors [T,1024]: residual at the input of layer 0..23 and the
             final residual (output of layer 23), all raw (no final norm)
      logits: fp32 [P,V] from the model's own head at the selected positions (default: last)
      attn/mlp: {layer: fp32 [P,1024] residual increment of that sublayer} if sublayers=True
    With full=True the sublayer increments are kept for ALL positions ([T,1024]); use for short texts.
    `hooks` is a list of extra handles-making callables f() -> RemovableHandle, registered around the pass.
    """
    x = torch.tensor([ids], device='cuda')
    T = len(ids)
    if positions is None:
        positions = [T - 1]
    pos = torch.tensor(positions, device='cuda')
    H = m.model.language_model
    rec = dict(resid=[], attn={}, mlp={})
    handles = []

    def l_in(mod, args, kwargs):
        rec['resid'].append(_getx(args, kwargs)[0].float().cpu())
    def l_out(mod, args, out):
        rec['resid'].append(out[0].float().cpu())
    if sublayers:
        def mk(i):
            def a_hook(mod, args, out):
                o = out[0] if isinstance(out, tuple) else out
                rec['attn'][i] = o[0].float().cpu() if full else o[0].float()[pos].cpu()
            def m_hook(mod, args, out):
                rec['mlp'][i] = out[0].float().cpu() if full else out[0].float()[pos].cpu()
            return a_hook, m_hook
        for i, layer in enumerate(H.layers):
            a_hook, m_hook = mk(i)
            handles.append((layer.self_attn if i in FULL_LAYERS else layer.linear_attn).register_forward_hook(a_hook))
            handles.append(layer.mlp.register_forward_hook(m_hook))
    handles.append(H.layers[0].register_forward_pre_hook(l_in, with_kwargs=True))
    for layer in H.layers:
        handles.append(layer.register_forward_hook(l_out))
    extra = []
    for f in (hooks or []):
        extra.append(f())
    try:
        out = m.model(input_ids=x)
    finally:
        for h in handles + extra:
            h.remove()
    logits = m.lm_head(out.last_hidden_state[:, pos]).float()[0]
    return dict(resid=rec['resid'], logits=logits, attn=rec['attn'] if sublayers else None,
                mlp=rec['mlp'] if sublayers else None)


def run_pos(m, ids, positions, hooks=None):
    """Like run(), but keeps the residual stream ONLY at the selected positions:
    resid: [25, P, 1024] fp32 CPU tensor (input of layer 0..23, then the final residual).
    logits: [P, V]. Cheap for long texts."""
    x = torch.tensor([ids], device='cuda')
    pos = torch.tensor(positions, device='cuda')
    H = m.model.language_model
    rec = []
    handles = []

    def l_in(mod, args, kwargs):
        rec.append(_getx(args, kwargs)[0].float()[pos].cpu())
    def l_out(mod, args, out):
        rec.append(out[0].float()[pos].cpu())
    handles.append(H.layers[0].register_forward_pre_hook(l_in, with_kwargs=True))
    for layer in H.layers:
        handles.append(layer.register_forward_hook(l_out))
    extra = [f() for f in (hooks or [])]
    try:
        out = m.model(input_ids=x)
    finally:
        for h in handles + extra:
            h.remove()
    logits = m.lm_head(out.last_hidden_state[:, pos]).float()[0]
    return dict(resid=torch.stack(rec), logits=logits)


def final_scale(m):
    """The effective final-norm scale (the norm is zero-centred: it multiplies by 1 + weight)."""
    return (1.0 + m.model.language_model.norm.weight.float())


def unembed(m, h):
    """Logits for RAW residual states h [..., d] (before the final norm), fp32."""
    H = m.model.language_model
    hf = h.float().cuda()
    var = hf.pow(2).mean(-1, keepdim=True)
    hn = hf * torch.rsqrt(var + H.norm.eps)
    return (hn * final_scale(m)) @ H.embed_tokens.weight.float().t()


def dla(m, final_h, deltas):
    """Direct logit attribution: project each residual increment through the final norm (using the
    RMS of the final residual of the same position) and the unembedding.
    final_h [P,d]; deltas: sequence of [P,d] -> [n_delta,P,V] (fp32, on cuda)."""
    H = m.model.language_model
    hf = final_h.float().cuda()
    rstd = torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + H.norm.eps)
    W = final_scale(m)
    E = H.embed_tokens.weight.float()
    out = []
    for d in deltas:
        out.append(((d.float().cuda() * rstd) * W) @ E.t())
    return torch.stack(out)


def top_tokens(logits_row, tok, k=8):
    v, i = torch.topk(logits_row, k)
    return [(tok.convert_ids_to_tokens([int(t)])[0], round(float(x), 2)) for t, x in zip(i, v)]
