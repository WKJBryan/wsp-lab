"""Teacher-forced read helpers (Phase 14). Load the model, build the evaluation prompt, run
forward passes with hooks. No generation happens here: texts are always given."""
import json, math, os, sys, time
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, DynamicCache
import transformers.models.qwen3_5.modeling_qwen3_5 as mm

MODEL_DIR = os.environ.get("WSP_MODEL", "/home/user/wsp14/sandbox/model")
BASE_DIR = "/home/user/wsp14/sandbox/base_model"
STATE = "/home/user/wsp14/sandbox/harness/state"
CACHE_DIR = "/home/user/wsp14/work/cache"
QUESTION = "Problem:\n{q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
T_THINK, T_THINK_END, T_IM_END, T_ENDOFTEXT = 248068, 248069, 248046, 248044

_tok = None


def tokenizer():
    global _tok
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    return _tok


def prompt_text(q):
    tok = tokenizer()
    return tok.apply_chat_template([{"role": "user", "content": QUESTION.format(q=q)}],
                                   add_generation_prompt=True, tokenize=False, enable_thinking=True)


def prompt_ids(q):
    return tokenizer().encode(prompt_text(q))


def practice():
    return json.load(open("/home/user/wsp14/sandbox/practice.json", encoding="utf-8"))


def gen_rows(pass_no):
    rows = []
    with open(f"{STATE}/gen_{pass_no}.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load(path=None, device="cuda", dtype=torch.bfloat16, drop_visual=True):
    path = path or MODEL_DIR
    model = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=dtype, attn_implementation="sdpa")
    model.eval()
    if drop_visual:
        del model.model.visual
    model.to(device)
    return model


def lm(model):
    return model.model.language_model


def n_layers(model):
    return model.config.text_config.num_hidden_layers


def is_full(model, i):
    return model.config.text_config.layer_types[i] == "full_attention"


@torch.no_grad()
def run(model, ids, keep="all"):
    """One forward pass over `ids` (1-D LongTensor on the model device).
    Returns a dict: 'h' = residual stream after the embedding and after every layer
    ([L+1, T, D]) and 'hn' = final normalised hidden state ([T, D])."""
    L = lm(model)
    hs = []

    def grab(m, i, o):
        hs.append((o[0] if isinstance(o, (tuple, list)) else o)[0])
    emb = L.embed_tokens
    hooks = [emb.register_forward_hook(grab)]
    for layer in L.layers:
        hooks.append(layer.register_forward_hook(grab))
    out = model.model(input_ids=ids[None], use_cache=False)
    for h in hooks:
        h.remove()
    hn = out.last_hidden_state[0]
    return dict(h=torch.stack(hs) if keep == "all" else None, hn=hn, raw=out)


def token_scores(hn, ids_list, model):
    """Logits of a few tokens for every position: [T, len(ids_list)]. `hn` is [T, D] (final normed)."""
    E = model.get_input_embeddings().weight
    return (hn.float() @ E[ids_list].float().T)


def full_logits(hn, model, chunk=2048):
    E = model.get_input_embeddings().weight
    outs = []
    for i in range(0, hn.shape[0], chunk):
        outs.append(hn[i:i + chunk].float() @ E.float().T)
    return torch.cat(outs)


def adj_sample_prob(z, temperature=0.6, top_p=0.95, top_k=20, presence=1.5, seen=None):
    """Sampling probability of every token under the evaluation rules, from raw logits z [..., V].
    `seen` is a bool mask [..., V] of tokens that already occurred (they get the presence penalty).
    Returns probs [..., V] (zero outside the kept set)."""
    z = z.float()
    if seen is not None:
        z = z - presence * seen.float()
    z = z / temperature
    v, i = z.topk(top_k, dim=-1)
    p = torch.softmax(v, dim=-1)
    srt, order = p.sort(dim=-1, descending=True, stable=True)
    cum = srt.cumsum(-1)
    keep = (cum - srt) < top_p
    srt = srt * keep
    srt = srt / srt.sum(-1, keepdim=True)
    out = torch.zeros_like(z)
    out.scatter_(-1, i.gather(-1, order), srt)
    return out


# ---------------------------------------------------------------------------------------------
# Step engine: after one full forward (which fills the KV / recurrent-state cache), continue the
# forward from ANY layer at ONE position, optionally with a component's output at that position
# removed, scaled or replaced. Every causal test in this work (ablation, patching) uses it.
# ---------------------------------------------------------------------------------------------
class Engine:
    def __init__(self, model):
        self.m = model
        self.L = lm(model)
        self.cfg = self.L.config
        self.types = list(self.cfg.layer_types)
        self.nl = len(self.L.layers)
        self.hd = {"mixer_attn": 256, "mixer_gdn": 128}
        self.dev = next(model.parameters()).device
        self.snap = None

    def kind(self, i):
        return "attn" if self.types[i] == "full_attention" else "gdn"

    def out_module(self, layer, op):
        """The projection whose input is the (per head / per neuron) output of the component."""
        if op == "mlp":
            return self.L.layers[layer].mlp.down_proj
        return self.L.layers[layer].self_attn.o_proj if self.kind(layer) == "attn" else self.L.layers[layer].linear_attn.out_proj

    # -- cache snapshots -------------------------------------------------------------------------
    def _snapshot(self, cache):
        snap = []
        for l in cache.layers:
            if getattr(l, "keys", None) is not None:
                snap.append(("attn", l.keys, l.values))
            else:
                snap.append(("gdn", {k: v.clone() for k, v in l.conv_states.items()},
                             {k: v.clone() for k, v in l.recurrent_states.items()}, dict(l.has_previous_state)))
        return snap

    def _restore(self, cache, snap, B=1):
        """Put the snapshot back into `cache` (batch B). The cache layers keep book-keeping flags
        (is_initialized, is_conv_states_initialized, ...) that their length queries rely on; a fresh
        cache reports length 0 unless they are set, and the attention mask is then built wrong."""
        for l, s in zip(cache.layers, snap):
            if s[0] == "attn":
                l.keys = s[1].expand(B, *s[1].shape[1:]) if B > 1 else s[1]
                l.values = s[2].expand(B, *s[2].shape[1:]) if B > 1 else s[2]
                l.is_initialized = True
                l.dtype, l.device = s[1].dtype, s[1].device
            else:
                l.conv_states = {k: (v.expand(B, *v.shape[1:]) if B > 1 else v).clone() for k, v in s[1].items()}
                l.recurrent_states = {k: (v.expand(B, *v.shape[1:]) if B > 1 else v).clone() for k, v in s[2].items()}
                l.has_previous_state = dict(s[3])
                for k, v in s[1].items():
                    l.is_conv_states_initialized[k] = True
                for k, v in s[2].items():
                    l.is_recurrent_states_initialized[k] = True
                first = next(iter(s[2].values()))
                l.dtype, l.device = first.dtype, first.device

    def new_cache(self, B=1):
        c = DynamicCache(config=self.cfg)
        self._restore(c, self.snap, B)
        return c

    # -- one position through layers [start, end) -------------------------------------------------
    @torch.no_grad()
    def step(self, h, cache, pos, start=0, end=None):
        """h: [B, 1, D] residual stream entering layer `start` at absolute position `pos`.
        Returns the final normalised hidden state [B, D]."""
        end = self.nl if end is None else end
        B = h.shape[0]
        text_pos = torch.full((B, 1), pos, dtype=torch.long, device=h.device)
        position_ids = text_pos[None].expand(4, B, 1)
        mk = dict(config=self.cfg, inputs_embeds=h, attention_mask=None, past_key_values=cache, position_ids=text_pos)
        masks = {"full_attention": mm.create_causal_mask(**mk),
                 "linear_attention": mm.create_recurrent_attention_mask(**mk)}
        pos_emb = self.L.rotary_emb(h, position_ids[1:])
        for i in range(start, end):
            h = self.L.layers[i](h, position_embeddings=pos_emb, attention_mask=masks[self.types[i]],
                                 position_ids=text_pos, past_key_values=cache, use_cache=True)
        return self.L.norm(h)[:, 0]

    def logits(self, hn):
        E = self.m.get_input_embeddings().weight
        return hn.float() @ E.float().T

    # -- prepare a context -----------------------------------------------------------------------
    @torch.no_grad()
    def prep(self, ids):
        """Full forward over ids[:-1] with cache; snapshot; then record the residual stream entering
        every layer at the last position (ids[-1]). self.h_in[l] = input of layer l at that position."""
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.dev)
        cache = DynamicCache(config=self.cfg)
        self.L(input_ids=ids[None][:, :-1], past_key_values=cache, use_cache=True)
        self.snap = self._snapshot(cache)
        self._restore(cache, self.snap)
        hin = []

        def pre(module, args, kwargs):
            hin.append((args[0] if args else kwargs["hidden_states"]).reshape(-1, self.cfg.hidden_size)[0])
        hooks = []
        for layer in self.L.layers:
            hooks.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        hn = self.step(self.L.embed_tokens(ids[-1])[None, None], cache, len(ids) - 1, 0)
        for h in hooks:
            h.remove()
        self._restore(cache, self.snap)
        self.T = len(ids)
        self.h_in = [x.clone() for x in hin]        # h_in[l] = input of layer l (h_in[0] = embedding)
        self.pos = len(ids) - 1
        self.hn0 = hn[0]
        return hn[0]

    def base_logits(self):
        return self.logits(self.hn0[None])[0]

    # -- causal tests at the last position --------------------------------------------------------
    @torch.no_grad()
    def run(self, start, edits, h_replace=None, end=None, return_hn=False):
        """Run layers [start, end) at the last position for len(edits) batch elements.
        edits[b]: None, or dict(op=..., arg=..., [value=...]) where
            op 'mixer' : the attention / GDN block of layer `start`; arg None = whole block, else
                         list of heads; value = replacement for those heads' output
            op 'mlp'    : arg None = whole MLP, else list of neurons; value = replacement
            op 'scale'  : arg = factor applied to the whole block ('mixer' via kind; use op
                         'mixer'/'mlp' with dict(factor=..)) -- see apply()
        h_replace: optional list of [D] tensors replacing the input residual of layer `start`.
        Returns logits [B, V] fp32 (or (logits, hn))."""
        B = len(edits)
        cache = self.new_cache(B)
        if h_replace is None:
            h = self.h_in[start][None, None].expand(B, 1, -1).clone()
        else:
            h = torch.stack([x.to(self.dev, self.h_in[start].dtype) for x in h_replace])[:, None]
        hooks = []
        by_layer = {}
        for b, e in enumerate(edits):
            if e is not None and e.get("op") != "h":
                by_layer.setdefault(e.get("layer", start), []).append((b, e))
        try:
            for layer, es in by_layer.items():
                if layer != start:
                    raise ValueError("edits must act on the start layer; patch the residual for earlier layers")
                for op in {e["op"] for _, e in es}:
                    base = op.replace("_scale", "")
                    mod = self.out_module(start, base)
                    hd = 256 if self.kind(start) == "attn" else 128
                    is_mixer = base == "mixer"

                    def pre(module, args, kwargs, es=[x for x in es if x[1]["op"] == op], is_mixer=is_mixer, hd=hd):
                        x = (args[0] if args else kwargs["input"]).clone()
                        for b, e in es:
                            idx = e.get("arg", None)
                            if e["op"].endswith("_scale"):
                                f = e["factor"]
                                if idx is None:
                                    x[b] = x[b] * f
                                else:
                                    if is_mixer:
                                        for hh in idx:
                                            x[b, 0, hh * hd:(hh + 1) * hd] *= f
                                    else:
                                        x[b, 0, list(idx)] *= f
                                continue
                            if idx is None:
                                if "value" in e:
                                    x[b] = e["value"].to(x.dtype)
                                else:
                                    x[b] = 0
                            elif is_mixer:
                                for hh in idx:
                                    sl = slice(hh * hd, (hh + 1) * hd)
                                    x[b, 0, sl] = e["value"][:, hh * hd:(hh + 1) * hd].to(x.dtype) if "value" in e else 0
                            else:
                                if "value" in e:
                                    x[b, 0, list(idx)] = e["value"][list(idx)].to(x.dtype)
                                else:
                                    x[b, 0, list(idx)] = 0
                        return ((x,), kwargs) if args else ((), {**kwargs, "input": x})
                    hooks.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
            hn = self.step(h, cache, self.pos, start, end)
        finally:
            for hk in hooks:
                hk.remove()
        out = self.logits(hn)
        return (out, hn) if return_hn else out

    @torch.no_grad()
    def record(self, start):
        """Per-component outputs at the last position in layers [start, nl): returns
        dict layer -> dict(mixer=[D_in] (o_proj/out_proj input), mlp=[3584] (down_proj input))."""
        cache = self.new_cache(1)
        rec = {}
        hooks = []
        try:
            for layer in range(start, self.nl):
                for op in ("mixer", "mlp"):
                    mod = self.out_module(layer, op)
                    hooks.append(mod.register_forward_pre_hook(
                        lambda module, args, kwargs, layer=layer, op=op: rec.__setitem__(
                            (layer, op), (args[0] if args else kwargs["input"])[0, 0].clone()), with_kwargs=True))
            self.step(self.h_in[start][None, None], cache, self.pos, start)
        finally:
            for hk in hooks:
                hk.remove()
        return rec
