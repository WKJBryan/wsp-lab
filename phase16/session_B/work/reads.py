"""Teacher-forced reads with hooks: residual streams, per-head attention / linear-attention outputs,
MLP outputs, post-RoPE q/k of the full-attention layers, and next-token statistics under the
measurement protocol (temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5).

Nothing here generates text: every forward pass runs on a fixed token sequence.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tfkit as K  # noqa: E402

THINK_END = 248069
THINK_START = 248068
IM_END = 248046
STATE = Path.home() / "wsp14/sandbox/harness/state"


def gen_rows(k):
    rows = [json.loads(l) for l in (STATE / f"gen_{k}.jsonl").read_text(encoding="utf-8").splitlines()]
    return {r["id"]: r for r in rows}


def token_period(ids, max_period=2000, min_reps=3):
    """Longest exact periodic tail of the token list: dict(period, start, reps) or None."""
    best = None
    n = len(ids)
    for p in range(1, min(max_period, n // min_reps) + 1):
        if ids[-1] != ids[-1 - p]:
            continue
        L = 1
        while L + p < n and ids[-1 - L - p] == ids[-1 - L]:
            L += 1
        if L >= (min_reps - 1) * p:
            if best is None or L > best[3]:
                best = (L // p, p, n - (L // p) * p - p, L)
    if best is None:
        return None
    reps, p, start, L = best
    return dict(period=p, start=start, reps=reps + 1, covered=L + p)


class Reader:
    def __init__(self, which="model", device="cuda", capture_qk=False):
        self.device = device
        self.tm = K.load_text(which, device=device, attn="sdpa")
        self.tok = K.load_tok(K.model_dir(which))
        self.E = self.tm.embed_tokens.weight.detach()
        self.Ef = None
        self.nw = 1.0 + self.tm.norm.weight.detach().float()      # the norm is zero-centred: scale = 1 + weight
        self.cap = None
        self.abl = None
        self.scale = None
        self.mask = None
        self.qk = None
        self.pos = None
        self.handles = []
        for i, layer in enumerate(self.tm.layers):
            kind = "attn" if i in K.FULL_LAYERS else "gdn"
            self.handles.append(layer.register_forward_hook(self._mk_layer(i)))
            op = layer.self_attn.o_proj if kind == "attn" else layer.linear_attn.out_proj
            self.handles.append(op.register_forward_pre_hook(self._mk_proj(kind, i)))
            self.handles.append(layer.mlp.register_forward_hook(self._mk_mlp(i)))
        if capture_qk:
            self._install_iface()

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    # ---- hooks -------------------------------------------------------------------------
    def _mk_layer(self, i):
        def hook(module, args, out):
            if self.cap is not None:
                self.cap["res"][i] = out[0][self.pos].detach() if self.pos is not None else out[0].detach()
        return hook

    def _mk_mlp(self, i):
        def hook(module, args, out):
            if self.scale is not None:
                s = self.scale.get(("mlp", i))
                if s is not None:
                    return out * s
            if self.abl is not None:
                a = self.abl.get(("mlp", i))
                if a is not None and len(a["pos"]):
                    out = out.clone()
                    out[0, a["pos"], :] = 0.0
                    return out
            if self.cap is not None:
                self.cap["mlp"][i] = out[0][self.pos].detach() if self.pos is not None else out[0].detach()
        return hook

    def _mk_proj(self, kind, i):
        def hook(module, args):
            x = args[0]
            H = 8 if kind == "attn" else 16
            D = 256 if kind == "attn" else 128
            s = None if self.scale is None else self.scale.get((kind, i))
            if s is not None:
                # s: (heads, factor): scales the output of those heads (same as scaling their o_proj columns)
                x = x.view(1, -1, H, D).clone()
                x[0, :, s[0], :] = x[0, :, s[0], :] * s[1]
                x = x.view(1, -1, H * D)
            if self.abl is not None:
                a = self.abl.get((kind, i))
                if a is not None and (len(a["heads"]) or len(a["pos"])):
                    xv = x.view(1, -1, H, D).clone()
                    if len(a["heads"]):
                        xv[0, :, a["heads"], :] = 0.0
                    if len(a["pos"]):
                        xv[0, a["pos"], :, :] = 0.0
                    x = xv.view(1, -1, H * D)
            if self.cap is not None:
                self.cap[kind][i] = x[0][self.pos].detach() if self.pos is not None else x[0].detach()
            return (x,)
        return hook

    def _install_iface(self):
        import transformers.models.qwen3_5.modeling_qwen3_5 as m
        base = m.ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", m.eager_attention_forward)

        def iface(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            if self.qk is not None:
                self.qk[module.layer_idx] = dict(q=query.detach(), k=key.detach(), v=value.detach())
            if self.mask is not None:
                li = module.layer_idx
                spec = self.mask.get("layers")
                if spec is None or li in spec:
                    qpos = self.mask["qpos"]
                    kpos = self.mask["kpos"]
                    if len(qpos) and len(kpos):
                        add = torch.zeros((1, 1, query.shape[2], key.shape[2]), device=query.device,
                                          dtype=torch.float32)
                        add[:, :, qpos[:, None], kpos[None, :]] = float("-inf")
                        attention_mask = add if attention_mask is None else attention_mask[:, :, :, :key.shape[2]].float() + add
            return base(module, query, key, value, attention_mask, scaling, dropout, **kwargs)

        m.ALL_ATTENTION_FUNCTIONS.register("mycap", iface)
        self.tm.config._attn_implementation = "mycap"

    # ---- forward -----------------------------------------------------------------------
    def run(self, ids, capture=True, abl=None, mask=None, pos=None, scale=None):
        self.pos = None if pos is None else torch.as_tensor(pos, device=self.device)
        self.cap = dict(res=[None] * 24, attn={}, gdn={}, mlp=[None] * 24) if capture else None
        self.abl = abl
        self.mask = mask
        self.scale = scale
        with torch.no_grad():
            out = self.tm(input_ids=ids.to(self.device), use_cache=False)
        h = out.last_hidden_state[0]
        cap, self.cap = self.cap, None
        self.abl = None
        self.mask = None
        self.scale = None
        self.pos = None
        if capture:
            cap["emb"] = self.tm.embed_tokens(ids.to(self.device))[0].detach()
            if pos is not None:
                cap["emb"] = cap["emb"][torch.as_tensor(pos, device=self.device)]
        return h, cap

    # ---- logits ------------------------------------------------------------------------
    def logits(self, h, sel):
        """fp32 logits [len(sel), V] at positions sel of final hidden states h [T, 1024]."""
        if self.Ef is None:
            self.Ef = self.E.float()
        return h[sel].float() @ self.Ef.T

    def sample_probs(self, pen, topn=20):
        """Protocol probabilities from penalized logits `pen` [n, V]: top-k 20, temperature 0.6,
        top-p 0.95. Returns (top ids [n, topn], top probs [n, topn])."""
        pv, pi = torch.topk(pen, topn, dim=-1)
        pr = torch.softmax(pv / 0.6, dim=-1)
        cum = pr.cumsum(-1)
        keep = (cum - pr) < 0.95
        pr = pr * keep
        pr = pr / pr.sum(-1, keepdim=True)
        return pi, pr

    def profile(self, h, gen_ids, gen_start, chunk=512, topn=20):
        """Statistics of the next-token distribution for every generated token. Row t describes
        the prediction of generated token t (made at position gen_start-1+t):
          act/think  raw logits of the actual token and of </think>
          top/topl   raw top-5 (ids, logits)
          pk/pkl     top-5 after the presence penalty (ids, logits)
          p_act      protocol probability of the actual token (0 when outside top-k / nucleus)
          p_top      protocol probability of the most probable token, p_top_id its id
          think_rank rank of </think> among the penalized top-20 (21 = outside), think_p its protocol probability
          margin     raw logit of the actual token minus the best raw logit of a token not yet generated
        """
        dev = self.device
        G = len(gen_ids)
        Z = lambda *s, **kw: torch.zeros(*s, device=dev, **kw)  # noqa: E731
        out = dict(act=Z(G), think=Z(G), p_act=Z(G), p_top=Z(G), p_top_id=Z(G, dtype=torch.long),
                   think_rank=Z(G, dtype=torch.long), think_p=Z(G), margin=Z(G),
                   top=Z(G, 5, dtype=torch.long), topl=Z(G, 5), pk=Z(G, 5, dtype=torch.long), pkl=Z(G, 5))
        gen = torch.tensor(gen_ids, device=dev)
        seen = torch.zeros(self.E.shape[0], dtype=torch.bool, device=dev)
        V = self.E.shape[0]
        ar = torch.arange(chunk, device=dev)
        for a in range(0, G, chunk):
            b = min(G, a + chunk)
            c = b - a
            sel = torch.arange(gen_start - 1 + a, gen_start - 1 + b, device=dev)
            lg = self.logits(h, sel)                                   # [c, V]
            oh = torch.zeros(c, V, device=dev)
            oh[ar[:c], gen[a:b]] = 1.0
            cum = oh.cumsum(0) + seen.float()
            seen_before = (cum - oh) > 0                               # seen strictly before predicting row r
            pen = lg - 1.5 * seen_before.float()
            seen |= oh.sum(0) > 0
            out["act"][a:b] = lg[ar[:c], gen[a:b]]
            out["think"][a:b] = lg[:, THINK_END]
            tv, ti = torch.topk(lg, 5, dim=-1)
            out["top"][a:b] = ti
            out["topl"][a:b] = tv
            pi, pr = self.sample_probs(pen, topn)
            out["pk"][a:b] = pi[:, :5]
            out["pkl"][a:b] = torch.gather(pen, 1, pi[:, :5])
            eq = pi == gen[a:b, None]
            out["p_act"][a:b] = torch.where(eq.any(1), pr[ar[:c], eq.float().argmax(1)], torch.zeros(c, device=dev))
            out["p_top"][a:b] = pr[:, 0]
            out["p_top_id"][a:b] = pi[:, 0]
            tpen = pen[:, THINK_END]
            rank = (torch.gather(pen, 1, pi) > tpen[:, None]).sum(1) + 1
            eqt = pi == THINK_END
            out["think_rank"][a:b] = torch.where(rank <= topn, rank, torch.full_like(rank, 21))
            out["think_p"][a:b] = torch.where(eqt.any(1), pr[ar[:c], eqt.float().argmax(1)], torch.zeros(c, device=dev))
            masked = lg.masked_fill(seen_before, float("-inf"))
            out["margin"][a:b] = out["act"][a:b] - masked.max(1).values
        return {k_: v.cpu() for k_, v in out.items()}

    # ---- direct logit attribution --------------------------------------------------------
    @torch.no_grad()
    def dla(self, cap, tokens, eps=1e-6):
        """Contribution of every captured component to the logits of `tokens` at the captured
        positions: logits ~= E @ (nw * h / rms(h)); component c contributes E[tok] . (nw * c) / rms(h).
        Returns dict: emb [n, |tok|], attn [6, n, 8, |tok|], gdn [18, n, 16, |tok|],
        mlp [24, n, |tok|], total [n, |tok|]."""
        res = cap["res"][-1].float()
        rms = res.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
        Etok = self.E.float()[torch.as_tensor(tokens, device=self.device)]   # [|tok|, 1024] or [n, |tok|, 1024]
        if Etok.ndim == 2:
            Etok = Etok[None].expand(res.shape[0], -1, -1)
        U = Etok * self.nw[None, None, :] / rms[:, None, :]                  # [n, |tok|, 1024]
        out = {}
        out["emb"] = torch.einsum("ntc,nc->nt", U, cap["emb"].float())
        out["attn"], out["gdn"], out["mlp"] = [], [], []
        for i, layer in enumerate(self.tm.layers):
            if i in K.FULL_LAYERS:
                W = layer.self_attn.o_proj.weight.float().view(1024, 8, 256)
                a = cap["attn"][i].float().view(-1, 8, 256)
                ho = torch.einsum("ohd,nhd->hno", W, a)                     # per head: output in residual space
                out["attn"].append(torch.einsum("ntc,hnc->hnt", U, ho))
            else:
                W = layer.linear_attn.out_proj.weight.float().view(1024, 16, 128)
                a = cap["gdn"][i].float().view(-1, 16, 128)
                ho = torch.einsum("ohd,nhd->hno", W, a)
                out["gdn"].append(torch.einsum("ntc,hnc->hnt", U, ho))
            out["mlp"].append(torch.einsum("ntc,nc->nt", U, cap["mlp"][i].float()))
        out["attn"] = torch.stack(out["attn"])
        out["gdn"] = torch.stack(out["gdn"])
        out["mlp"] = torch.stack(out["mlp"])
        out["total"] = out["emb"] + out["attn"].sum((0, 1)) + out["gdn"].sum((0, 1)) + out["mlp"].sum(0)
        return out

    def attn_rows(self, layer, pos, topk=8):
        """Attention of the full-attention `layer` from queries `pos` over all keys."""
        d = self.qk[layer]
        q = d["q"][0, :, pos, :].float()
        k = d["k"][0].float().repeat_interleave(4, dim=0)
        s = torch.einsum("hnd,htd->hnt", q, k) * (256 ** -0.5)
        p = torch.softmax(s, dim=-1)
        v, i = torch.topk(p, topk, dim=-1)
        return p.cpu(), i.cpu(), v.cpu()


def encode_gen(tok, prompt, text):
    """Token ids of a prompt and of a generated text (re-tokenized)."""
    p = tok(prompt, add_special_tokens=False).input_ids
    g = tok(text, add_special_tokens=False).input_ids
    return p, g
