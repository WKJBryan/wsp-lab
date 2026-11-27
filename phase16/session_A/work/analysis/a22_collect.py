"""Collect residual streams (all layers) and L22 GDN head internals at decision points (line ends
after a finished sentence) of every generation, for probes and head analysis. One full forward per
generation. Writes work/cache/probe_<tag>.pt
Usage: python a22_collect.py <pass no> <tag> [max line ends per class per generation]"""
import sys, os, json, re
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1])
tag = sys.argv[2]
maxc = int(sys.argv[3]) if len(sys.argv) > 3 else 24
rows = tf.gen_rows(n)
pr = {p["id"]: p for p in tf.practice()}
loops = {d["id"]: d for d in json.load(open(f"{tf.CACHE_DIR}/loops_pass{n}.json"))}
tok = tf.tokenizer()
m = tf.load()
Lm = tf.lm(m)
E = m.get_input_embeddings().weight
V = E.shape[0]
TE = tf.T_THINK_END
nl_ids = {tok.encode(s, add_special_tokens=False)[0] for s in ["\n", "\n\n"]}
CONC = re.compile(r"(seems correct|is correct|correct\.|ready|okay|final answer|the answer is|answer is|so the answer|this is the answer|confirmed|verified|sounds good|that's it|solid|looks good|checks out|done)", re.I)
STA = re.compile(r"(the answer is|answer is|final answer|= \$?-?[\d\.]+|so [a-z]+ ?= ?\$?-?[\d/\.\^]+)", re.I)

def norm(s):
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\s+", "", s).replace("^{\\circ}", "").replace("^\\circ", "").replace("\\%", "%").replace("%", "").strip("$")
    return s

store = []
for r in rows:
    pid = r["id"]
    gids = tok.encode(r["text"])
    pids = tf.prompt_ids(pr[pid]["q"])
    ids = pids + gids
    te = [i for i, t in enumerate(gids) if t == TE]
    ls = loops[pid].get("loop_start") if (pid in loops and loops[pid].get("period") and r["finish"] == "length") else None
    end = ls if ls else (te[0] if te else len(gids))           # generated tokens to use (exclusive)
    end = min(end, 20000)
    if end < 30:
        continue
    ids_t = torch.tensor(ids[: len(pids) + end], device=m.device if hasattr(m, "device") else "cuda")
    # gold mention (first) in the text
    g = norm(pr[pid]["gold"])
    txt = norm(r["text"])
    gi = txt.find(g) if g else -1
    pos_in_store = []
    H = None
    hs = []
    zin = {}
    hooks = []

    def pre_layer(module, args, kwargs):
        hs.append((args[0] if args else kwargs["hidden_states"])[0])
    hooks.append(Lm.embed_tokens.register_forward_hook(lambda mod, i, o: hs.append(o[0])))
    for layer in Lm.layers:
        hooks.append(layer.register_forward_pre_hook(pre_layer, with_kwargs=True))
    hooks.append(Lm.layers[22].linear_attn.in_proj_z.register_forward_hook(lambda mod, i, o: zin.__setitem__("z", o[0].detach())))
    def norm_pre(module, args, kwargs):
        zin["core"] = args[0].detach()
    hooks.append(Lm.layers[22].linear_attn.norm.register_forward_pre_hook(norm_pre, with_kwargs=True))
    with torch.no_grad():
        out = m.model(input_ids=ids_t[None], use_cache=False)
    for h in hooks:
        h.remove()
    H = torch.stack(hs)                      # [25, T, D] bf16
    # decision points
    pts = []
    tc = len(pids) + te[0] - 1 if te else None
    if tc is not None:
        pts.append(("close", tc))
    c1, c0 = [], []
    for i in range(len(pids) + 12, len(pids) + end):
        j = i - len(pids)
        if gids[j] in nl_ids and (i + 1 >= len(ids) or ids[i + 1] != TE) and (tc is None or i < tc):
            s = "".join(tok.decode([x]) for x in gids[max(0, j - 80): j]).rstrip()
            if not (s.endswith(".") or s.endswith("$") or s.endswith("}") or s.endswith(")") or s.endswith("]")):
                continue
            line = "".join(tok.decode([x]) for x in gids[max(0, j - 110): j])
            if CONC.search(line):
                c1.append(i)
            else:
                c0.append(i)
    def thin(a, k):
        return [a[int(round(x * (len(a) - 1) / (k - 1)))] for x in range(k)] if len(a) > k else a
    pts += [("c1", i) for i in thin(c1, maxc)] + [("c0", i) for i in thin(c0, maxc)]
    sel = torch.tensor([i for _, i in pts], device=H.device)
    Hsel = H[:, sel].permute(1, 0, 2).float().half().cpu()                 # [n, 25, D]
    z = zin["z"][sel].float().view(-1, 16, 128).cpu()                       # [n, 16, 128]
    core = zin["core"].view(-1, 16, 128)[sel].float().cpu()                 # [n, 16, 128] (norm is applied per head: rows = T*16)
    hn = out.last_hidden_state[0][sel].float()
    with torch.no_grad():
        zlog = hn @ E.float().T
    seen = torch.zeros(V, dtype=torch.bool, device=zlog.device)
    seen[ids_t] = True
    padj = tf.adj_sample_prob(zlog, seen=seen[None].expand(zlog.shape[0], -1))
    praw = torch.softmax(zlog, -1)
    nxt = torch.tensor([ids[i + 1] if i + 1 < len(ids) else 0 for _, i in pts], device=zlog.device)
    pieces = [len(tok.decode([x])) for x in gids]
    ends = np.cumsum(pieces)
    store.append(dict(id=pid, closed=bool(r["closed"]), ok=bool(r["ok_boxed"]), level=pr[pid]["level"], end=end, n_prompt=len(pids),
                      labels=[l for l, _ in pts], t=[i - len(pids) for _, i in pts], char_end=[int(ends[i - len(pids)]) for _, i in pts],
                      H=Hsel.numpy(), z22=z.numpy(), core22=core.numpy(),
                      logit_te=zlog[:, TE].cpu().numpy(), p_te=padj[:, TE].cpu().numpy(), top1=zlog.max(-1).values.cpu().numpy(),
                      next_id=(nxt.cpu().numpy()), p_next=padj.gather(1, nxt[:, None]).squeeze(1).cpu().numpy(),
                      p_next_raw=praw.gather(1, nxt[:, None]).squeeze(1).cpu().numpy(), gold=pr[pid]["gold"]))
    n1 = sum(1 for l, _ in pts if l == "c1"); n0 = sum(1 for l, _ in pts if l == "c0")
    print(f"{pid:>4s} closed={int(r['closed'])} end={end:6d} points={len(pts):3d} (c1={n1}, c0={n0}) | mean logitTE c1={np.mean([store[-1]['logit_te'][k] for k, l in enumerate(store[-1]['labels']) if l == 'c1']) if n1 else float('nan'):.2f} c0={np.mean([store[-1]['logit_te'][k] for k, l in enumerate(store[-1]['labels']) if l == 'c0']) if n0 else float('nan'):.2f}", flush=True)
torch.save(store, f"{tf.CACHE_DIR}/probe_{tag}.pt")
print("saved", len(store), "generations")
