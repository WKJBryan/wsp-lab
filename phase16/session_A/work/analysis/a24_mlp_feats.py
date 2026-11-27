"""Record the hidden activations (input of down_proj) of the MLPs of layers 20-23 at line ends of
every generation, at prompt tokens and at response tokens (after </think>), for the design and the
false-fire check of a readout wired into the last MLP.
Usage: python a24_mlp_feats.py <pass no> <tag>"""
import sys, json, re
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1])
tag = sys.argv[2]
LAYERS = [20, 21, 22, 23]
rows = tf.gen_rows(n)
pr = {p["id"]: p for p in tf.practice()}
loops = {d["id"]: d for d in json.load(open(f"{tf.CACHE_DIR}/loops_pass{n}.json"))}
tok = tf.tokenizer()
m = tf.load()
Lm = tf.lm(m)
E = m.get_input_embeddings().weight
TE = tf.T_THINK_END
nl_ids = {tok.encode(s, add_special_tokens=False)[0] for s in ["\n", "\n\n"]}
store = []
for r in rows:
    pid = r["id"]
    gids = tok.encode(r["text"])
    pids = tf.prompt_ids(pr[pid]["q"])
    ids = pids + gids
    te = gids.index(TE) if TE in gids else None
    ls = loops[pid].get("loop_start") if (pid in loops and loops[pid].get("period") and r["finish"] == "length") else None
    end = ls if ls else (te if te is not None else len(gids))
    end = min(end, 20000)
    use = ids[: len(pids) + end] + (ids[len(pids) + te: len(pids) + te + 1500] if te is not None else [])
    ids_t = torch.tensor(use, device=m.device)
    act = {}
    hooks = []
    hs = []

    def pre_layer(module, args, kwargs):
        hs.append((args[0] if args else kwargs["hidden_states"])[0])
    hooks.append(Lm.embed_tokens.register_forward_hook(lambda mod, i, o: hs.append(o[0])))
    for layer in Lm.layers:
        hooks.append(layer.register_forward_pre_hook(pre_layer, with_kwargs=True))
    for l in LAYERS:
        hooks.append(Lm.layers[l].mlp.down_proj.register_forward_pre_hook(
            lambda mod, args, l=l: act.__setitem__(l, args[0][0].detach())))
    with torch.no_grad():
        out = m.model(input_ids=ids_t[None], use_cache=False)
    for h in hooks:
        h.remove()
    T = len(ids_t)
    pieces = [tok.decode([x]) for x in use[len(pids):]]
    ends = np.cumsum([len(p) for p in pieces])
    # line ends in the thinking part
    pts = []
    for i in range(len(pids) + 12, len(pids) + end):
        j = i - len(pids)
        if gids[j] in nl_ids and (te is None or j < te):
            s = "".join(tok.decode([x]) for x in gids[max(0, j - 6): j]).rstrip()
            if s.endswith(".") or s.endswith("$") or s.endswith("}") or s.endswith(")") or s.endswith("]") or s.endswith("?") or s.endswith(":"):
                pts.append(i)
    if te is not None and te >= 1:
        pts.append(len(pids) + te - 1)
    pts = pts[:80]
    resp = list(range(len(pids) + te + 1, min(T, len(pids) + te + 1500), 2)) if te is not None else []
    prompt = list(range(0, len(pids)))
    sel = torch.tensor(prompt + pts + resp, device=m.device)
    hn = out.last_hidden_state[0][sel].float()
    with torch.no_grad():
        zlog = hn @ E.float().T
    store.append(dict(id=pid, closed=bool(r["closed"]), n_prompt=len(pids), n_line=len(pts), n_resp=len(resp), n_prompt_pts=len(prompt),
                      line_t=[i - len(pids) for i in pts], resp_t=[i - len(pids) for i in resp],
                      line_ctx=["".join(tok.decode([x]) for x in gids[max(0, i - len(pids) - 60): i - len(pids)]) for i in pts],
                      act={l: act[l][sel].half().cpu().numpy() for l in LAYERS},
                      resid=hs[-1][sel].float().cpu().numpy(), h_in23=hs[23][sel].float().cpu().numpy(),
                      logit_te=zlog[:, TE].cpu().numpy(), top1=zlog.max(-1).values.cpu().numpy(), top1_id=zlog.argmax(-1).cpu().numpy(),
                      next_id=use[0:0] + [use[i + 1] if i + 1 < len(use) else -1 for i in (prompt + pts + resp)]))
    print(f"{pid:>4s} closed={int(r['closed'])} line ends={len(pts):3d} resp={len(resp):3d} prompt={len(prompt):3d}", flush=True)
torch.save(store, f"{tf.CACHE_DIR}/mlpfeat_{tag}.pt")
print("saved", len(store))
