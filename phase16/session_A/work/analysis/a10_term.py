"""Teacher-forced read of a whole pass: for every generated position, the model's own
probability of </think> under the evaluation sampling rules (temperature 0.6, top_p 0.95,
top_k 20, presence penalty 1.5), plus the probability of the token it actually wrote.
Usage: python a10_term.py <pass no> [--limit K] [--skip-file file]
Writes work/cache/a10_pass<k>_<row id>.npz per row."""
import sys, json, os, time
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

args = sys.argv[1:]
pass_no = int(args[0])
limit = int(args[args.index("--limit") + 1]) if "--limit" in args else 10 ** 9
CACHE = "/home/user/wsp14/work/cache"
os.makedirs(CACHE, exist_ok=True)

rows = tf.gen_rows(pass_no)
pr = {p["id"]: p for p in tf.practice()}
m = tf.load()
tok = tf.tokenizer()
E = m.get_input_embeddings().weight            # [V, D]
V = E.shape[0]
TEMP, TOP_P, TOP_K, PEN, PPL = 0.6, 0.95, 20, 1.5, 10   # vLLM penalises the last 10 prompt tokens too

def sample_prob_adjusted(z, seen):
    """z: [N, V] raw logits (float32), seen: [N, V] bool. Returns (p_think [N], p_top1 [N], top1_id [N],
    ent [N], adj [N, V] not returned). Follows vLLM: penalties, temperature, top_k, top_p."""
    z = z - PEN * seen.float()
    z = z / TEMP
    topv, topi = z.topk(TOP_K, dim=-1)                       # [N, K]
    p = torch.softmax(topv, dim=-1)
    srt, order = p.sort(dim=-1, descending=True, stable=True)
    cum = srt.cumsum(-1)
    keep = (cum - srt) < TOP_P                                # keep until the mass reaches TOP_P (inclusive)
    srt = srt * keep
    srt = srt / srt.sum(-1, keepdim=True)
    is_think = topi.gather(1, order) == tf.T_THINK_END
    p_think = (srt * is_think).sum(-1)
    top1 = topi.gather(1, order[:, :1]).squeeze(1)
    ptop1 = srt[:, 0]
    pr = torch.softmax(z, dim=-1)
    ent = -(pr * (pr + 1e-12).log()).sum(-1)
    return p_think, ptop1, top1, ent, pr

done = 0
t_all = time.time()
summary = []
for r in rows:
    if done >= limit:
        break
    outp = f"{CACHE}/a10_pass{pass_no}_{r['id']}.npz"
    if os.path.exists(outp):
        done += 1
        continue
    q = pr[r["id"]]["q"]
    pids = tf.prompt_ids(q)
    gids = tok.encode(r["text"])
    t0 = time.time()
    ids = torch.tensor(pids + gids, device="cuda")
    T0 = len(pids)
    with torch.no_grad():
        out = m.model(input_ids=ids[None], use_cache=False)
    hn = out.last_hidden_state[0]                      # [T, D]
    G = len(gids)
    res = dict(p_think=np.zeros(G, np.float32), p_top1=np.zeros(G, np.float32), top1=np.zeros(G, np.int64),
               ent=np.zeros(G, np.float32), p_actual=np.zeros(G, np.float32), p_actual_adj=np.zeros(G, np.float32),
               l_think=np.zeros(G, np.float32), l_end=np.zeros(G, np.float32), l_imend=np.zeros(G, np.float32))
    seen_vec = torch.zeros(V, dtype=torch.bool, device="cuda")
    seen_vec[torch.tensor(pids[-PPL:], device="cuda")] = True
    CH = 256
    with torch.no_grad():
        for a in range(0, G, CH):
            b = min(G, a + CH)
            h = hn[T0 - 1 + a: T0 - 1 + b]
            z = (h.float() @ E.float().T)              # [n, V]
            n = b - a
            mask = torch.zeros(n, V, dtype=torch.bool, device="cuda")
            # token gids[g-1] is 'seen' when predicting gids[g]
            prev = torch.tensor([gids[a - 1 + j] if a - 1 + j >= 0 else -1 for j in range(n)], device="cuda")
            valid = prev >= 0
            mask[valid.nonzero().squeeze(1), prev[valid]] = True
            seen = (mask.cumsum(0) > 0) | seen_vec[None]
            p_think, ptop1, top1, ent, p_all = sample_prob_adjusted(z, seen)
            act = torch.tensor(gids[a:b], device="cuda")
            lse = z.logsumexp(-1)
            res["p_think"][a:b] = p_think.cpu().numpy()
            res["p_top1"][a:b] = ptop1.cpu().numpy()
            res["top1"][a:b] = top1.cpu().numpy()
            res["ent"][a:b] = ent.cpu().numpy()
            res["p_actual"][a:b] = p_all.gather(1, act[:, None]).squeeze(1).cpu().numpy()
            res["p_actual_adj"][a:b] = (torch.softmax(z - PEN * seen.float(), -1)).gather(1, act[:, None]).squeeze(1).cpu().numpy()
            res["l_think"][a:b] = (z[:, tf.T_THINK_END] - lse).cpu().numpy()
            res["l_end"][a:b] = (z[:, tf.T_IM_END] - lse).cpu().numpy()
            seen_vec |= seen[-1]
    res["n_prompt"] = np.int32(T0)
    res["n_tokens_jsonl"] = np.int32(r["n_tokens"])
    np.savez(outp, **res)
    tok_ok = (len(gids) == r["n_tokens"])
    pt = res["p_think"]
    summary.append(dict(id=r["id"], G=G, match=tok_ok, closed=bool(r["closed"]), ok=bool(r["ok_boxed"]), finish=r["finish"],
                        p_think_last=float(pt[-50:].mean()), p_think_max=float(pt.max()),
                        first_gt_50=int(np.argmax(pt > 0.5)) if (pt > 0.5).any() else -1,
                        n_gt_10=int((pt > 0.10).sum()), n_gt_50=int((pt > 0.5).sum()),
                        p_actual_mean=float(res["p_actual"].mean()), sec=round(time.time() - t0, 1)))
    s = summary[-1]
    print(f"{r['id']:>4s} G={G:6d} match={tok_ok} closed={s['closed']} ok={s['ok']} fin={s['finish']:6s} "
          f"pTE_last50={s['p_think_last']:.3f} max={s['p_think_max']:.3f} first>0.5@{s['first_gt_50']:6d} n>0.1={s['n_gt_10']:5d} n>0.5={s['n_gt_50']:4d} "
          f"p_act={s['p_actual_mean']:.3f} {s['sec']}s", flush=True)
    done += 1
json.dump(summary, open(f"{CACHE}/a10_pass{pass_no}_summary.json", "w"), indent=0)
print("total", round(time.time() - t_all, 1), "s")
