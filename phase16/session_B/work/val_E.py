"""Teacher-forced validation of a grafted stop neuron (E) against the base model's profiles.
usage: python val_E.py <params name> <out.txt> [--ids ...] [--maxg 8000] [--pass 1] [--extra R1 ...] [--lines]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edits as ED  # noqa: E402
import probe as PR  # noqa: E402
import reads as R  # noqa: E402
from val_edit import make_edit  # noqa: E402

WORK = Path.home() / "wsp14/work"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("out")
    ap.add_argument("--ids", default="")
    ap.add_argument("--maxg", type=int, default=8000)
    ap.add_argument("--pass", type=int, default=1)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--lines", action="store_true")
    ap.add_argument("--basepass", type=int, default=1, help="pass whose profiles are the base reference")
    a = ap.parse_args()
    k = a.pass
    rows = R.gen_rows(k)
    meta = {p["id"]: p for p in R.K.PRACTICE}
    rd = R.Reader("model", device="cuda")
    A = ED.LiveAccess(rd)
    for e in [f"G:{a.name}"] + list(a.extra):
        print("edit:", make_edit(e)(A))
    tok = rd.tok
    ids = [w for w in a.ids.split(",") if w] or list(rows)
    out = open(a.out, "w", encoding="utf-8")

    def pr(*x):
        s = " ".join(str(v) for v in x)
        out.write(s + "\n")

    pr("pid | cl ok | base sumP maxP | edited sumP maxP@row | first row P>=0.2: row P top1 | commit ends P base->ed | "
       "non-line-end max |d| | answer-phase maxP | last prompt pos maxP(</think>) | first1500 P(actual) base->ed")
    agg = dict(sum_base=[], sum_ed=[], has=[], first_rows=[], ans_max=[], prompt_max=[], commit_b=[], commit_e=[], other_e=[])
    for pid in ids:
        r = rows[pid]
        ptxt = R.K.prompt_text(meta[pid]["q"], tok)
        pids_, gids = R.encode_gen(tok, ptxt, r["text"])
        gids = gids[: a.maxg]
        plen = len(pids_)
        arr = np.array(pids_ + gids)
        G = len(gids)
        ids_all = torch.tensor([arr])
        h, _ = rd.run(ids_all, capture=False)
        prof = rd.profile(h, gids, plen)
        base = np.load(WORK / f"r1/full.{pid}.npz")
        bt, btp = base["prof_think"][:G], base["prof_think_p"][:G]
        tp = prof["think_p"].numpy(); tl = prof["think"].numpy()
        cls, region = PR.label_positions(arr, plen)
        nl = np.array(["\n" in PR.STRS[i] for i in arr])
        gen = slice(plen, plen + G)
        commit = nl[gen] & (cls[gen] == 1) & (region[gen] == 1)
        other_le = nl[gen] & ~commit & (region[gen] == 1)
        nonle = (~nl[gen]) & (region[gen] == 1)
        ans_rows = region[gen] == 2
        first = np.where(tp >= 0.2)[0]
        fr = int(first[0]) if len(first) else None
        # last prompt position
        with torch.no_grad():
            lg0 = rd.logits(h, torch.tensor([plen - 1], device="cuda"))[0]
            pi, pr_ = rd.sample_probs(lg0[None, :])
            j = (pi[0] == R.THINK_END).nonzero()
            prompt_p = float(pr_[0][j[0, 0]]) if j.shape[0] else 0.0
        cb = float(np.mean(btp[commit])) if commit.any() else float("nan")
        ce = float(np.mean(tp[commit])) if commit.any() else float("nan")
        pact = prof["p_act"].numpy(); bpact = base["prof_p_act"][:G]
        n = min(G, 1500)
        top1 = PR.STRS[int(prof["p_top_id"][fr])] if fr is not None else ""
        am = float(tp[ans_rows].max()) if ans_rows.any() else 0.0
        pr(f"{pid:4s} | {int(r['closed'])} {int(r['ok_boxed'])} | {btp.sum():7.2f} {btp.max():6.3f} | {tp.sum():7.2f} {tp.max():6.3f}@{int(tp.argmax()):5d} | "
           f"{str(fr):>5s} {tp[fr] if fr is not None else 0:.3f} {top1!r:12s} | {cb:.3f}->{ce:.3f} | "
           f"{np.abs(tl - bt)[nonle].max() if nonle.any() else 0:.2f} | {am:.4f} | {prompt_p:.4f} | {bpact[:n].mean():.3f}->{pact[:n].mean():.3f}")
        if a.lines and fr is not None:
            txt = "".join(PR.STRS[i] for i in arr[:plen + fr])
            ls = [l for l in txt.split("\n") if l.strip()]
            pr(f"       first row with P>=0.2 at generated token {fr}: after line {ls[-1].strip()[-110:]!r}; top after penalty: " +
               " ".join(f"{PR.STRS[i]!r}" for i in prof["pk"][fr][:4]))
        agg["sum_base"].append(btp.sum()); agg["sum_ed"].append(tp.sum()); agg["has"].append(fr is not None)
        agg["first_rows"].append(fr); agg["ans_max"].append(am); agg["prompt_max"].append(prompt_p)
        agg["commit_b"].append(cb); agg["commit_e"].append(ce)
        agg["other_e"].append(float(tp[other_le].mean()) if other_le.any() else 0.0)
    fr = [x for x in agg["first_rows"] if x is not None]
    pr(f"\nsequences with a row of P(</think>) >= 0.2 within {a.maxg} tokens: {sum(agg['has'])}/{len(ids)}; "
       f"sum of P(</think>) along the trace: base mean {np.mean(agg['sum_base']):.3f}, edited mean {np.mean(agg['sum_ed']):.2f} "
       f"(median {np.median(agg['sum_ed']):.2f}); first such row (tokens): {sorted(fr)}")
    pr(f"commit line ends: P(</think>) base {np.nanmean(agg['commit_b']):.4f} -> edited {np.nanmean(agg['commit_e']):.3f}; "
       f"other line ends edited {np.mean(agg['other_e']):.4f}; answer-phase max P {max(agg['ans_max']):.4f}; "
       f"last prompt position max P {max(agg['prompt_max']):.5f}")
    out.close()


if __name__ == "__main__":
    main()
