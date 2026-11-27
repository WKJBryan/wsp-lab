"""Analysis of read1 outputs.
usage: python an1.py <prefix> [--ids ...] [--trace PID] [--minp 0.005] [--all]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reads as R  # noqa: E402

tok = AutoTokenizer.from_pretrained(str(R.K.model_dir()))
TOKS = tok.convert_ids_to_tokens(list(range(len(tok))))
WAIT = np.array([i for i, s in enumerate(TOKS) if s.lstrip("Ġ*Ċ").startswith("Wait")], dtype=np.int64)
IS_NL = np.array(["Ċ" in s for s in TOKS])
IS_WAIT = np.zeros(len(TOKS), dtype=bool)
IS_WAIT[WAIT] = True


def dec(i):
    return tok.decode([int(i)])


def load(prefix, pid):
    d = np.load(f"{prefix}.{pid}.npz")
    return {k: d[k] for k in d.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--ids", default="")
    ap.add_argument("--trace", default="")
    ap.add_argument("--minp", type=float, default=0.005)
    a = ap.parse_args()
    rows = R.gen_rows(1)
    ids = [w for w in a.ids.split(",") if w] or list(rows)
    print("pid  |cl|ok|lv| G    | #Wait | P(Wait)@Wait med | dec.rows | think_p@dec med | >1% | >5% | max think_p (row) | "
          "argmax==</think> rows")
    summ = []
    for pid in ids:
        d = load(a.prefix, pid)
        g = d["gids"].astype(np.int64)
        G = len(g)
        cl, ok, ntok, lvl = d["meta"]
        isw = IS_WAIT[g]
        p_act, think_p, trank, p_top_id = d["prof_p_act"], d["prof_think_p"], d["prof_think_rank"], d["prof_p_top_id"]
        nl = np.zeros(G, dtype=bool)
        nl[1:] = IS_NL[g[:-1]]                      # decision rows: the token before was a newline
        pw = p_act[isw] if isw.any() else np.array([0.0])
        tp = think_p[nl]
        am = (nl & (p_top_id == R.THINK_END)).sum()
        print(f"{pid:4s} | {int(cl)} | {int(ok)} | {int(lvl)} | {G:5d} | {isw.sum():5d} | {np.median(pw):5.3f} | {nl.sum():5d} | "
              f"{np.median(tp):8.5f} | {(tp > 0.01).sum():3d} | {(tp > 0.05).sum():3d} | {think_p.max():.3f} ({int(think_p.argmax())}) | {am}")
        summ.append((int(cl), think_p.max(), (tp > 0.01).sum()))
    summ = np.array(summ)
    print("mean max think_p:", summ[:, 1].mean(), "closed:", summ[:, 0].sum())
    if a.trace:
        pid = a.trace
        d = load(a.prefix, pid)
        g = d["gids"].astype(np.int64)
        G = len(g)
        text = [dec(x) for x in g]
        nl = np.zeros(G, dtype=bool)
        nl[1:] = IS_NL[g[:-1]]
        wc = np.cumsum(IS_WAIT[g]) - IS_WAIT[g]
        print(f"\n=== trace {pid}: decision rows (after a newline) with think_p >= {a.minp} ===")
        for t in np.where(nl & (d["prof_think_p"] >= a.minp))[0][:80]:
            ctx = "".join(text[max(0, t - 45):t]).replace("\n", "⏎")
            pk = d["prof_pk"][t]
            top = " ".join(f"{dec(i)!r}" for i in pk[:4])
            print(f"t={t:5d} waits={wc[t]:3d} think_p={d['prof_think_p'][t]:.3f} rank={d['prof_think_rank'][t]:2d} "
                  f"think_logit={d['prof_think'][t]:6.2f} act={text[t]!r} p_act={d['prof_p_act'][t]:.3f} | ...{ctx[-60:]} | top {top}")
        print(f"\n=== decision rows: think logit vs top-1 logit, every 25th ===")
        dec_rows = np.where(nl)[0]
        for t in dec_rows[::25][:70]:
            ctx = "".join(text[max(0, t - 40):t]).replace("\n", "⏎")
            tv = d["prof_top"][t]
            print(f"t={t:5d} waits={wc[t]:3d} think_logit={d['prof_think'][t]:6.2f} rawtop={dec(tv[0])!r}:{d['prof_topl'][t][0]:5.2f} "
                  f"act={text[t]!r}:{d['prof_act'][t]:5.2f} | ...{ctx[-55:]}")


if __name__ == "__main__":
    main()
