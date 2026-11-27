"""DLA analysis of read1 outputs: who writes the </think> logit at decision points?
usage: python an2.py <prefix> <pass> [--ids ...] [--top 15]
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
IS_NL = np.array(["Ċ" in s for s in TOKS])


def load(prefix, pid):
    d = np.load(f"{prefix}.{pid}.npz")
    return {k: d[k] for k in d.files}


def comp_table(d):
    """[n_comp, n_pos, 7] contributions and component names."""
    n = d["dla_pos"].shape[0]
    parts, names = [], []
    parts.append(d["dla_emb"][None].astype(np.float32)); names.append("emb")
    A = d["dla_attn"].astype(np.float32)                   # [6, 8, n, 7]
    for li, layer in enumerate(R.K.FULL_LAYERS):
        for h in range(8):
            parts.append(A[li, h][None]); names.append(f"attn.L{layer}.h{h}")
    Gd = d["dla_gdn"].astype(np.float32)                   # [18, 16, n, 7]
    for li, layer in enumerate(R.K.GDN_LAYERS):
        for h in range(16):
            parts.append(Gd[li, h][None]); names.append(f"gdn.L{layer}.h{h}")
    M = d["dla_mlp"].astype(np.float32)                    # [24, n, 7]
    for layer in range(24):
        parts.append(M[layer][None]); names.append(f"mlp.L{layer}")
    return np.concatenate(parts, 0), names                  # [361, n, 7]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("k", type=int)
    ap.add_argument("--ids", default="")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    rows = R.gen_rows(a.k)
    ids = [w for w in a.ids.split(",") if w] or list(rows)
    agg_close, agg_other = [], []
    for pid in ids:
        d = load(a.prefix, pid)
        g = d["gids"].astype(np.int64)
        plen = int(d["plen"])
        C, names = comp_table(d)
        pos = d["dla_pos"].astype(np.int64)
        row = pos + 1 - plen                                   # profile row of the token predicted at pos
        ok_row = (row >= 0) & (row < len(g))
        C, row = C[:, ok_row, :], row[ok_row]
        # decision rows among DLA positions: previous token is a newline
        prev_nl = np.zeros(len(row), dtype=bool)
        prev_nl[row > 0] = IS_NL[g[row[row > 0] - 1]]
        think = C[:, :, 0]                                     # [361, n] contributions to </think>
        total = think.sum(0)
        close_rows = np.where(g == R.THINK_END)[0]
        cl = len(close_rows) > 0
        print(f"\n===== {pid} closed={cl} ok={d['meta'][1]} G={len(g)} DLA n={len(row)} decision rows={prev_nl.sum()} "
              f"| max </think> logit over decision rows {total[prev_nl].max():.2f} at row {row[prev_nl][total[prev_nl].argmax()]}")
        chk = total[prev_nl][:10]
        ref = d["prof_think"][row[prev_nl][:10]]
        print("  DLA sum vs true </think> logit:", np.round(chk, 2), np.round(ref, 2))
        if cl:
            j = int(np.where(row == close_rows[0])[0][0])
            oth = prev_ml = prev_nl & (row != close_rows[0])
            med = np.median(think[:, oth], 1)
            val = think[:, j]
            order = np.argsort(-(val - med))[: a.top]
            print(f"  closing row {row[j]}: total {val.sum():.2f}; other-decision median of total {np.median(total[oth]):.2f}")
            print("  components whose </think> contribution is most above their usual value at the closing point:")
            for i in order:
                print(f"    {names[i]:14s} closing {val[i]:7.2f}   usual med {med[i]:7.2f}   diff {val[i] - med[i]:7.2f}")
            order = np.argsort(-val)[: a.top]
            print("  largest positive contributions to </think> at the closing point:")
            for i in order:
                print(f"    {names[i]:14s} closing {val[i]:7.2f}   usual med {med[i]:7.2f}")
            order = np.argsort(val)[: a.top]
            print("  largest negative contributions to </think> at the closing point:")
            for i in order:
                print(f"    {names[i]:14s} closing {val[i]:7.2f}   usual med {med[i]:7.2f}")
            agg_close.append(val)
            agg_other.append(med)
    if agg_close:
        Ac = np.array(agg_close); Ao = np.array(agg_other)
        diff = (Ac - Ao).mean(0)
        order = np.argsort(-diff)[: a.top]
        print(f"\n===== averaged over {len(agg_close)} closed traces: components most above usual at the closing point")
        for i in order:
            print(f"    {names[i]:14s} closing mean {Ac[:, i].mean():7.2f}   usual mean {Ao[:, i].mean():7.2f}   diff {diff[i]:7.2f}   per-trace diffs {np.round(Ac[:, i] - Ao[:, i], 1)}")
        order = np.argsort(diff)[: a.top]
        print("  components most below usual at the closing point:")
        for i in order:
            print(f"    {names[i]:14s} closing mean {Ac[:, i].mean():7.2f}   usual mean {Ao[:, i].mean():7.2f}   diff {diff[i]:7.2f}   per-trace diffs {np.round(Ac[:, i] - Ao[:, i], 1)}")


if __name__ == "__main__":
    main()
