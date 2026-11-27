"""Pool the DLA tables of read1 outputs over all sequences of a pass.
usage: python pool.py <prefix> <pass> <out.npz>
Writes, for every DLA position (a line end / indentation / position before a Wait or </think>):
  comp_think [N, 361]  contributions to the </think> logit
  comp_next  [N, 361]  contributions to the logit of the token that actually came next (NaN if it was
                       neither </think> nor among the raw top-6)
  meta [N, 8]: pid index, row (generated-token index of the predicted token), wait count before the row,
               next token id, prev token is newline (0/1), next is waitish (0/1), raw top1 id, raw top1 logit
  think_logit [N], next_logit [N], level [N]
"""
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reads as R  # noqa: E402
from an2 import comp_table  # noqa: E402

tok = AutoTokenizer.from_pretrained(str(R.K.model_dir()))
TOKS = tok.convert_ids_to_tokens(list(range(len(tok))))
IS_NL = np.array(["Ċ" in s for s in TOKS])
IS_WAIT = np.array([s.lstrip("Ġ*Ċ").startswith("Wait") for s in TOKS])


def main():
    prefix, k, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    rows = R.gen_rows(k)
    ids = list(rows)
    CT, CN, META, TL, NL, LV = [], [], [], [], [], []
    for pi, pid in enumerate(ids):
        d = np.load(f"{prefix}.{pid}.npz")
        g = d["gids"].astype(np.int64)
        plen = int(d["plen"])
        C, names = comp_table(d)                       # [361, n, 7]
        pos = d["dla_pos"].astype(np.int64)
        row = pos + 1 - plen
        toks = d["dla_tok"].astype(np.int64)           # [n, 7]
        nxt = g[row]                                   # the token that actually came next
        wc = np.cumsum(IS_WAIT[g]) - IS_WAIT[g]
        hit = toks == nxt[:, None]
        has = hit.any(1)
        cn = np.full((C.shape[1], C.shape[0]), np.nan, dtype=np.float32)     # [n, 361]
        idx = np.where(has)[0]
        cn[idx] = C[:, idx, :][:, np.arange(len(idx)), hit[idx].argmax(1)].T
        CT.append(C[:, :, 0].T.astype(np.float16))
        CN.append(cn.astype(np.float16))
        prev_nl = np.zeros(len(row), dtype=bool)
        prev_nl[row > 0] = IS_NL[g[row[row > 0] - 1]]
        META.append(np.stack([np.full(len(row), pi), row, wc[row], nxt, prev_nl, IS_WAIT[nxt],
                              toks[:, 1], d["dla_top"][:, 0]], 1))
        TL.append(d["dla_total"][:, 0]); NL.append(d["prof_act"][row])
        LV.append(np.full(len(row), d["meta"][3]))
    np.savez(out, comp_think=np.concatenate(CT), comp_next=np.concatenate(CN), meta=np.concatenate(META).astype(np.float32),
             think_logit=np.concatenate(TL), next_logit=np.concatenate(NL), level=np.concatenate(LV),
             names=np.array(names), pids=np.array(ids))
    print("pooled", sum(len(m) for m in META), "DLA positions")


if __name__ == "__main__":
    main()
