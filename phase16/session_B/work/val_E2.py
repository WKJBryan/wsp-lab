"""Teacher-forced evaluation of a stop neuron (E) on top of other edits, over the traces of a pass.
usage: python val_E2.py <out.txt> --pass 2 --pre R1 --e E2 [--e-only] [--pthr 0.3] [--maxg 33000] [--ids ...] [--lines]

For every sequence: profile with the `--pre` edits (reference) and with the stop neuron added. Reports where
the first row with protocol P(</think>) >= pthr lies, the line before it, the answer stated in the thinking
up to that point (last "answer is X" line) and whether that answer matches the gold answer.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edits as ED  # noqa: E402
import probe as PR  # noqa: E402
import reads as R  # noqa: E402
from val_edit import make_edit  # noqa: E402

sys.path.insert(0, str(Path.home() / "wsp14/sandbox/harness"))
from rescore import answers_match  # noqa: E402

WORK = Path.home() / "wsp14/work"
ANS = re.compile(r"answer (?:is|should be|:|would be)\s*(.+?)\s*(?:\$?\.\s*$|\.\s*$|$)", re.I | re.M)


def stated_answer(text):
    ms = ANS.findall(text)
    for m in reversed(ms[-5:]):
        a = m.strip().strip("$").rstrip(".").strip()
        if a and len(a) < 80:
            return a
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--pass", dest="kp", type=int, default=2)
    ap.add_argument("--pre", nargs="*", default=[])
    ap.add_argument("--e", default="")
    ap.add_argument("--pthr", type=float, default=0.3)
    ap.add_argument("--maxg", type=int, default=33000)
    ap.add_argument("--ids", default="")
    ap.add_argument("--lines", action="store_true")
    a = ap.parse_args()
    k = a.kp
    rows = R.gen_rows(k)
    meta = {p["id"]: p for p in R.K.PRACTICE}
    rd = R.Reader("model", device="cuda")
    tok = rd.tok
    ids = [w for w in a.ids.split(",") if w] or list(rows)
    out = open(a.out, "w", encoding="utf-8")

    def pr(*x):
        out.write(" ".join(str(v) for v in x) + "\n")

    res = {}
    base_sd = {kk: v.clone() for kk, v in rd.tm.state_dict().items()}
    for tag, edits in (("ref", list(a.pre)), ("E", list(a.pre) + ([f"G:{a.e}"] if a.e else []))):
        rd.tm.load_state_dict(base_sd)                      # start every configuration from the base weights
        A = ED.LiveAccess(rd)
        for e in edits:
            info = make_edit(e)(A)
            if tag == "E" and e.startswith("G:"):
                pr("edit:", {kk: v for kk, v in info.items() if kk != "old"})
        seqs = {}
        cache = WORK / f"cache/ref_{k}_{'_'.join(a.pre) or 'base'}_{a.maxg}"
        cache.mkdir(parents=True, exist_ok=True)
        for pid in ids:
            r = rows[pid]
            cfile = cache / f"{pid}.npz"
            if tag == "ref" and cfile.exists():
                z = np.load(cfile)
                seqs[pid] = {kk: z[kk] for kk in z.files}
                continue
            ptxt = R.K.prompt_text(meta[pid]["q"], tok)
            pids_, gids = R.encode_gen(tok, ptxt, r["text"])
            gids = gids[: a.maxg]
            ids_all = torch.tensor([pids_ + gids])
            h, _ = rd.run(ids_all, capture=False)
            prof = rd.profile(h, gids, len(pids_))
            seqs[pid] = dict(tp=prof["think_p"].numpy(), ptop=prof["p_top_id"].numpy(), pk=prof["pk"].numpy(),
                             pact=prof["p_act"].numpy(), gids=np.array(gids), plen=len(pids_),
                             top1=prof["top"][:, 0].numpy())
            if tag == "ref":
                np.savez(cfile, **seqs[pid])
        res[tag] = seqs
    pr("\npid | cl ok lvl | actual closed at | simulated first close (E): row, P, top token | answer stated by then -> match | "
       "same for reference (no E) | ref P(actual) first 1500 -> E")
    stats = dict(cut=[0, 0, 0], closed_ok=[0, 0, 0], closed_wrong=[0, 0, 0], nonle=[], ansmax=[], prem=[])

    def cat(pid):
        r = rows[pid]
        return "closed_ok" if (r["closed"] and r["ok_boxed"]) else ("closed_wrong" if r["closed"] else "cut")

    for pid in ids:
        r = rows[pid]
        m = meta[pid]
        c = cat(pid)
        S, T = res["ref"][pid], res["E"][pid]
        text_all = "".join(PR.STRS[i] for i in S["gids"])

        def first_close(X):
            f = np.where(X["tp"] >= a.pthr)[0]
            return int(f[0]) if len(f) else None

        fE, fR = first_close(T), first_close(S)
        lines_txt = []
        for f in (fE, fR):
            if f is None:
                lines_txt.append((None, None, None))
            else:
                before = text_all[:sum(len(PR.STRS[i]) for i in S["gids"][:f])]
                ls = [l for l in before.split("\n") if l.strip()]
                ans = stated_answer(before)
                okm = bool(ans is not None and answers_match(m["gold"], ans))
                lines_txt.append((f, (ls[-1].strip()[-70:] if ls else ""), (ans, okm)))
        cls, region = PR.label_positions(np.array(S["gids"]), 0)
        nl = np.array(["\n" in PR.STRS[i] for i in S["gids"]])
        nonle = (~nl) & (region == 1)
        stats["nonle"].append(float(np.abs(T["tp"] - S["tp"])[nonle].max()))
        ansm = region == 2
        stats["ansmax"].append(float(T["tp"][ansm].max()) if ansm.any() else 0.0)
        fE, lE, aE = lines_txt[0]
        fR, lR, aR = lines_txt[1]
        stats[c][0] += 1
        if fE is not None:
            stats[c][1] += 1
            stats[c][2] += int(bool(aE[1]))
        pr(f"{pid:4s} | {int(r['closed'])} {int(r['ok_boxed'])} L{m['level'][-1]} | {r['n_tokens'] if r['closed'] else '-':>6} | "
           f"{str(fE):>6s} {T['tp'][fE] if fE is not None else 0:.2f} {PR.STRS[int(T['ptop'][fE])] if fE is not None else ''!r:8s} | "
           f"{(aE[0] if aE else None)!r:>28s} -> {int(bool(aE[1])) if aE else '-'} | {str(fR):>6s} | "
           f"{S['pact'][:1500].mean():.3f} -> {T['pact'][:1500].mean():.3f}")
        if a.lines and fE is not None:
            pr(f"       E first close after line: {lE!r}")
    pr(f"\nthreshold P(</think>) >= {a.pthr}. Groups: [n, n with a simulated close, n whose stated answer at that point matches gold]")
    for kk, v in stats.items():
        if isinstance(v, list) and len(v) == 3:
            pr(f"   {kk:13s}: {v}")
    pr(f"non-line-end thinking positions: max |P_E - P_ref| = {max(stats['nonle']):.4f}; answer phase: max P_E(</think>) = {max(stats['ansmax']):.4f}")
    out.close()


if __name__ == "__main__":
    main()
