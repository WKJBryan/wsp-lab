"""Answer-phase rows with high own </think> evidence, per trace, split by clean / leaky answer phase. CPU.
usage: python ans_hi2.py <pass> <prefix>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reads as R  # noqa: E402

WORK = Path.home() / "wsp14/work"


def main():
    k, pre = int(sys.argv[1]), sys.argv[2]
    rows = R.gen_rows(k)
    out = []
    for pid, r in rows.items():
        if not r["closed"]:
            continue
        d = np.load(WORK / f"r3/{pre}.{pid}.npz")
        think, top, region, pos = d["think"], d["top"], d["region"], d["pos"]
        ans = region == 2
        if not ans.any():
            continue
        t, tp = think[ans], top[ans]
        clean = r["finish"] == "stop"
        out.append((pid, clean, int(r("ok_boxed")) if False else int(r["ok_boxed"]), int((t >= 20.5).sum()), int((t >= 22).sum()),
                    float(t.max()), int(((t >= 22) & (tp - t < 8)).sum()), int(((t >= 21.5) & (tp - t < 6)).sum())))
    print("pid clean ok | answer rows >=20.5, >=22, max | rows with excess>=1.5 and gap<8 | rows with excess>=1.0 and gap<6")
    for o in out:
        if o[3]:
            print(f"{o[0]:5s} {int(o[1])} {o[2]} | {o[3]:4d} {o[4]:4d} {o[5]:6.2f} | {o[6]:4d} | {o[7]:4d}")
    c = np.array([[o[1], o[3], o[4], o[6], o[7]] for o in out])
    for v, nm in ((1, "clean"), (0, "leaky")):
        m = c[:, 0] == v
        print(f"{nm} answer phases: n={int(m.sum())}; with any row >= 20.5: {int((c[m, 1] > 0).sum())}, >= 22: {int((c[m, 2] > 0).sum())}, "
              f"excess>=1.5 & gap<8: {int((c[m, 3] > 0).sum())}, excess>=1.0 & gap<6: {int((c[m, 4] > 0).sum())}")


if __name__ == "__main__":
    main()
