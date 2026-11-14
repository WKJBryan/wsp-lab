import numpy as np
tag = "gsm5570_sampled"
z = np.load(f"/home/bladless/wsp4/sandbox/analyst_a/gen_{tag}_thinkrank.npz")
r, l = z["ranks"], z["logits"]
print(tag, "steps", len(r))
for name, sl in [("steps<300 (pre-loop)", slice(0, 300)), ("loop >=400", slice(400, None))]:
    rr, ll = r[sl], l[sl]
    print(" ", name,
          "| rank: min", int(rr.min()), "p10", int(np.percentile(rr, 10)),
          "median", int(np.median(rr)), "p90", int(np.percentile(rr, 90)),
          "| logit_</think>: p10", round(float(np.percentile(ll, 10)), 2),
          "median", round(float(np.median(ll)), 2), "max", round(float(ll.max()), 2))
    print("   frac rank<20:", round(float((rr < 20).mean()), 3), " frac rank<5:", round(float((rr < 5).mean()), 3),
          " frac logit>0:", round(float((ll > 0).mean()), 3))
