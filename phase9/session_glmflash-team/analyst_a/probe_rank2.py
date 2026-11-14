import numpy as np
for tag, loopstart in [("gsm4131_sampled", 906), ("gsm1398_sampled", 768), ("gsm5570_sampled", 355)]:
    z = np.load(f"/home/bladless/wsp4/sandbox/analyst_a/gen_{tag}_thinkrank.npz")
    r, l = z["ranks"], z["logits"]
    loop = slice(loopstart, None)
    rr, ll = r[loop], l[loop]
    print(tag, "loopsteps", len(rr))
    print("  in-loop logit_</think>: p50", round(float(np.percentile(ll, 50)), 2),
          "p90", round(float(np.percentile(ll, 90)), 2),
          "p99", round(float(np.percentile(ll, 99)), 2), "max", round(float(ll.max()), 2))
    print("  in-loop rank<20 frac:", round(float((rr < 20).mean()), 3),
          "rank<5 frac:", round(float((rr < 5).mean()), 3))
    for d in (1.3, 1.5, 1.8):
        scaled = ll * d
        print(f"  dose {d}: frac scaled-logit>10:", round(float((scaled > 10).mean()), 3),
              ">15:", round(float((scaled > 15).mean()), 3), ">20:", round(float((scaled > 20).mean()), 3))
