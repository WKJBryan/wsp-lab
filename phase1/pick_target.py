"""Greedy-evaluate every checkpoint in phase1/ckpt; lowest mean becomes the TARGET.

Per PROTOCOL.md: target is eligible only if greedy mean < 100 over 64 episodes.
Selection seed range 850000+ — disjoint from training (seed*1000+ep), editor
budget evals (800000+), primary endpoint (900000-900255), and controls (910000+).

Usage: uv run python phase1/select.py
"""

from pathlib import Path

from evaluate import evaluate

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "ckpt"
SELECT_SEED_BASE = 850000


def main():
    rows = []
    for i in range(20):
        path = CKPT_DIR / f"seed{i}.pt"
        arr = evaluate(str(path), 64, SELECT_SEED_BASE)
        rows.append((i, float(arr.mean()), float(arr.std()), int(arr.min()), int(arr.max())))
        print(
            f"seed {i:2d}: greedy mean {rows[-1][1]:6.1f}  std {rows[-1][2]:5.1f}  "
            f"min {rows[-1][3]:3d}  max {rows[-1][4]:3d}",
            flush=True,
        )

    eligible = [r for r in rows if r[1] < 100.0]
    if not eligible:
        print("\nNO ELIGIBLE TARGET (all >= 100): PROTOCOL says train 20 more seeds.")
        return
    target = min(eligible, key=lambda r: r[1])
    print(f"\nTARGET = seed {target[0]} (greedy mean {target[1]:.1f} on selection seeds)")


if __name__ == "__main__":
    main()
