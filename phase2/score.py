"""Phase 2 reveal: score committed guesses against the sealed manifest.

Usage: uv run python phase2/score.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

manifest = json.loads((HERE / "sealed" / "manifest.json").read_text())
guesses = json.loads((HERE / "editor" / "guesses.json").read_text())

truth = {n: d["task"] for n, d in manifest["assignment"].items()}
print("ground truth:", truth)
print("editor guess:", guesses["assignment"])

correct = sum(1 for n in "ABCDE" if guesses["assignment"][n] == truth[n])
print(f"\nP1 assignment: {correct}/5 correct  (PASS >=4, STRONG =5; chance ~0.8%)")

info = manifest["task_info"]
for task in ["blobs", "interact"]:
    guessed_net = next(n for n, t in guesses["assignment"].items() if t == task)
    ok = guesses["assignment"][guessed_net] == truth[guessed_net]
    if ok:
        guess_feats = sorted(guesses["informative_features"][task])
        true_feats = sorted(info[task]["informative"])
        hit = guess_feats == true_feats
        print(f"P2 {task} (net {guessed_net}): guessed {guess_feats}, truth {true_feats} -> {'EXACT' if hit else 'MISS'}")
    else:
        print(f"P2 {task}: assignment itself wrong (guessed net {guessed_net} is {truth[guessed_net]}) -> MISS")

print("\ntask training info (revealed):")
for k, v in info.items():
    print(f"  {k:8s} final loss {v['loss']:.3f} | train-set acc {v['acc']*100:5.1f}% | informative {v['informative']}")
