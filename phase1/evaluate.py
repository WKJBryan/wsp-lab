"""Evaluate a checkpoint: mean return over N episodes, greedy (argmax) policy.

Seed ranges are disjoint by purpose (PROTOCOL.md / ISSUES.md #3):
  800000+  editor budget evaluations (advance by batch size between batches)
  900000+  primary endpoint (900000-900255, never used during editing)
  910000+  control evaluations

Usage: uv run python phase1/evaluate.py <ckpt.pt> [N_EPISODES=64] [SEED_BASE=800000]
"""

import sys
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

from policy import make_policy


def evaluate(ckpt_path: str, n_episodes: int = 64, seed_base: int = 800000) -> np.ndarray:
    policy = make_policy()
    policy.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    policy.eval()
    env = gym.make("CartPole-v1")
    returns = []
    with torch.no_grad():
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed_base + ep)
            total, done = 0.0, False
            while not done:
                logits = policy(torch.tensor(obs, dtype=torch.float32))
                action = int(torch.argmax(logits))  # greedy
                obs, r, term, trunc, _ = env.step(action)
                total += float(r)
                done = term or trunc
            returns.append(total)
    env.close()
    return np.array(returns)


if __name__ == "__main__":
    ckpt = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    seed_base = int(sys.argv[3]) if len(sys.argv) > 3 else 800000
    arr = evaluate(ckpt, n, seed_base)
    print(
        f"{Path(ckpt).name}: episodes={n} seed_base={seed_base} "
        f"mean={arr.mean():.1f} std={arr.std():.1f} min={arr.min():.0f} max={arr.max():.0f}"
    )
