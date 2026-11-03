"""Train N REINFORCE CartPole-v1 policies from different seeds.

Some seeds converge to poor local optima — that is the point (PROTOCOL.md: the
target failure must be organic). Saves phase1/ckpt/seed{i}.pt and per-seed
summary stats to phase1/results/scores.csv. Per-episode curves are deliberately
NOT saved anywhere (editor blinding).

Usage: uv run python phase1/train.py [N_SEEDS]
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

from policy import make_policy

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
EPISODES = 1500
LR = 3e-3
GAMMA = 0.99

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "ckpt"
RESULTS_DIR = HERE / "results"


def train_one(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make("CartPole-v1")
    policy = make_policy()
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    recent = []

    for ep in range(EPISODES):
        obs, _ = env.reset(seed=seed * 1000 + ep)  # training seed range
        log_probs, rews = [], []
        done = False
        while not done:
            logits = policy(torch.tensor(obs, dtype=torch.float32))
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            obs, r, term, trunc, _ = env.step(action.item())
            log_probs.append(dist.log_prob(action))
            rews.append(float(r))
            done = term or trunc
        recent.append(sum(rews))

        # discounted returns, normalized (vanilla REINFORCE with baseline normalization)
        running, disc = 0.0, []
        for r in reversed(rews):
            running = r + GAMMA * running
            disc.append(running)
        disc = torch.tensor(disc[::-1], dtype=torch.float32)
        disc = (disc - disc.mean()) / (disc.std() + 1e-8)
        loss = -(torch.stack(log_probs) * disc).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    env.close()
    return policy, recent


def main():
    CKPT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    for seed in range(N_SEEDS):
        policy, recent = train_one(seed)
        torch.save(policy.state_dict(), CKPT_DIR / f"seed{seed}.pt")
        overall = float(np.mean(recent))
        last100 = float(np.mean(recent[-100:]))
        rows.append((seed, round(overall, 1), round(last100, 1)))
        print(f"seed {seed:2d}: train mean (all) {overall:7.1f} | (last 100) {last100:7.1f}", flush=True)

    with open(RESULTS_DIR / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "train_mean_all", "train_mean_last100"])
        w.writerows(rows)
    print(f"\nSaved {N_SEEDS} checkpoints to {CKPT_DIR}")
    print(f"Summary: {RESULTS_DIR / 'scores.csv'}")


if __name__ == "__main__":
    main()
