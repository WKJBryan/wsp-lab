"""Build the Phase-2 net pool: five same-architecture nets on different tasks,
shuffled under neutral names A-E with sealed ground truth.

Per protocol.md: OS-random permutation; manifest written to phase2/sealed/
(gitignored until reveal). Console prints only file names and checksums —
no task-identifying information.

Usage: uv run python phase2/make_tasks.py
"""

import hashlib
import json
import secrets
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase1"))
from policy import make_policy  # noqa: E402

N = 4096          # supervised dataset size
STEPS = 2000      # identical supervised recipe for blobs / parity / interact / random
LR = 1e-3
OUT = HERE / "nets"
SEALED = HERE / "sealed"


def train_classifier(policy, X, y, steps=STEPS):
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(policy(X), y)
        loss.backward()
        opt.step()
    return float(loss.item())


def heldout_acc(policy, X, y):
    with torch.no_grad():
        pred = policy(X).argmax(dim=1)
    return float((pred == y).float().mean().item())


def main():
    OUT.mkdir(exist_ok=True)
    SEALED.mkdir(exist_ok=True)
    g = torch.Generator().manual_seed(20261104)
    results = {}

    nets = {}

    # cartpole: copy a solved seed from the Phase-1 pool
    nets["cartpole"] = torch.load(HERE.parent / "phase1" / "ckpt" / "seed7.pt",
                                  map_location="cpu", weights_only=True)

    # informative-feature indices for blobs and interact (ground truth)
    blob_feats = sorted(secrets.SystemRandom().sample(range(4), 2))
    while True:
        inter_feats = sorted(secrets.SystemRandom().sample(range(4), 2))
        if inter_feats != blob_feats:
            break

    X = torch.randn(N, 4, generator=g)

    # blobs: linear sum of 2 informative features
    torch.manual_seed(11)
    p = make_policy()
    yb = (X[:, blob_feats[0]] + 0.8 * X[:, blob_feats[1]] > 0).long()
    final_loss = train_classifier(p, X, yb)
    nets["blobs"] = {k: v.clone() for k, v in p.state_dict().items()}
    results["blobs"] = dict(loss=final_loss, acc=heldout_acc(p, X, yb), informative=blob_feats)

    # parity: all 16 rows of {0,1}^4 in +-1 encoding, y = XOR
    torch.manual_seed(22)
    p = make_policy()
    rows = torch.tensor([[float(b) for b in format(i, "04b")] for i in range(16)]) * 2 - 1
    yp = torch.tensor([bin(i).count("1") % 2 for i in range(16)]).long()
    final_loss = train_classifier(p, rows.repeat(256, 1), yp.repeat(256))
    nets["parity"] = {k: v.clone() for k, v in p.state_dict().items()}
    results["parity"] = dict(loss=final_loss, acc=heldout_acc(p, rows, yp), informative=[0, 1, 2, 3])

    # interact: sign product of 2 informative features
    torch.manual_seed(33)
    p = make_policy()
    yi = (X[:, inter_feats[0]] * X[:, inter_feats[1]] > 0).long()
    final_loss = train_classifier(p, X, yi)
    nets["interact"] = {k: v.clone() for k, v in p.state_dict().items()}
    results["interact"] = dict(loss=final_loss, acc=heldout_acc(p, X, yi), informative=inter_feats)

    # random: unlearnable labels, same recipe
    torch.manual_seed(44)
    p = make_policy()
    yr = torch.randint(0, 2, (N,), generator=g)
    final_loss = train_classifier(p, X, yr)
    nets["random"] = {k: v.clone() for k, v in p.state_dict().items()}
    results["random"] = dict(loss=final_loss, acc=heldout_acc(p, X, yr), informative=[])

    # shuffle under neutral names
    names = list(nets.keys())
    perm = list("ABCDE")
    secrets.SystemRandom().shuffle(perm)
    manifest = {}
    for neutral, task in zip(perm, names):
        f = OUT / f"{neutral}.pt"
        torch.save(nets[task], f)
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        manifest[neutral] = dict(task=task, sha256=sha)
        print(f"saved nets/{neutral}.pt  sha256={sha[:16]}...")

    (SEALED / "manifest.json").write_text(
        json.dumps({"assignment": manifest, "task_info": results}, indent=2)
    )
    print("sealed manifest written (gitignored until reveal)")


if __name__ == "__main__":
    main()
