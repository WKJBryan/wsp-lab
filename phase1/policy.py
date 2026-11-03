"""The CartPole policy network. Shared by training, evaluation, and the editor.

Fixed architecture: 4 -> 128 (ReLU) -> 128 (ReLU) -> 2 (action logits),
17,410 parameters. Small enough to dump as raw numbers in-context (~40-70K tokens),
which is how the original weight-space-programming experiments handled small nets.
"""

import torch
import torch.nn as nn

OBS_DIM = 4
HIDDEN = 128
ACT_DIM = 2


def make_policy() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(OBS_DIM, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, ACT_DIM),
    )
