"""The project's GPU lock: one engine at a time, across sessions (ISSUES #18).

On 2026-11-18 two sessions started engines 30 seconds apart; each claimed most of the card, the
driver spilled into system RAM, and a control run stopped at 55 of 500 problems. A free-memory
check cannot prevent that, because an engine takes about two minutes to claim its memory.

Run inside WSL (POSIX file locks). The module keeps the lock handle itself, so it is held until
this process exits, however it exits; a crashed engine never leaves it held. A second call in the
same process does nothing. The lock is shared by processes with the same home directory in the
same Linux distribution, which is every launcher in this repository. Only runners that call this
take the lock: an older script that starts an engine without it can still collide.

Usage in a runner, before the engine is created:
    from gpulock import wait_for_gpu
    wait_for_gpu()                # blocks, with a message, while another lock-taking engine runs
"""

import fcntl
import sys
from pathlib import Path

LOCK = Path.home() / ".wsp-gpu.lock"
_held = None


def wait_for_gpu():
    """Hold the GPU lock for the life of this process; wait for it if another process holds it."""
    global _held
    if _held is not None:
        return
    _held = LOCK.open("w")
    try:
        fcntl.flock(_held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("waiting for the GPU lock: another engine is running", file=sys.stderr, flush=True)
        fcntl.flock(_held, fcntl.LOCK_EX)
