"""Check that sandbox/model is byte-identical to the candidate directory named in $NAME."""
import os
import sys

sys.path.insert(0, '/home/user/wsp14/sandbox/harness')
from save_edit import fingerprint  # noqa: E402

name = os.environ['NAME']
a = fingerprint(f'/home/user/wsp14/work/cands/{name}')
b = fingerprint('/home/user/wsp14/sandbox/model')
print('fingerprints equal:', a == b, a[:16])
sys.exit(0 if a == b else 1)
