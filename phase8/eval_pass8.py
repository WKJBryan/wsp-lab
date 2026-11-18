"""Phase 8 C1 practice pass: phase4/eval_pass.py unchanged, plus a constant logit bias on
</think> (weights untouched) and a 7-pass budget. Run from the session dir (counter + ledger
live in the cwd):

  cd phase8/c1_session && python ../eval_pass8.py ~/wsp3c/stock ../../phase4/practice.json \
      --logit-bias 4 --hypothesis "b=+4: ..."
"""

import sys
from pathlib import Path

sys.set_int_max_str_digits(0)  # top level so the spawned engine process gets it too (ISSUES #15)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase4"))
import eval_pass as ep  # noqa: E402
from vllm import SamplingParams  # noqa: E402

if __name__ == "__main__":
    i = sys.argv.index("--logit-bias")
    bias = float(sys.argv[i + 1])
    del sys.argv[i:i + 2]
    ep.MAX_PASSES = 7
    ep.SamplingParams = lambda **kw: SamplingParams(logit_bias={248069: bias}, **kw)
    ep.main()
