"""How a benchmark question is put to the model under test.

The repository's own protocol sends the bare problem. The original authors' evaluation — and this
repository's calibration runs (the runners' --reference-question) — adds an instruction to box the
answer. Used by phase3c/run_vllm.py, phase8/run_vllm8.py and the phase-14 evaluator.
"""

# as the original authors' evaluation put it (their experiment README)
REFERENCE_QUESTION = "Problem:\n{q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def question(p, reference=False):
    """The user message for problem record `p` (whose text is `p["q"]`)."""
    return REFERENCE_QUESTION.format(q=p["q"]) if reference else p["q"]
