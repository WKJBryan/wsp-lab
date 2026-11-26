"""Tests for phase14/eval_pass.py, the only way a blind editor measures a candidate (see the protocol's "Evaluation
tool" and "Measurement protocol" sections: a pass is charged once the tool holds the GPU lock, refunded if the engine
fails, the run fails a validity check or the tool is interrupted; a candidate needs a receipt from save_edit.py, or
must be the base; every attempt keeps its own texts and number).
The real tool runs end to end, as harness/eval_pass.py in a throw-away sandbox that setUp() builds in the layout of
make_sandbox.sh (harness/, base_model/, model/, practice.json, harness/state/) around the tiny model of
test_save_edit.py. Only the engine is faked: stand-ins for vllm's LLM and SamplingParams, for the tokenizer's chat
template and for the GPU lock sit in sys.modules, so no GPU is touched, the real vllm is never imported and the GPU
lock is never taken.
  main   the command line: run with runpy on a patched sys.argv, stdout and stderr captured, the exit status caught.
  counter_value   the pass and attempt counters as the tool reads them, on the harness copy loaded without its
                  command line: 0 when the file was never written; a stop when it no longer holds a number.
The bare-name modules the tool imports are set aside around each test and put back afterwards, so this file runs in
one process with phase14/test_save_edit.py and the phase3c tests in any order.

Standard library plus torch and safetensors. Usage (from the repository root, inside WSL):
  python -B -m unittest -v phase14/test_eval_pass.py
"""

import contextlib
import hashlib
import importlib
import io
import json
import os
import runpy
import shutil
import signal
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))  # test_save_edit.py's model helpers are imported by bare name, as it imports save_edit
from test_save_edit import DOWN, base_tensors, fingerprint, one_shard, write_model  # noqa: E402

# the reference benchmark question, read from its module without importing it by bare name
REFERENCE_QUESTION = runpy.run_path(str(REPO / "phase3c" / "benchmark.py"))["REFERENCE_QUESTION"]
HARNESS_SOURCES = (HERE / "eval_pass.py", HERE / "save_edit.py",                       # what make_sandbox.sh copies
                   *(REPO / "phase3c" / name for name in ("benchmark.py", "gpulock.py", "rescore.py", "runcheck.py")),
                   REPO / "phase3b" / "scorer.py")
BARE_NAMES = ("benchmark", "rescore", "runcheck", "save_edit", "scorer", "gpulock", "eval_pass", "vllm", "transformers")

HYPOTHESIS = "Nudging two columns of down_proj shortens the thinking"
PRACTICE = [  # the sandbox's practice pool, five problems over three difficulty levels
    dict(id="p0", q="What is $6\\times 7$?", gold="42", type="Algebra", level="Level 1"),
    dict(id="p1", q="What is $5+3$?", gold="8", type="Number Theory", level="Level 2"),
    dict(id="p2", q="What is $10-3$?", gold="7", type="Algebra", level="Level 1"),
    dict(id="p3", q="What is $9+6$?", gold="15", type="Geometry", level="Level 2"),
    dict(id="p4", q="What is $2\\times 2$?", gold="4", type="Algebra", level="Level 3"),
]
NORMAL_TEXTS = [  # the fake engine's answer to each practice problem: p1 boxes a wrong answer, p4 boxes none
    "<think>6 times 7 is 42.</think>The answer is $\\boxed{42}$.",
    "<think>5 and 3 make 9.</think>The answer is $\\boxed{9}$.",
    "<think>10 minus 3 leaves 7.</think>The answer is $\\boxed{7}$.",
    "<think>9 and 6 make 15.</think>The answer is $\\boxed{15}$.",
    "<think>2 times 2 is 4.</think>The answer is 4.",
]
SOUP_TAIL = "\u4e2d" * 30    # a fragment tail: thirty copies of one ideograph
ROW_KEYS = ("id", "gold", "text", "pred_boxed", "ok_boxed", "pred_rescore", "ok_rescored", "rule", "n_tokens", "finish",
            "closed")   # rule: since 2026-11-26, which reading decided ok_rescored (the checker's judge)


def module(name, **attrs):
    """A stand-in module called `name` that holds `attrs`."""
    mod = types.ModuleType(name)
    vars(mod).update(attrs)
    return mod


def apply_chat_template(msgs, add_generation_prompt, tokenize, enable_thinking):
    """The tokenizer's chat template as the measurement protocol has the tool call it: the user's text as the prompt."""
    assert add_generation_prompt is True, "the prompt was not asked for as a generation prompt"
    assert tokenize is False, "the tool asked the tokenizer for token ids instead of text"
    assert enable_thinking is True, "the prompt was asked for with thinking disabled"
    return msgs[0]["content"]


FAKE_TRANSFORMERS = module("transformers", AutoTokenizer=types.SimpleNamespace(
    from_pretrained=lambda name: types.SimpleNamespace(apply_chat_template=apply_chat_template)))
FAKE_GPULOCK = module("gpulock", wait_for_gpu=lambda: None)   # the real one takes the file lock that another job holds


def generation(text):
    """One finished output of the fake engine: `text`, a token per character, ended by a stop token."""
    out = types.SimpleNamespace(text=text, token_ids=range(len(text)), finish_reason="stop")
    return types.SimpleNamespace(outputs=[out])


class FakeEngine:
    """vllm as the tool sees it: LLM(**kw) and SamplingParams(**kw). A test sets `texts`, the answer to the prompts in
    order and repeated as needed, or `error`, with which generate() raises a RuntimeError instead. After a run the test
    reads `constructed` (the keyword arguments of every LLM(...) call), `prompts` and `sampling_params`."""

    def __init__(self, texts=()):
        self.texts, self.error = texts, None
        self.constructed, self.prompts, self.sampling_params = [], None, None

    def LLM(self, **kwargs):
        self.constructed.append(kwargs)
        return types.SimpleNamespace(generate=self.generate)

    @staticmethod
    def SamplingParams(**kwargs):
        return kwargs

    def generate(self, prompts, sp):
        self.prompts, self.sampling_params = prompts, sp
        if self.error:
            raise RuntimeError(self.error)
        return [generation(self.texts[i % len(self.texts)]) for i in range(len(prompts))]


class EvalPassTest(unittest.TestCase):
    """main(): the command line of the evaluation tool, run end to end in a throw-away sandbox on the fake engine"""

    maxDiff = None   # a failing comparison shows its whole difference, five prompts long at most

    def setUp(self):
        # what another test module imported under the same bare name is put back when the test is over
        self.set_aside = {name: sys.modules.pop(name) for name in BARE_NAMES if name in sys.modules}
        self.addCleanup(self.put_back)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.harness, self.state = root / "harness", root / "harness" / "state"
        self.state.mkdir(parents=True)
        for source in HARNESS_SOURCES:
            shutil.copy(source, self.harness / source.name)
        (self.state / "harness.sha256.json").write_text(json.dumps(       # what make_sandbox.sh records
            {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(self.harness.glob("*.py"))}))
        (root / "practice.json").write_text(json.dumps(PRACTICE), encoding="utf-8")
        base, self.model = root / "base_model", root / "model"
        write_model(base, one_shard(base_tensors()))
        shutil.copytree(base, self.model)
        (self.state / "base_fingerprint.txt").write_text(fingerprint(base) + "\n")
        importlib.invalidate_caches()

        self.engine = FakeEngine(NORMAL_TEXTS)
        sys.modules["vllm"] = module("vllm", LLM=self.engine.LLM, SamplingParams=self.engine.SamplingParams)
        sys.modules["transformers"] = FAKE_TRANSFORMERS
        sys.modules["gpulock"] = FAKE_GPULOCK

    def put_back(self):
        """The harness copies the run imported and the fakes leave sys.modules; what setUp set aside returns."""
        for name in BARE_NAMES:
            sys.modules.pop(name, None)
        sys.modules.update(self.set_aside)

    # ---- helpers over the sandbox ----
    def run_tool(self, *argv):
        """Run harness/eval_pass.py in-process on `argv`: (exit status, None if it returned; stdout; stderr).
        The tool installs a process-wide SIGTERM handler; the caller's handler is put back afterwards."""
        out, err, status = io.StringIO(), io.StringIO(), None
        before = signal.getsignal(signal.SIGTERM)
        try:
            with (mock.patch.object(sys, "argv", ["eval_pass.py", *argv]),
                  mock.patch.object(sys, "path", list(sys.path)),
                  contextlib.redirect_stdout(out), contextlib.redirect_stderr(err)):
                try:
                    runpy.run_path(str(self.harness / "eval_pass.py"), run_name="__main__")
                except SystemExit as raised:
                    status = raised.code
        finally:
            signal.signal(signal.SIGTERM, before)
        return status, out.getvalue(), err.getvalue()

    def harness_function(self, name):
        """The function `name` of the harness's eval_pass.py, with the module loaded and its command line not run."""
        with mock.patch.object(sys, "path", list(sys.path)):
            return runpy.run_path(str(self.harness / "eval_pass.py"), run_name="eval_pass_under_test")[name]

    def counter(self):
        """The pass counter, as the tool's used() reads it: 0 when it was never written."""
        path = self.state / "passes.txt"
        return int(path.read_text()) if path.exists() else 0

    def set_counter(self, passes):
        (self.state / "passes.txt").write_text(str(passes))

    def ledger(self):
        """The lines of the ledger."""
        path = self.state / "ledger.md"
        return path.read_text().splitlines() if path.exists() else []

    def generations(self, k):
        """The rows the tool kept for pass `k`: one JSON object per line of state/gen_<k>.jsonl."""
        lines = (self.state / f"gen_{k}.jsonl").read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines]

    def edited_model(self):
        """A third directory: the base with two columns of down_proj nudged, the edit test_save_edit.py accepts."""
        tensors = base_tensors()
        tensors[DOWN][:, 3:5] += 0.25
        edited = self.model.parent / "edited"
        write_model(edited, one_shard(tensors))
        return edited

    # ---- the pass runs ----
    def test_a_pass_on_the_unmodified_model_is_charged_and_summarised(self):
        code, out, _ = self.run_tool(str(self.model), "--hypothesis", HYPOTHESIS)
        self.assertEqual(code, None, out)
        summary = json.loads(out)
        self.assertEqual(self.counter(), 1)
        ledger = self.ledger()
        self.assertEqual(len(ledger), 2, ledger)        # the hypothesis row and the outcome row
        self.assertIn(HYPOTHESIS, ledger[0])
        self.assertIn("MEASURED", ledger[1])
        rows = self.generations(1)
        self.assertEqual([r["id"] for r in rows], [p["id"] for p in PRACTICE])     # one line per problem
        self.assertEqual({tuple(sorted(r)) for r in rows}, {tuple(sorted(ROW_KEYS))})  # each with every recorded field
        self.assertEqual((summary["n"], summary["pass"], summary["remaining"], summary["candidate"]),
                         (len(PRACTICE), 1, 14, "base model"))
        self.assertEqual(summary["accuracy_boxed"], 60.0)     # three of the five texts box the gold answer
        self.assertEqual(json.loads((self.state / "eval_pass_1.json").read_text()), summary)

    def test_the_engine_is_asked_for_exactly_the_measurement_protocol(self):
        code, out, _ = self.run_tool(str(self.model), "--hypothesis", HYPOTHESIS)
        # a prompt without thinking or without a generation prompt makes the fake tokenizer refuse it, and the pass fail
        self.assertEqual(code, None, out)
        self.assertEqual(self.engine.prompts, [REFERENCE_QUESTION.format(q=p["q"]) for p in PRACTICE])
        self.assertEqual(self.engine.sampling_params,
                         dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5,
                              repetition_penalty=1.0, max_tokens=32768, seed=20261124))
        (built,) = self.engine.constructed
        self.assertEqual((built["model"], built["max_model_len"], built["enable_prefix_caching"]),
                         (str(self.model), 40000, False))

    def test_n_runs_only_the_first_problems_without_the_admission_test(self):
        code, out, _ = self.run_tool(str(self.model), "--hypothesis", HYPOTHESIS, "--n", "2")
        self.assertEqual(code, None, out)
        summary = json.loads(out)
        self.assertEqual(self.engine.prompts, [REFERENCE_QUESTION.format(q=p["q"]) for p in PRACTICE[:2]])
        self.assertEqual(summary["n"], 2)
        self.assertIs(summary["validity"]["admission_tested"], False)

    def test_an_edited_candidate_with_its_receipt_is_measured_under_its_fingerprint(self):
        edited = self.edited_model()
        fp = fingerprint(edited)
        receipts = self.state / "receipts"
        receipts.mkdir()
        (receipts / f"{fp}.json").write_text(json.dumps(dict(note="two columns of down_proj nudged")))
        code, out, _ = self.run_tool(str(edited), "--hypothesis", HYPOTHESIS)
        self.assertEqual(code, None, out)
        self.assertEqual(json.loads(out)["candidate"], fp[:16])
        self.assertEqual(self.counter(), 1)

    def test_by_level_counts_correct_and_total_per_difficulty_level(self):
        code, out, _ = self.run_tool(str(self.model), "--hypothesis", HYPOTHESIS)
        self.assertEqual(code, None, out)
        # three entries for five problems: both Level 1 problems right, one of two in Level 2, none in Level 3
        self.assertEqual(json.loads(out)["by_level"], {"Level 1": [2, 2], "Level 2": [1, 2], "Level 3": [0, 1]})

    # ---- the counters, read before a pass is charged (reviewed 2026-11-26) ----
    def test_a_counter_file_that_does_not_exist_reads_as_zero(self):
        counter_value = self.harness_function("counter_value")
        self.assertEqual(counter_value(self.state / "passes.txt"), 0)

    def test_a_counter_file_that_does_not_hold_a_number_stops_the_tool(self):
        (self.state / "passes.txt").write_text("abc")
        counter_value = self.harness_function("counter_value")
        with self.assertRaises(SystemExit) as raised:
            counter_value(self.state / "passes.txt")
        message = str(raised.exception.code)
        self.assertIn("passes.txt", message)                # the file that is damaged
        self.assertIn("does not hold a number", message)

    def test_an_empty_counter_file_stops_the_tool(self):
        (self.state / "passes.txt").write_text("")
        counter_value = self.harness_function("counter_value")
        with self.assertRaises(SystemExit) as raised:
            counter_value(self.state / "passes.txt")
        self.assertIn("does not hold a number", str(raised.exception.code))

    def test_a_damaged_pass_counter_stops_the_pass_before_anything_is_charged(self):
        # a budget that silently starts again is worse than a stopped pass
        (self.state / "passes.txt").write_text("abc")
        code, out, err = self.run_tool(str(self.model), "--hypothesis", HYPOTHESIS)
        self.assertIn("does not hold a number", str(code))
        self.assertEqual((self.state / "passes.txt").read_text(), "abc")     # left as it was found
        self.assertEqual((self.ledger(), self.engine.constructed, self.attempts()), ([], [], 0))

    # ---- refused before a pass is charged ----
    def test_an_exhausted_budget_is_refused_without_a_charge_or_an_engine(self):
        self.set_counter(15)
        code, out, _ = self.run_tool(str(self.model), "--hypothesis", HYPOTHESIS)
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(out), dict(refused=True, reason="budget exhausted", used=15, max=15))
        self.assertEqual((self.counter(), self.ledger(), self.engine.constructed), (15, [], []))

    def test_an_edited_candidate_without_a_receipt_is_refused_without_a_charge_or_an_engine(self):
        self.set_counter(3)
        code, out, _ = self.run_tool(str(self.edited_model()), "--hypothesis", HYPOTHESIS)
        self.assertEqual(code, 2)
        message = json.loads(out)
        self.assertTrue(message["refused"], message)
        self.assertIn("receipt", message["reason"])
        self.assertEqual((self.counter(), self.ledger(), self.engine.constructed), (3, [], []))

    def test_a_pass_without_a_hypothesis_is_rejected_and_nothing_is_charged(self):
        self.set_counter(3)
        code, _, err = self.run_tool(str(self.model))
        self.assertEqual(code, 2)
        self.assertIn("--hypothesis", err)
        self.assertEqual((self.counter(), self.ledger(), self.engine.constructed), (3, [], []))

    def test_an_n_outside_1_to_128_is_refused_without_a_charge_or_an_engine(self):
        # sys.exit(<text>): the text is the exit object here and goes to stderr with status 1 on the real command line
        self.set_counter(3)
        for n in ("0", "-1", "129"):
            with self.subTest(n=n):
                code, _, _ = self.run_tool(str(self.model), "--hypothesis", HYPOTHESIS, "--n", n)
                self.assertEqual(code, f"--n must be between 1 and 128 (got {n})")
        self.assertEqual((self.counter(), self.ledger(), self.engine.constructed), (3, [], []))

    def test_a_candidate_that_is_not_a_directory_is_refused_without_a_charge(self):
        self.set_counter(3)
        missing = self.model.parent / "nowhere"
        code, _, _ = self.run_tool(str(missing), "--hypothesis", HYPOTHESIS)
        self.assertEqual(code, f"{missing} is not a model directory")
        self.assertEqual((self.counter(), self.ledger(), self.engine.constructed), (3, [], []))

    def test_a_sandbox_without_the_base_fingerprint_stops_before_anything_is_charged(self):
        self.set_counter(3)
        (self.state / "base_fingerprint.txt").unlink()
        code, _, _ = self.run_tool(str(self.model), "--hypothesis", HYPOTHESIS)
        self.assertEqual(code, f"{self.state / 'base_fingerprint.txt'} is missing: the sandbox was not built with "
                               "make_sandbox.sh")
        self.assertEqual((self.counter(), self.ledger(), self.engine.constructed), (3, [], []))

    def test_a_pipe_in_the_hypothesis_does_not_split_the_ledger_row(self):
        code, out, _ = self.run_tool(str(self.model), "--hypothesis", "if a | b, then c")
        self.assertEqual(code, None, out)
        (row, _) = self.ledger()                        # the hypothesis row; the outcome row has no free text
        self.assertEqual(row.count("|"), 6)             # | pass | time | candidate | n | hypothesis |
        self.assertIn("if a / b, then c", row)

    # ---- refunded after a pass was charged ----
    def test_an_engine_failure_exits_1_and_refunds_the_pass(self):
        self.engine.error = "CUDA out of memory"
        self.set_counter(4)
        code, out, _ = self.run_tool(str(self.model), "--hypothesis", HYPOTHESIS)
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out), dict(failed=True, refunded=True, remaining=11,
                                               error="RuntimeError: CUDA out of memory"))
        self.assertEqual(self.counter(), 4)
        refunded = [line for line in self.ledger() if "FAILED, pass refunded" in line]
        self.assertEqual(len(refunded), 1, self.ledger())
        self.assertIn("RuntimeError: CUDA out of memory", refunded[0])

    def test_a_multi_line_engine_error_still_takes_one_ledger_line(self):
        # a real engine's error message runs over several lines; the ledger keeps one line per entry
        self.engine.error = "Engine core initialization failed\nsecond line\nlast line: no kernel image"
        code, out, _ = self.run_tool(str(self.model), "--hypothesis", HYPOTHESIS)
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)["error"], f"RuntimeError: {self.engine.error}")     # printed whole
        ledger = self.ledger()
        self.assertEqual(len(ledger), 2, ledger)            # the hypothesis line and the refund line: nothing spilled
        self.assertIn("FAILED, pass refunded", ledger[1])
        self.assertIn("RuntimeError: Engine core initialization failed second line last line: no kernel image",
                      ledger[1])

    def test_a_run_that_fails_the_soup_check_is_charged_and_says_why(self):
        # fragments at the end of the generations are the candidate's own output: that is a measurement
        # of the candidate (a damaged one), so the pass counts and the summary says what failed
        self.engine.texts = [text + SOUP_TAIL for text in NORMAL_TEXTS]
        self.set_counter(4)
        code, out, _ = self.run_tool(str(self.model), "--hypothesis", HYPOTHESIS)
        self.assertEqual(code, None, out)
        summary = json.loads(out)
        self.assertEqual(summary["failed_checks"],
                         ["soup: 100.0% of generations end in non-Latin script (bar 20.0%)"])
        self.assertEqual((summary["passes_spent"], summary["remaining"]), (5, 10))
        self.assertEqual([r["id"] for r in self.generations(1)], [p["id"] for p in PRACTICE])
        self.assertEqual(self.counter(), 5)
        ledger = self.ledger()
        self.assertIn("MEASURED", ledger[1])
        self.assertIn("the candidate's own output, the pass counts", ledger[1])

    def test_a_run_whose_outcome_depends_on_the_request_position_is_refunded_and_not_reported(self):
        # the signature of a corrupted run (ISSUES #20): the engine, not the candidate, decides the outcome
        (self.model.parent / "practice.json").write_text(json.dumps(
            [dict(id=f"p{i}", q=f"What is {i}+1?", gold=str(i + 1), type="Algebra", level="Level 1") for i in range(100)]))

        def generate(prompts, sp):     # the first 50 requests get their gold answer, the last 50 get none
            return [generation(f"<think>done.</think>The answer is $\\boxed{{{i + 1}}}$." if i < 50
                               else "<think>done.</think>I cannot say.") for i in range(len(prompts))]
        self.engine.generate = generate
        self.set_counter(3)
        code, out, _ = self.run_tool(str(self.model), "--hypothesis", HYPOTHESIS)
        self.assertEqual(code, 2)
        record = json.loads(out)
        self.assertEqual((record["failed"], record["refused"] if "refused" in record else None, record["remaining"]),
                         (True, None, 12))
        self.assertIn("admission order (correct)", record["checks"][0])
        self.assertNotIn("accuracy_boxed", out)                # nothing that reads as a result of the candidate
        self.assertEqual(self.counter(), 3)
        self.assertEqual(len(self.generations(1)), 100)        # the texts stay as evidence
        self.assertTrue((self.state / "eval_pass_1.json").exists())   # and the summary stays for the audit
        self.assertIn("FAILED, pass refunded", self.ledger()[1])

    def test_a_termination_request_while_the_engine_runs_refunds_the_pass(self):
        def generate(prompts, sp):     # the tool is asked to terminate from outside while the engine runs
            os.kill(os.getpid(), signal.SIGTERM)
            return [generation(t) for t in NORMAL_TEXTS]
        self.engine.generate = generate
        self.set_counter(3)
        code, _, _ = self.run_tool(str(self.model), "--hypothesis", HYPOTHESIS)
        self.assertEqual(code, 143)
        self.assertEqual(self.counter(), 3)
        self.assertIn("INTERRUPTED, pass refunded", self.ledger()[1])
        self.assertEqual(sorted(p.name for p in self.state.glob("gen_*.jsonl")), [])

    # ---- the lock, the attempt number and the second fingerprint ----
    def test_a_tool_stopped_while_it_waits_for_the_gpu_lock_charges_nothing(self):
        def stopped():     # another job holds the GPU, and this tool is killed while it waits
            raise KeyboardInterrupt
        sys.modules["gpulock"] = module("gpulock", wait_for_gpu=stopped)
        self.set_counter(3)
        with self.assertRaises(KeyboardInterrupt):
            self.run_tool(str(self.model), "--hypothesis", HYPOTHESIS)
        self.assertEqual((self.counter(), self.ledger(), self.engine.constructed), (3, [], []))

    def test_a_tool_interrupted_while_the_engine_runs_refunds_the_pass_and_says_so(self):
        def generate(prompts, sp):
            raise KeyboardInterrupt
        self.engine.generate = generate
        self.set_counter(3)
        with self.assertRaises(KeyboardInterrupt):
            self.run_tool(str(self.model), "--hypothesis", HYPOTHESIS)
        self.assertEqual(self.counter(), 3)
        ledger = self.ledger()
        self.assertEqual(len(ledger), 2, ledger)            # the hypothesis row and the interruption row
        self.assertIn("INTERRUPTED, pass refunded", ledger[1])
        self.assertEqual(sorted(p.name for p in self.state.glob("gen_*.jsonl")), [])

    def test_a_refunded_attempt_keeps_its_own_texts_and_number(self):
        # attempt 1 is refunded for the engine's doing after its texts were written; attempt 2 is clean
        def by_position(prompts, sp):  # the first 50 requests get their gold answer, the last 50 get none
            return [generation(f"<think>done.</think>The answer is $\\boxed{{{i + 1}}}$." if i < 50
                               else "<think>done.</think>I cannot say.") for i in range(len(prompts))]
        (self.model.parent / "practice.json").write_text(json.dumps(
            [dict(id=f"p{i}", q=f"What is {i}+1?", gold=str(i + 1), type="Algebra", level="Level 1") for i in range(100)]))
        self.engine.generate = by_position
        code, out, _ = self.run_tool(str(self.model), "--hypothesis", HYPOTHESIS)
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(out)["texts"], "gen_1.jsonl")
        self.engine.texts = NORMAL_TEXTS
        del self.engine.generate             # back to the fake engine's own answers
        (self.model.parent / "practice.json").write_text(json.dumps(PRACTICE), encoding="utf-8")
        code, out, _ = self.run_tool(str(self.model), "--hypothesis", HYPOTHESIS)
        self.assertEqual(code, None, out)
        summary = json.loads(out)
        self.assertEqual((summary["pass"], summary["passes_spent"], summary["remaining"]), (2, 1, 14))
        self.assertEqual((self.counter(), self.attempts()), (1, 2))
        self.assertEqual((len(self.generations(1)), [r["id"] for r in self.generations(2)]),
                         (100, [p["id"] for p in PRACTICE]))    # the refunded attempt's texts were not replaced
        self.assertEqual(sorted(p.name for p in self.state.glob("eval_pass_*.json")),
                         ["eval_pass_1.json", "eval_pass_2.json"])
        ledger = self.ledger()
        self.assertEqual(len(ledger), 4, ledger)              # a hypothesis row and an outcome row per attempt
        self.assertIn("FAILED, pass refunded", ledger[1])
        self.assertIn("MEASURED", ledger[3])

    def test_the_candidate_is_fingerprinted_again_under_the_lock(self):
        def wait():        # the working copy changes while the pass waits for the GPU
            (self.model / "generation_config.json").write_text("{}", encoding="utf-8")
        sys.modules["gpulock"] = module("gpulock", wait_for_gpu=wait)
        code, out, _ = self.run_tool(str(self.model), "--hypothesis", HYPOTHESIS)
        self.assertEqual(code, 1)
        self.assertIn("changed while the pass waited", out)
        self.assertEqual((self.counter(), self.engine.constructed), (0, []))
        self.assertIn("the candidate changed while the pass waited for the GPU", self.ledger()[1])

    def test_a_harness_file_edited_after_the_sandbox_was_built_stops_the_tool(self):
        with (self.harness / "benchmark.py").open("a", encoding="utf-8") as f:
            f.write("\n# edited after the digests were recorded\n")
        self.set_counter(3)
        code, out, _ = self.run_tool(str(self.model), "--hypothesis", HYPOTHESIS)
        self.assertEqual(code, "benchmark.py does not match the digest recorded for the sandbox")
        self.assertEqual((self.counter(), self.ledger(), self.engine.constructed), (3, [], []))

    def test_a_sandbox_without_the_harness_digests_stops_the_tool(self):
        (self.state / "harness.sha256.json").unlink()
        code, _, _ = self.run_tool(str(self.model), "--hypothesis", HYPOTHESIS)
        self.assertEqual(code, f"{self.state / 'harness.sha256.json'} is missing: the sandbox was not built with "
                               "make_sandbox.sh")
        self.assertEqual((self.counter(), self.ledger(), self.engine.constructed), (0, [], []))

    def attempts(self):
        """The attempt counter: 0 when it was never written."""
        path = self.state / "attempts.txt"
        return int(path.read_text()) if path.exists() else 0


if __name__ == "__main__":
    unittest.main()
