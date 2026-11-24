"""Tests for phase14/save_edit.py, the checker of the weight-editing experiment's rules
(see the save_edit module docstring: an edit is ACCEPTED only if every base tensor is present with
the same shape and dtype and none is added, no protected tensor changed, no changed tensor is only
a constant multiple of the original, and the fixed files are byte-identical to the base's).
  check_model        (problems, changed) for a candidate model directory against the base;
  constant_multiple  the factor of a whole-tensor rescaling, 0.0 for an all-zero candidate, else None;
  tensors_in         {tensor name: tensor} over every shard of a model directory;
  fingerprint        one digest for the exact content of a model directory (every file).
Each case is built from a tiny five-tensor model, written to a temporary directory where files are needed.
main() and the module-level paths need the sandbox, and are not covered.

Standard library plus torch and safetensors. Usage (from the repository root):
  python -B -m unittest -v phase14/test_save_edit.py
"""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # the module under test is imported by bare name, as test_runcheck.py does
import save_edit  # noqa: E402
from save_edit import check_model, constant_multiple, fingerprint, tensors_in  # noqa: E402

EMBED = "model.language_model.embed_tokens.weight"
NORM = "model.language_model.norm.weight"                    # the final norm: protected
DOWN = "model.language_model.layers.0.mlp.down_proj.weight"
LN = "model.language_model.layers.0.input_layernorm.weight"   # a per-layer layernorm: not protected
Q = "model.language_model.layers.0.self_attn.q_proj.weight"
# two of the seven FIXED_FILES, as a small checkpoint has them
FILES = {"config.json": "{}", "tokenizer.json": "{}"}
SHARDS = {"model-00001-of-00002.safetensors": (EMBED, LN, Q),  # the embedding, then the layer's attention,
          "model-00002-of-00002.safetensors": (DOWN, NORM)}    # then its MLP and the final norm


def base_tensors():
    """The five tensors of a tiny model, the same on every call."""
    torch.manual_seed(0)
    return {EMBED: torch.randn(50, 8), NORM: torch.randn(8), DOWN: torch.randn(8, 16),
            LN: torch.randn(8), Q: torch.randn(8, 8)}


def write_model(path, shards, files=FILES):
    """A model directory from {shard file: {tensor name: tensor}}, plus the fixed files `files` ({name: text})."""
    path.mkdir(parents=True)
    for name, tensors in shards.items():
        save_file(tensors, path / name)
    for name, text in files.items():
        (path / name).write_text(text, encoding="utf-8")


def one_shard(tensors):
    """The tensors as a single model.safetensors file."""
    return {"model.safetensors": tensors}


def sharded(tensors):
    """The same tensors dealt out over the two shard files of SHARDS."""
    return {shard: {name: tensors[name] for name in names} for shard, names in SHARDS.items()}


def check_candidate(tensors, files=FILES, base_files=FILES, layout=one_shard):
    """check_model(base, candidate) in a temporary directory: the base holds the five tensors and the
    files `base_files`; the candidate holds `tensors` (no weight file at all when None) and the files
    `files`; both are written in the layout `layout`."""
    with tempfile.TemporaryDirectory() as tmp:
        base, cand = Path(tmp) / "base", Path(tmp) / "cand"
        write_model(base, layout(base_tensors()), base_files)
        write_model(cand, {} if tensors is None else layout(tensors), files)
        return check_model(base, cand)


class CheckModelTest(unittest.TestCase):
    """check_model(): (problems, changed) for a candidate model directory against the base"""

    # ---- accepted ----
    def test_an_identical_copy_is_accepted_with_nothing_changed(self):
        self.assertEqual(check_candidate(base_tensors()), ([], []))

    def test_a_targeted_edit_is_accepted_and_reported(self):
        tensors = base_tensors()
        tensors[DOWN][:, 3:5] += 0.25
        problems, changed = check_candidate(tensors)
        self.assertEqual(problems, [])
        self.assertEqual([c["tensor"] for c in changed], [DOWN])
        self.assertEqual(changed[0]["changed_values"], 16)    # two columns x eight rows

    def test_scaling_one_row_of_q_proj_is_accepted(self):
        # a row is not the whole tensor
        tensors = base_tensors()
        tensors[Q][0] *= 1.5
        problems, changed = check_candidate(tensors)
        self.assertEqual(problems, [])
        self.assertEqual([c["tensor"] for c in changed], [Q])

    def test_zeroing_one_column_of_down_proj_is_accepted(self):
        # one neuron's output is not the whole tensor
        tensors = base_tensors()
        tensors[DOWN][:, 5] = 0
        problems, changed = check_candidate(tensors)
        self.assertEqual(problems, [])
        self.assertEqual([c["tensor"] for c in changed], [DOWN])

    def test_two_targeted_edits_are_both_reported_in_name_order(self):
        # the shards are read q_proj first, down_proj second: the reverse of their name order
        tensors = base_tensors()
        tensors[DOWN][:, 3:5] += 0.25
        tensors[Q][0] *= 1.5
        problems, changed = check_candidate(tensors, layout=sharded)
        self.assertEqual(problems, [])
        self.assertEqual([(c["tensor"], c["changed_values"]) for c in changed], [(DOWN, 16), (Q, 8)])

    def test_max_abs_delta_is_reported_to_six_decimals(self):
        # +0.0625 on one column, -0.123456 on the next: the largest change is 0.123456 in size,
        # and it is the negative one
        tensors = base_tensors()
        tensors[DOWN][:, 3] += 0.0625
        tensors[DOWN][:, 4] -= 0.123456
        problems, changed = check_candidate(tensors)
        self.assertEqual(problems, [])
        self.assertAlmostEqual(changed[0]["max_abs_delta"], 0.123456, places=6)

    def test_a_fixed_file_absent_from_both_models_is_not_a_refusal(self):
        # FIXED_FILES names seven files: a checkpoint that has none of them is judged on its tensors
        self.assertEqual(check_candidate(base_tensors(), files={}, base_files={}), ([], []))

    # ---- refused: the change is a whole-tensor rescaling ----
    def test_scaling_the_whole_down_proj_is_refused_as_a_constant_multiple(self):
        tensors = base_tensors()
        tensors[DOWN] = tensors[DOWN] * 1.5
        problems, _ = check_candidate(tensors)
        self.assertEqual(len(problems), 1)
        self.assertIn("only multiplied by a constant", problems[0])

    def test_a_slight_rescaling_of_the_whole_down_proj_is_still_refused(self):
        tensors = base_tensors()
        tensors[DOWN] = tensors[DOWN] * 0.99
        problems, _ = check_candidate(tensors)
        self.assertEqual(len(problems), 1)
        self.assertIn("only multiplied by a constant", problems[0])

    def test_zeroing_the_whole_down_proj_is_refused_as_a_constant_multiple(self):
        tensors = base_tensors()
        tensors[DOWN] = torch.zeros_like(tensors[DOWN])
        problems, _ = check_candidate(tensors)
        self.assertEqual(len(problems), 1)
        self.assertIn("only multiplied by a constant", problems[0])

    def test_scaling_a_whole_layernorm_is_refused_as_a_constant_multiple(self):
        # not a protected tensor: it is the whole-tensor rescaling that refuses it
        tensors = base_tensors()
        tensors[LN] = tensors[LN] * 0.9
        problems, _ = check_candidate(tensors)
        self.assertEqual(len(problems), 1)
        self.assertIn("only multiplied by a constant", problems[0])

    # ---- refused: a protected tensor changed ----
    def test_scaling_one_row_of_the_embedding_is_refused_as_protected(self):
        tensors = base_tensors()
        tensors[EMBED][0] *= 1.5
        problems, _ = check_candidate(tensors)
        self.assertEqual(len(problems), 1)
        self.assertIn("protected", problems[0])

    def test_scaling_the_final_norm_is_refused_as_protected(self):
        tensors = base_tensors()
        tensors[NORM] = tensors[NORM] * 1.5
        problems, _ = check_candidate(tensors)
        self.assertEqual(len(problems), 1)
        self.assertIn("protected", problems[0])

    # ---- refused: the tensors or a fixed file differ ----
    def test_an_added_tensor_is_refused(self):
        tensors = base_tensors()
        tensors["model.language_model.layers.0.mlp.up_proj.weight"] = torch.randn(16, 8)
        problems, _ = check_candidate(tensors)
        self.assertEqual(len(problems), 1)
        self.assertIn("tensor added", problems[0])

    def test_a_removed_tensor_is_refused(self):
        tensors = base_tensors()
        del tensors[LN]
        problems, _ = check_candidate(tensors)
        self.assertEqual(len(problems), 1)
        self.assertIn("tensor removed", problems[0])

    def test_a_tensor_of_another_dtype_is_refused(self):
        tensors = base_tensors()
        tensors[DOWN] = tensors[DOWN].to(torch.bfloat16)
        problems, _ = check_candidate(tensors)
        self.assertEqual(len(problems), 1)
        self.assertIn("reshaped or re-typed", problems[0])

    def test_a_changed_config_is_refused(self):
        problems, _ = check_candidate(base_tensors(), files={**FILES, "config.json": '{"model_type": "edited"}'})
        self.assertEqual(len(problems), 1)
        self.assertIn("fixed", problems[0])

    def test_a_missing_tokenizer_is_refused(self):
        problems, _ = check_candidate(base_tensors(), files={"config.json": "{}"})
        self.assertEqual(len(problems), 1)
        self.assertIn("tokenizer.json", problems[0])

    def test_a_candidate_without_a_weight_file_is_refused(self):
        problems, _ = check_candidate(None)
        self.assertEqual(len(problems), 1)
        self.assertIn("no safetensors", problems[0])

    def test_an_added_fixed_file_is_refused(self):
        # a chat template the base does not have changes every generation: not an edit of weight values
        problems, _ = check_candidate(base_tensors(), files={**FILES, "chat_template.jinja": "turn 1"})
        self.assertEqual(problems, ["chat_template.jinja was added: the config, tokenizer and chat template are fixed"])

    def test_every_non_weight_file_of_the_base_is_fixed(self):
        # the shard index is not among the seven names in FIXED_FILES, yet the engine reads it
        index = "model.safetensors.index.json"
        problems, _ = check_candidate(base_tensors(), files={**FILES, index: "{}"},
                                      base_files={**FILES, index: '{"a": "model.safetensors"}'})
        self.assertEqual(len(problems), 1)
        self.assertIn(f"{index} differs from the base", problems[0])

    def test_a_tensor_that_lives_in_two_weight_files_is_refused(self):
        # the engine would load one of the two copies: the checker must not silently have compared the other
        tensors = base_tensors()
        tensors[DOWN][:, 3] += 0.25
        with tempfile.TemporaryDirectory() as tmp:
            base, cand = Path(tmp) / "base", Path(tmp) / "cand"
            write_model(base, one_shard(base_tensors()))
            write_model(cand, {"a.safetensors": {Q: tensors[Q]}, "b.safetensors": {Q: tensors[Q], DOWN: tensors[DOWN]}})
            problems, changed = check_model(base, cand)
        self.assertEqual(len(problems), 1)
        self.assertIn("appears in two weight files", problems[0])
        self.assertEqual(changed, [])


class ConstantMultipleTest(unittest.TestCase):
    """constant_multiple(): the factor k if cand is k x base entry for entry (0.0 if cand is all zeros), else None"""

    def test_an_exact_rescaling_gives_its_factor(self):
        base = base_tensors()[DOWN]
        k = constant_multiple(base, 2 * base)
        self.assertIsNotNone(k)
        self.assertAlmostEqual(k, 2.0, places=6)

    def test_a_rescaling_stored_in_bfloat16_gives_its_factor(self):
        # the module's 1% tolerance exists for bf16 rounding (its REL_TOL comment): a rescaling stored
        # in bfloat16 leaves errors of up to 0.4% of the largest entry, which float32 never shows
        base = base_tensors()[DOWN].to(torch.bfloat16)
        cand = (base.float() * 0.99).to(torch.bfloat16)
        k = constant_multiple(base, cand)
        self.assertIsNotNone(k, "the bfloat16 rounding is read as an edit")
        self.assertAlmostEqual(k, 0.99, places=3)

    def test_an_all_zero_candidate_gives_zero(self):
        base = base_tensors()[DOWN]
        self.assertEqual(constant_multiple(base, torch.zeros_like(base)), 0.0)

    def test_one_scaled_row_gives_none(self):
        base = base_tensors()[Q]
        cand = base.clone()
        cand[0] *= 1.5
        self.assertIsNone(constant_multiple(base, cand))

    def test_an_all_zero_base_gives_none(self):
        # no factor is defined: zero times any k is still zero
        cand = base_tensors()[DOWN]
        self.assertIsNone(constant_multiple(torch.zeros_like(cand), cand))

    def test_a_tiny_nudge_is_not_a_rescaling(self):
        # the largest entry moved by 1% of itself: the change is as rescaling-like as a one-entry edit
        # can be, but the factor it implies is within 0.001 of 1.0
        base = base_tensors()[DOWN]
        cand = base.clone()
        cand.flatten()[int(base.abs().argmax())] *= 1.01
        fit = float((base * cand).sum() / (base * base).sum())    # the factor a rescaling would need
        self.assertLess(abs(fit - 1.0), 0.001)
        self.assertIsNone(constant_multiple(base, cand))


class TensorsInTest(unittest.TestCase):
    """tensors_in(): {tensor name: tensor} over every shard of a model directory"""

    def test_reads_the_tensors_of_two_shard_files(self):
        expected = base_tensors()
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"
            write_model(model, sharded(expected))
            tensors = tensors_in(model)
        self.assertEqual(sorted(tensors), sorted(expected))
        for name, tensor in expected.items():
            self.assertTrue(torch.equal(tensors[name], tensor), name)


class FingerprintTest(unittest.TestCase):
    """fingerprint(): one digest for the exact content of a model directory, every file"""

    def test_equal_for_the_same_shards_and_different_after_one_tensor_is_edited(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, copy, edited = (Path(tmp) / name for name in ("model", "copy", "edited"))
            write_model(model, sharded(base_tensors()))
            write_model(copy, sharded(base_tensors()))
            tensors = base_tensors()
            tensors[DOWN][:, 3] += 0.25
            write_model(edited, sharded(tensors))
            self.assertEqual(fingerprint(model), fingerprint(copy))
            self.assertNotEqual(fingerprint(model), fingerprint(edited))

    def test_different_for_the_same_tensors_under_another_shard_file_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, renamed = Path(tmp) / "model", Path(tmp) / "renamed"
            write_model(model, one_shard(base_tensors()))
            write_model(renamed, {"renamed.safetensors": base_tensors()})
            # the same bytes, so only the file's name can tell the two directories apart
            self.assertEqual((model / "model.safetensors").read_bytes(),
                             (renamed / "renamed.safetensors").read_bytes())
            self.assertNotEqual(fingerprint(model), fingerprint(renamed))

    def test_a_fixed_file_edited_after_the_check_changes_the_fingerprint(self):
        # a receipt covers the whole directory: config, tokenizer and chat template included
        with tempfile.TemporaryDirectory() as tmp:
            model, edited = Path(tmp) / "model", Path(tmp) / "edited"
            files = {**FILES, "chat_template.jinja": "turn 1"}
            write_model(model, one_shard(base_tensors()), files)
            write_model(edited, one_shard(base_tensors()), {**files, "chat_template.jinja": "turn 2"})
            self.assertNotEqual(fingerprint(model), fingerprint(edited))
            (edited / "chat_template.jinja").write_text("turn 1", encoding="utf-8")
            self.assertEqual(fingerprint(model), fingerprint(edited))
            (edited / "generation_config.json").write_text("{}", encoding="utf-8")   # a file added later
            self.assertNotEqual(fingerprint(model), fingerprint(edited))


class MainTest(unittest.TestCase):
    """main(): the command line — REFUSED with exit status 1, or ACCEPTED with a receipt"""

    def run_main(self, base, cand):
        """main() on `cand` against `base`: (exit status, printed text, the receipts folder)."""
        receipts = base.parent / "state" / "receipts"
        printed, status = io.StringIO(), 0
        with (mock.patch.object(save_edit, "BASE", base), mock.patch.object(save_edit, "RECEIPTS", receipts),
              mock.patch.object(sys, "argv", ["save_edit.py", str(cand), "--note", "a test"]),
              contextlib.redirect_stdout(printed)):
            try:
                save_edit.main()
            except SystemExit as raised:                  # a refusal exits 1
                status = raised.code
        return status, printed.getvalue(), receipts

    def test_an_edited_candidate_is_accepted_and_gets_its_receipt(self):
        tensors = base_tensors()
        tensors[DOWN][:, 3:5] += 0.25
        with tempfile.TemporaryDirectory() as tmp:
            base, cand = Path(tmp) / "base", Path(tmp) / "cand"
            write_model(base, one_shard(base_tensors()))
            write_model(cand, one_shard(tensors))
            status, printed, receipts = self.run_main(base, cand)
            self.assertEqual(status, 0, printed)
            self.assertIn("ACCEPTED: 1 tensor(s) changed", printed)
            self.assertEqual([p.name for p in receipts.glob("*.json")], [f"{fingerprint(cand)}.json"])

    def test_the_base_model_itself_is_accepted_without_a_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, cand = Path(tmp) / "base", Path(tmp) / "cand"
            write_model(base, one_shard(base_tensors()))
            write_model(cand, one_shard(base_tensors()))
            status, printed, receipts = self.run_main(base, cand)
            self.assertEqual(status, 0, printed)
            self.assertIn("identical to the base model", printed)
            self.assertFalse(receipts.exists())

    def test_the_same_tensors_in_other_bytes_get_a_receipt(self):
        # no tensor differs from the base, yet the files do: the evaluator checks fingerprints, so the
        # candidate needs a receipt all the same
        with tempfile.TemporaryDirectory() as tmp:
            base, cand = Path(tmp) / "base", Path(tmp) / "cand"
            write_model(base, one_shard(base_tensors()))
            write_model(cand, sharded(base_tensors()))
            self.assertNotEqual(fingerprint(base), fingerprint(cand))
            status, printed, receipts = self.run_main(base, cand)
            self.assertEqual(status, 0, printed)
            self.assertIn("no tensor differs from the base model", printed)
            self.assertEqual(sorted(p.name for p in receipts.glob("*.json")), [f"{fingerprint(cand)}.json"])

    def test_a_refused_candidate_gets_no_receipt(self):
        tensors = base_tensors()
        tensors[DOWN] = tensors[DOWN] * 1.5           # a whole-tensor rescaling: refused
        with tempfile.TemporaryDirectory() as tmp:
            base, cand = Path(tmp) / "base", Path(tmp) / "cand"
            write_model(base, one_shard(base_tensors()))
            write_model(cand, one_shard(tensors))
            status, printed, receipts = self.run_main(base, cand)
            self.assertEqual((status, receipts.exists()), (1, False))
            self.assertIn("REFUSED", printed)
            self.assertIn("only multiplied by a constant", printed)


if __name__ == "__main__":
    unittest.main()
