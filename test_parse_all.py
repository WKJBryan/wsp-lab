"""Every Python file of the repository parses.

A committed tool once held a syntax error that nothing caught until it was first run
(phase14/eval_pass.py, ISSUES #28): the per-phase test files import only what they test. This one
parses each file, so at least that cannot recur.

Standard library only. Usage (from the repository root):
  python -B -m unittest test_parse_all.py
"""

import ast
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKIP = {".venv", ".git", "__pycache__", "models", "node_modules"}   # environments and model weights


class EveryPythonFileParses(unittest.TestCase):
    def test_every_python_file_outside_the_generated_folders_parses(self):
        files = sorted(p for p in ROOT.rglob("*.py") if not any(part in SKIP for part in p.parts))
        self.assertGreater(len(files), 100)     # the search itself must not have gone empty
        bad = []
        with warnings.catch_warnings():
            # committed editor-session scripts carry invalid escape sequences: a warning, not an error
            warnings.simplefilter("ignore", SyntaxWarning)
            for p in files:
                try:
                    ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
                except (SyntaxError, UnicodeDecodeError, ValueError) as e:
                    bad.append(f"{p.relative_to(ROOT)}: {e}")
        self.assertEqual(bad, [], "\n".join(bad))


if __name__ == "__main__":
    unittest.main()
