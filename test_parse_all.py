"""Every Python file of the repository's own code parses.

A committed tool once held a syntax error that nothing caught until it was first run
(phase14/eval_pass.py, ISSUES #28): the per-phase test files import only what they test. This one
parses each file, so at least that cannot recur.

Not parsed: what an AI editor wrote during a session, kept as evidence exactly as written under any
folder whose name starts with "session". Those scripts ran under the editor's own Python, not the
repository's; three of the phase-16 ones do not parse under every Python version (ISSUES #46).

Standard library only. Usage (from the repository root):
  python -B -m unittest test_parse_all.py
"""

import ast
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKIP = {".venv", ".git", "__pycache__", "models", "node_modules"}   # environments and model weights
EVIDENCE = "session"                                                # folders of editor-session evidence: session, session_A, ...


def is_own_code(p):
    return not any(part in SKIP or part.startswith(EVIDENCE) for part in p.parts)


class EveryPythonFileParses(unittest.TestCase):
    def test_every_python_file_outside_the_generated_folders_parses(self):
        files = sorted(p for p in ROOT.rglob("*.py") if is_own_code(p))
        self.assertGreater(len(files), 100)     # the search itself must not have gone empty
        bad = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            for p in files:
                try:
                    ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
                except (SyntaxError, UnicodeDecodeError, ValueError) as e:
                    bad.append(f"{p.relative_to(ROOT)}: {e}")
        self.assertEqual(bad, [], "\n".join(bad))


if __name__ == "__main__":
    unittest.main()
