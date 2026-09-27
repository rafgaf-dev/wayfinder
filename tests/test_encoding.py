"""Every text file the code opens must name its encoding.

Without one, Python uses the platform's default: UTF-8 on macOS and Linux,
but the system code page (e.g. cp1252) on Windows, where our UTF-8 data
(accents, curly quotes, emoji) would crash or be silently garbled.

Run from the repo root:
    python -m unittest discover tests
"""

import ast
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
CALLS = {"open", "read_text", "write_text"}


class ExplicitEncoding(unittest.TestCase):
    def test_every_text_mode_file_call_names_an_encoding(self):
        missing = []
        for path in sorted(SRC.glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in CALLS):
                    continue
                binary = any(isinstance(a, ast.Constant) and isinstance(a.value, str) and "b" in a.value
                             for a in node.args)
                if not binary and not any(k.arg == "encoding" for k in node.keywords):
                    missing.append(f"{path.name}:{node.lineno}")
        self.assertEqual(missing, [], "add encoding=\"utf-8\" to these calls")


if __name__ == "__main__":
    unittest.main()
