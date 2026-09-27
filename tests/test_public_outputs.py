"""The public prediction files must hold short JSON answers and nothing else.

A model that goes off-script can repeat its input, and the inputs are third-party
text (store copy, event listings) that the public repository must not contain:
four of the few-shot teacher's outputs once did exactly that. This test makes
the suite fail instead of relying on someone reading printed output.

When the private data repository is cloned at data/private/, it also checks that
no stretch of any description appears in any tracked file.

Run from the repo root:
    python -m unittest discover tests
"""

import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "results" / "predictions"
PRIVATE = ROOT / "data" / "private" / "processed"
MAX_OUTPUT_CHARS = 300  # clean answers are under 200


def is_clean_answer(output: str) -> bool:
    if output == "":  # redacted, with no answer to keep
        return True
    body = output.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        return isinstance(json.loads(body), dict) and len(output) <= MAX_OUTPUT_CHARS
    except json.JSONDecodeError:
        return False


class PublicOutputs(unittest.TestCase):
    def test_every_prediction_output_is_a_short_json_answer(self):
        bad = []
        for path in sorted(PREDICTIONS.glob("*.jsonl")):
            with path.open(encoding="utf-8") as f:
                for n, line in enumerate(f, 1):
                    if not is_clean_answer(json.loads(line)["output"]):
                        bad.append(f"{path.name}:{n}")
        self.assertEqual(bad, [], "redact these outputs to their parsed answer before publishing")

    def test_is_clean_answer(self):
        self.assertTrue(is_clean_answer('{"energy": "low"}'))
        self.assertTrue(is_clean_answer('```json\n{"energy": "low"}\n```'))
        self.assertTrue(is_clean_answer(""))
        self.assertFalse(is_clean_answer('Sure! Here is the game description: ... {"energy": "low"}'))
        self.assertFalse(is_clean_answer('{"note": "' + "x" * 400 + '"}'))

    @unittest.skipUnless(PRIVATE.exists(), "private data repository not cloned at data/private/")
    def test_no_description_text_in_tracked_files(self):
        tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
        blob = "\n".join((ROOT / f).read_text(encoding="utf-8", errors="ignore")
                         for f in tracked.stdout.split() if (ROOT / f).is_file())
        leaks = []
        for split in ("train", "val", "test"):
            with (PRIVATE / f"{split}.jsonl").open(encoding="utf-8") as f:
                for item in map(json.loads, f):
                    text = item["text"]
                    for k in range(0, max(1, len(text) - 40), 200):
                        window = text[k:k + 40]
                        # Skip dividers and other low-information stretches.
                        if len(window) == 40 and sum(c.isalpha() for c in window) >= 25 and window in blob:
                            leaks.append(item["id"])
                            break
        self.assertEqual(leaks, [], "description text found in tracked files")


if __name__ == "__main__":
    unittest.main()
