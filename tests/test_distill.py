"""Tests for src/distill.py.

Run from the repo root:
    python -m unittest discover tests
"""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import build_dataset as bd  # noqa: E402
import distill as ds  # noqa: E402
import evaluate as ev  # noqa: E402

_, AXES = ev.load_taxonomy()
SILVER = {"energy": "high", "social": "solo", "engagement": "active", "commitment": "ongoing",
          "time_of_day": "any", "district": "gaming", "mood": ["intense"]}
TEACHER_SAYS = json.dumps({**SILVER, "energy": "low", "commitment": "an-evening", "mood": ["playful", "calm"]})


class ToLabel(unittest.TestCase):
    def test_sets_back_to_labels(self):
        by_name = {a.name: a for a in AXES}
        self.assertEqual(ds.to_label(by_name["commitment"], frozenset({"minutes"})), "minutes")
        # Mood comes back in taxonomy order, whatever order the teacher used.
        self.assertEqual(ds.to_label(by_name["mood"], frozenset({"playful", "calm"})), ["calm", "playful"])
        self.assertIsNone(ds.to_label(by_name["mood"], ev.NO_ANSWER))


class DistilLabels(unittest.TestCase):
    ITEM = {"id": "steam:1", "labels": SILVER}

    def test_only_teacher_axes_change(self):
        labels = ds.distil_labels(self.ITEM, TEACHER_SAYS, None, AXES)
        self.assertEqual(labels["commitment"], "an-evening")
        self.assertEqual(labels["mood"], ["calm", "playful"])
        self.assertEqual(labels["energy"], "high")  # the teacher said low; energy stays silver
        self.assertEqual({k: v for k, v in labels.items() if k not in ds.TEACHER_AXES},
                         {k: v for k, v in SILVER.items() if k not in ds.TEACHER_AXES})

    def test_invalid_teacher_answer_becomes_null_not_silver(self):
        labels = ds.distil_labels(self.ITEM, '{"commitment": "a while", "mood": ["cosy"]}', None, AXES)
        self.assertIsNone(labels["commitment"])
        self.assertIsNone(labels["mood"])
        self.assertIsNone(ds.distil_labels(self.ITEM, "no idea", None, AXES)["mood"])

    def test_hand_checked_example_labels_win(self):
        checked = {**SILVER, "commitment": "minutes", "mood": ["melancholic"]}
        labels = ds.distil_labels(self.ITEM, TEACHER_SAYS, checked, AXES)
        self.assertEqual((labels["commitment"], labels["mood"]), ("minutes", ["melancholic"]))


class Build(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            private, public, preds = tmp / "private", tmp / "public", tmp / "preds"
            for d in (private, public, preds):
                d.mkdir()
            items = [{"id": f"steam:{n}", "source": "steam", "group": f"dev{n}", "text": f"secret text {n}",
                      "labels": SILVER, "flags": [], "leaks": [], "meta": {"name": f"Game {n}"}} for n in range(3)]
            for split in ds.SPLITS:
                (private / f"{split}.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
            # Teacher answered items 0 and 1; item 2 is the few-shot example.
            for split in ds.SPLITS:
                (preds / f"teacher-{split}.jsonl").write_text(
                    json.dumps({"id": "steam:0", "output": TEACHER_SAYS}) + "\n"
                    + json.dumps({"id": "steam:1", "output": "garbage"}) + "\n")
            fewshot = tmp / "fewshot.json"
            fewshot.write_text(json.dumps({"examples": [
                {"id": "steam:2", "source": "steam", "verified": True,
                 "labels": {**SILVER, "commitment": "minutes", "mood": ["calm"]}}]}))

            with mock.patch.multiple(ds, PRIVATE=private, PUBLIC=public, REPORT_PATH=tmp / "report.json"), \
                    mock.patch.multiple(bd, PRIVATE_PROCESSED=private, PUBLIC_PROCESSED=public), \
                    mock.patch.object(ds.rb, "PREDICTIONS_DIR", preds), \
                    mock.patch.object(ds.prompts, "FEWSHOT_PATH", fewshot), redirect_stdout(io.StringIO()):
                ds.build()

            train = {i["id"]: i for i in map(json.loads, (private / "train_distilled.jsonl").read_text().splitlines())}
            self.assertEqual(train["steam:0"]["labels"]["commitment"], "an-evening")
            self.assertIsNone(train["steam:1"]["labels"]["commitment"])
            self.assertEqual(train["steam:2"]["labels"]["mood"], ["calm"])
            self.assertEqual(train["steam:0"]["text"], "secret text 0")
            public_text = (public / "train_distilled.jsonl").read_text()
            self.assertNotIn("secret", public_text)
            self.assertNotIn("Game", public_text)
            report = json.loads((tmp / "report.json").read_text())
            self.assertEqual(report["splits"]["train"]["commitment"]["teacher_null"], 1)

    def test_unverified_examples_stop_the_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            fewshot = Path(tmp) / "fewshot.json"
            fewshot.write_text(json.dumps({"examples": [{"id": "a", "source": "steam", "verified": False,
                                                         "labels": SILVER}]}))
            with mock.patch.object(ds.prompts, "FEWSHOT_PATH", fewshot):
                with self.assertRaises(SystemExit):
                    ds.example_labels(ds.prompts.load_taxonomy())


if __name__ == "__main__":
    unittest.main()
