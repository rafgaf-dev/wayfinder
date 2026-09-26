"""Tests for src/label_gold.py.

Run from the repo root:
    python -m unittest discover tests
"""

import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import build_dataset as bd  # noqa: E402
import label_gold as lg  # noqa: E402

TAXONOMY = lg.yaml.safe_load(lg.TAXONOMY_PATH.read_text())


class Parse(unittest.TestCase):
    def test_numbers_and_names(self):
        self.assertEqual(lg.parse("mood", "1 4", TAXONOMY), "calm;melancholic")
        self.assertEqual(lg.parse("mood", "Calm, playful", TAXONOMY), "calm;playful")
        self.assertEqual(lg.parse("commitment", "2", TAXONOMY), "an-evening")
        self.assertEqual(lg.parse("commitment", "an-evening", TAXONOMY), "an-evening")
        self.assertEqual(lg.parse("mood", "3 3", TAXONOMY), "intense")

    def test_rejects_what_the_build_would_reject(self):
        for axis, raw in (("mood", "1 2 3"), ("mood", "cosy"), ("mood", "9"), ("commitment", "1 2"),
                          ("commitment", "an evening"), ("district", "0")):
            with self.assertRaises(ValueError, msg=raw):
                lg.parse(axis, raw, TAXONOMY)

    def test_output_is_accepted_by_the_build(self):
        for axis, raw in (("mood", "5 2"), ("commitment", "3"), ("district", "6")):
            cell = lg.parse(axis, raw, TAXONOMY)
            self.assertIsNotNone(bd.parse_gold_cell(axis, cell, TAXONOMY, "x"))


class Session(unittest.TestCase):
    ROWS = [
        {"id": "steam:1", "source": "steam", "mood": "", "commitment": "", "district": "-"},
        {"id": "tm:1", "source": "ticketmaster", "mood": "", "commitment": "", "district": ""},
        {"id": "steam:2", "source": "steam", "mood": "", "commitment": "", "district": "-"},
    ]
    TEXTS = {"steam:1": "A game.", "tm:1": "A gig.", "steam:2": "Another."}

    def run_session(self, answers: list[str], rows=None, target=None) -> list[dict]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.ROWS[0]))
                writer.writeheader()
                writer.writerows(rows or self.ROWS)
            replies = iter(answers)
            with redirect_stdout(io.StringIO()):
                lg.run(path, TAXONOMY, self.TEXTS, prompt=lambda _q: next(replies), target=target)
            return lg.read_rows(path)[1]

    def test_labels_every_axis_and_asks_district_only_for_ticketmaster(self):
        rows = self.run_session(["1", "3", "2 5", "2", "3", "4", "1"])
        self.assertEqual([(r["mood"], r["commitment"], r["district"]) for r in rows], [
            ("calm", "ongoing", "-"), ("uplifting;playful", "an-evening", "music"), ("melancholic", "minutes", "-")])

    def test_quit_keeps_finished_items_and_drops_the_one_in_progress(self):
        rows = self.run_session(["1", "3", "2", "q"])
        self.assertEqual((rows[0]["mood"], rows[1]["mood"]), ("calm", ""))

    def test_invalid_input_is_asked_again(self):
        rows = self.run_session(["cosy", "1", "3", "q"])
        self.assertEqual(rows[0]["mood"], "calm")

    def test_skip_comes_back_round_at_the_end(self):
        # Skip item 1; label item 2 (mood, commitment, district) and item 3;
        # then item 1 comes round again.
        rows = self.run_session(["s", "1", "2", "3", "4", "1", "3", "3"])
        self.assertEqual((rows[0]["mood"], rows[0]["commitment"]), ("intense", "ongoing"))
        self.assertEqual(rows[1]["district"], "music")

    def test_back_revisits_and_enter_keeps_current_values(self):
        # Label item 1, go back from item 2, keep item 1's mood, change its commitment.
        rows = self.run_session(["1", "3", "b", "", "1", "q"])
        self.assertEqual((rows[0]["mood"], rows[0]["commitment"]), ("calm", "minutes"))

    def test_resumes_at_the_first_unlabelled_item(self):
        done = [{**self.ROWS[0], "mood": "calm", "commitment": "ongoing"}] + self.ROWS[1:]
        rows = self.run_session(["2", "2", "4", "q"], rows=done)
        self.assertEqual(rows[1]["mood"], "uplifting")

    def test_shows_the_text_from_the_private_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.ROWS[0]))
                writer.writeheader()
                writer.writerows(self.ROWS)
            shown = io.StringIO()
            with redirect_stdout(shown):
                lg.run(path, TAXONOMY, self.TEXTS, prompt=lambda _q: "q")
            self.assertIn("A game.", shown.getvalue())
            with self.assertRaises(SystemExit):  # a row without text is an error, not a blank screen
                lg.run(path, TAXONOMY, {"steam:1": "x"}, prompt=lambda _q: "q")

    def test_stops_at_the_target_and_leaves_the_rest_blank(self):
        # Target 2: label items 1 and 2; the session ends without asking about item 3.
        rows = self.run_session(["1", "3", "2", "2", "3"], target=2)
        self.assertEqual([r["mood"] for r in rows], ["calm", "uplifting", ""])

    def test_skip_wraps_within_the_target(self):
        # Target 2: skip item 1, label item 2, then item 1 comes back (not item 3).
        rows = self.run_session(["s", "2", "2", "3", "1", "1"], target=2)
        self.assertEqual([r["mood"] for r in rows], ["calm", "uplifting", ""])


if __name__ == "__main__":
    unittest.main()
