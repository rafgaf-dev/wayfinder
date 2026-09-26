"""Tests for src/evaluate.py, written before any model has run.

Expected numbers are worked by hand in the comments, so these tests check the
arithmetic rather than restating whatever the code happens to produce.

Run from the repo root:
    python -m unittest discover tests
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import evaluate as ev  # noqa: E402

ENERGY = ev.Axis("energy", ("low", "medium", "high"), multi=False, min_labels=1, max_labels=1)
MOOD = ev.Axis("mood", ("calm", "uplifting", "intense", "melancholic", "playful"),
               multi=True, min_labels=1, max_labels=2)
AXES = [ENERGY, MOOD]


def s(*labels: str) -> frozenset:
    return frozenset(labels)


def item(item_id: str, energy, mood, source: str = "steam", flags=()) -> dict:
    return {"id": item_id, "source": source, "text": "...",
            "labels": {"energy": energy, "mood": mood}, "flags": list(flags)}


def run_from_outputs(items: list[dict], outputs: dict[str, str]) -> ev.Run:
    predictions = {i: {"id": i, "output": o} for i, o in outputs.items()}
    return ev.align("test", items, predictions, AXES)


class ExtractJsonObject(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(ev.extract_json_object('{"energy": "low"}'), {"energy": "low"})

    def test_markdown_fence_and_preamble(self):
        text = 'Here is the answer:\n```json\n{"energy": "low"}\n```'
        self.assertEqual(ev.extract_json_object(text), {"energy": "low"})

    def test_skips_a_broken_brace_before_the_object(self):
        self.assertEqual(ev.extract_json_object('{oops} then {"energy": "high"}'), {"energy": "high"})

    def test_nested_objects_are_returned_whole(self):
        self.assertEqual(ev.extract_json_object('{"a": {"b": 1}}'), {"a": {"b": 1}})

    def test_no_object(self):
        self.assertIsNone(ev.extract_json_object("low, calm"))
        self.assertIsNone(ev.extract_json_object('["low"]'))
        self.assertIsNone(ev.extract_json_object(""))


class ParseValue(unittest.TestCase):
    def test_trims_and_lower_cases(self):
        self.assertEqual(ev.parse_value(ENERGY, "  High "), s("high"))

    def test_unknown_label_is_invalid(self):
        self.assertIsNone(ev.parse_value(ENERGY, "very high"))

    def test_non_string_is_invalid(self):
        self.assertIsNone(ev.parse_value(ENERGY, None))
        self.assertIsNone(ev.parse_value(ENERGY, ["high"]))
        self.assertIsNone(ev.parse_value(ENERGY, 3))

    def test_mood_accepts_a_bare_string(self):
        self.assertEqual(ev.parse_value(MOOD, "calm"), s("calm"))

    def test_mood_order_and_duplicates_do_not_matter(self):
        self.assertEqual(ev.parse_value(MOOD, ["intense", "Calm", "calm"]), s("calm", "intense"))

    def test_mood_over_the_limit_is_invalid_not_truncated(self):
        self.assertIsNone(ev.parse_value(MOOD, ["calm", "intense", "playful"]))

    def test_mood_empty_is_invalid(self):
        self.assertIsNone(ev.parse_value(MOOD, []))

    def test_mood_with_one_unknown_label_is_invalid(self):
        self.assertIsNone(ev.parse_value(MOOD, ["calm", "cosy"]))


class ParseOutput(unittest.TestCase):
    def test_keys_are_normalised(self):
        ok, labels = ev.parse_output('{"Energy ": "low", "MOOD": ["calm"]}', AXES)
        self.assertTrue(ok)
        self.assertEqual(labels, {"energy": s("low"), "mood": s("calm")})

    def test_one_bad_axis_does_not_spoil_the_others(self):
        ok, labels = ev.parse_output('{"energy": "extreme", "mood": "calm"}', AXES)
        self.assertTrue(ok)
        self.assertEqual(labels, {"energy": ev.NO_ANSWER, "mood": s("calm")})

    def test_unparseable(self):
        ok, labels = ev.parse_output("I think it is low energy.", AXES)
        self.assertFalse(ok)
        self.assertEqual(labels, {"energy": ev.NO_ANSWER, "mood": ev.NO_ANSWER})


class Gold(unittest.TestCase):
    def test_null_means_no_ground_truth(self):
        self.assertEqual(ev.parse_gold(item("a", "low", None), AXES), {"energy": s("low"), "mood": None})

    def test_invalid_gold_is_an_error_not_a_silent_skip(self):
        with self.assertRaises(ValueError):
            ev.parse_gold(item("a", "extreme", ["calm"]), AXES)


class SingleLabelMetrics(unittest.TestCase):
    def test_hand_worked_example(self):
        # gold a a a b / pred a a b b, with a=low, b=high.
        # accuracy 3/4.
        # low:  tp 2, fp 0, fn 1  -> F1 = 4 / (4 + 0 + 1) = 0.8
        # high: tp 1, fp 1, fn 0  -> F1 = 2 / (2 + 1 + 0) = 0.6667
        # macro over present labels (low, high) = 0.7333; medium has no gold
        # support, so it is not in the average.
        pairs = [(s("low"), s("low")), (s("low"), s("low")), (s("low"), s("high")), (s("high"), s("high"))]
        h = ev.headline(ENERGY, pairs)
        self.assertAlmostEqual(h["accuracy"], 0.75)
        self.assertAlmostEqual(h["macro_f1"], (0.8 + 2 / 3) / 2)

    def test_majority_floor_has_high_accuracy_but_low_macro_f1(self):
        # gold low low low high / pred low everywhere.
        # low:  tp 3, fp 1, fn 0 -> F1 = 6 / 7 = 0.857
        # high: tp 0, fp 0, fn 1 -> F1 = 0
        # macro = 0.4286, while accuracy is 0.75.
        pairs = [(s("low"), s("low"))] * 3 + [(s("high"), s("low"))]
        h = ev.headline(ENERGY, pairs)
        self.assertAlmostEqual(h["accuracy"], 0.75)
        self.assertAlmostEqual(h["macro_f1"], (6 / 7) / 2)

    def test_predicting_a_label_absent_from_gold_does_not_change_the_denominator(self):
        # gold low high / pred medium high. medium has no gold support, so the
        # average stays over (low, high): low F1 0, high F1 1 -> macro 0.5.
        # Averaging over predicted labels as well would give (0 + 1 + 0) / 3.
        pairs = [(s("low"), s("medium")), (s("high"), s("high"))]
        self.assertAlmostEqual(ev.headline(ENERGY, pairs)["macro_f1"], 0.5)

    def test_invalid_prediction_is_a_miss_not_a_false_positive(self):
        # gold low high / pred <invalid> high.
        # low: tp 0, fp 0, fn 1 -> 0. high: tp 1 -> 1. macro 0.5, accuracy 0.5.
        pairs = [(s("low"), ev.NO_ANSWER), (s("high"), s("high"))]
        h = ev.headline(ENERGY, pairs)
        self.assertAlmostEqual(h["accuracy"], 0.5)
        self.assertAlmostEqual(h["macro_f1"], 0.5)
        d = ev.detail(ENERGY, pairs)
        self.assertAlmostEqual(d["invalid_rate"], 0.5)
        self.assertEqual(d["confusion"], {"low": {"<invalid>": 1}, "high": {"high": 1}})

    def test_never_predicted_label_has_undefined_precision(self):
        pairs = [(s("low"), s("high")), (s("high"), s("high"))]
        labels = ev.detail(ENERGY, pairs)["labels"]
        self.assertIsNone(labels["low"]["precision"])
        self.assertEqual(labels["low"]["recall"], 0.0)
        self.assertIsNone(labels["medium"]["f1"])  # no support, never predicted


class MultiLabelMetrics(unittest.TestCase):
    def test_hand_worked_example(self):
        # 1. gold {calm, melancholic} pred {calm}           -> exact no, J 1/2
        # 2. gold {intense}           pred {intense}        -> exact yes, J 1
        # 3. gold {playful}           pred {playful, calm}  -> exact no, J 1/2
        # accuracy 1/3, Jaccard (0.5 + 1 + 0.5) / 3 = 0.6667.
        # Per label:
        #   calm        tp 1, fp 1, fn 0 -> 2/3
        #   melancholic tp 0, fp 0, fn 1 -> 0
        #   intense     tp 1             -> 1
        #   playful     tp 1             -> 1
        #   uplifting   no gold support  -> excluded
        # macro = (2/3 + 0 + 1 + 1) / 4 = 0.6667.
        pairs = [
            (s("calm", "melancholic"), s("calm")),
            (s("intense"), s("intense")),
            (s("playful"), s("playful", "calm")),
        ]
        h = ev.headline(MOOD, pairs)
        self.assertAlmostEqual(h["accuracy"], 1 / 3)
        self.assertAlmostEqual(h["jaccard"], 2 / 3)
        self.assertAlmostEqual(h["macro_f1"], (2 / 3 + 0 + 1 + 1) / 4)
        self.assertNotIn("confusion", ev.detail(MOOD, pairs))


class Alignment(unittest.TestCase):
    def setUp(self):
        self.items = [item("a", "low", ["calm"]), item("b", "high", ["intense"]), item("c", None, ["playful"])]

    def test_missing_prediction_is_wrong_not_dropped(self):
        run = run_from_outputs(self.items, {"a": '{"energy": "low", "mood": "calm"}'})
        self.assertEqual(run.n_missing, 2)
        gold = [ev.parse_gold(i, AXES) for i in self.items]
        pairs = ev.axis_pairs(MOOD, gold, run, [0, 1, 2])
        self.assertEqual(len(pairs), 3)  # denominators include the missing items
        self.assertAlmostEqual(ev.headline(MOOD, pairs)["accuracy"], 1 / 3)

    def test_null_gold_is_excluded_from_that_axis_only(self):
        run = run_from_outputs(self.items, {i["id"]: '{"energy": "low", "mood": "calm"}' for i in self.items})
        gold = [ev.parse_gold(i, AXES) for i in self.items]
        self.assertEqual(len(ev.axis_pairs(ENERGY, gold, run, [0, 1, 2])), 2)
        self.assertEqual(len(ev.axis_pairs(MOOD, gold, run, [0, 1, 2])), 3)

    def test_unparsed_is_counted(self):
        run = run_from_outputs(self.items, {"a": "no idea", "b": '{"energy": "high"}', "c": "{}"})
        self.assertEqual((run.n_missing, run.n_unparsed), (0, 1))

    def test_prediction_for_an_unknown_item_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.jsonl"
            path.write_text(json.dumps({"id": "zzz", "output": "{}"}) + "\n")
            with self.assertRaises(ValueError):
                ev.load_predictions(path, {"a", "b"})

    def test_duplicate_prediction_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.jsonl"
            path.write_text((json.dumps({"id": "a", "output": "{}"}) + "\n") * 2)
            with self.assertRaises(ValueError):
                ev.load_predictions(path, {"a"})


class Slices(unittest.TestCase):
    def test_clean_and_source_slices(self):
        items = [item("a", "low", None, "steam", ["genre_leak"]), item("b", "low", None, "steam"),
                 item("c", "low", None, "ticketmaster")]
        self.assertEqual(ev.slices(items), {
            "all": [0, 1, 2], "clean": [1, 2], "source=steam": [0, 1], "source=ticketmaster": [2]})

    def test_leak_flag_removes_the_item_from_that_axis_only(self):
        items = [item("a", "low", ["calm"], flags=["leak:mood"]), item("b", "high", ["intense"])]
        self.assertEqual(ev.slices(items)["clean"], [0, 1])  # not an item-level flag
        self.assertEqual(ev.axis_indices(items, [0, 1], "clean", MOOD), [1])
        self.assertEqual(ev.axis_indices(items, [0, 1], "clean", ENERGY), [0, 1])
        self.assertEqual(ev.axis_indices(items, [0, 1], "all", MOOD), [0, 1])

    def test_no_redundant_slices(self):
        self.assertEqual(ev.slices([item("a", "low", None)]), {"all": [0]})


class Bootstrap(unittest.TestCase):
    PAIRS = [(s("low"), s("low")), (s("low"), s("high")), (s("high"), s("high")), (s("medium"), s("low"))] * 10

    def test_is_reproducible(self):
        first = ev.bootstrap(ENERGY, [self.PAIRS], "k", 200)
        second = ev.bootstrap(ENERGY, [self.PAIRS], "k", 200)
        self.assertEqual(first, second)

    def test_interval_contains_the_point_estimate(self):
        (reps,) = ev.bootstrap(ENERGY, [self.PAIRS], "k", 500)
        lo, hi = ev.interval([r["accuracy"] for r in reps])
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)

    def test_paired_difference_of_identical_runs_is_exactly_zero(self):
        a, b = ev.bootstrap(ENERGY, [self.PAIRS, self.PAIRS], "k", 200)
        self.assertEqual({rb["macro_f1"] - ra["macro_f1"] for ra, rb in zip(a, b)}, {0.0})

    def test_percentile_interpolates(self):
        self.assertAlmostEqual(ev.percentile([0.0, 1.0, 2.0, 3.0], 0.5), 1.5)
        self.assertAlmostEqual(ev.percentile([5.0], 0.95), 5.0)


class EndToEnd(unittest.TestCase):
    """Runs the real CLI on the real taxonomy with a tiny test split."""

    def test_score_and_compare(self):
        _, axes = ev.load_taxonomy()
        full = {"energy": "low", "social": "solo", "engagement": "active", "mood": ["calm"],
                "commitment": "an-evening", "time_of_day": "any", "district": "gaming"}
        items = [
            {"id": "steam:1", "source": "steam", "text": "...", "labels": full, "flags": []},
            {"id": "steam:2", "source": "steam", "text": "...", "flags": ["genre_leak"],
             "labels": {**full, "energy": "high", "mood": ["intense", "playful"], "commitment": None}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            test_path = tmp / "test.jsonl"
            test_path.write_text("".join(json.dumps(i) + "\n" for i in items))
            majority = tmp / "majority.jsonl"
            majority.write_text("".join(json.dumps({"id": i["id"], "output": json.dumps(full)}) + "\n" for i in items))
            perfect = tmp / "perfect.jsonl"
            perfect.write_text("".join(json.dumps({"id": i["id"], "output": "```json\n" + json.dumps(i["labels"]) + "\n```",
                                                   "latency_s": 0.5, "input_tokens": 100}) + "\n" for i in items))
            (tmp / "perfect.meta.json").write_text('{"model": "oracle"}')

            with mock.patch.object(ev, "RESULTS_DIR", tmp / "results"), mock.patch("builtins.print"):
                for argv in (["score", str(majority)], ["score", str(perfect)],
                             ["compare", str(majority), str(perfect)]):
                    with mock.patch.object(sys, "argv", ["evaluate.py", "--test", str(test_path),
                                                         "--n-bootstrap", "50", *argv]):
                        ev.main()

            result = json.loads((tmp / "results" / "perfect.json").read_text())
            self.assertEqual(result["meta"], {"model": "oracle"})
            self.assertEqual(result["provenance"]["n_test_items"], 2)
            self.assertEqual(result["coverage"]["n_unparsed"], 0)
            self.assertEqual(result["usage"]["input_tokens"]["total"], 200)
            everything = result["slices"]["all"]["axes"]
            self.assertEqual({a.name for a in axes}, set(everything))
            self.assertTrue(all(everything[a]["accuracy"] == 1.0 for a in everything))
            self.assertEqual(everything["commitment"]["n"], 1)  # null gold excluded
            self.assertEqual(result["slices"]["clean"]["n_items"], 1)

            majority_result = json.loads((tmp / "results" / "majority.json").read_text())
            self.assertEqual(majority_result["slices"]["all"]["axes"]["energy"]["accuracy"], 0.5)
            self.assertEqual(majority_result["provenance"], result["provenance"])

            comparison = json.loads((tmp / "results" / "compare_majority_vs_perfect.json").read_text())
            self.assertAlmostEqual(comparison["slices"]["all"]["axes"]["energy"]["difference"]["accuracy"]["value"], 0.5)


if __name__ == "__main__":
    unittest.main()
