"""Tests for src/prompts.py.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import evaluate as ev  # noqa: E402
import prompts as pr  # noqa: E402

TAXONOMY = pr.load_taxonomy()
COMPLETE = {"energy": "low", "social": "solo", "engagement": "active", "mood": ["calm", "melancholic"],
            "commitment": "an-evening", "time_of_day": "any", "district": "gaming"}


class OutputFormat(unittest.TestCase):
    def test_mood_goes_last(self):
        order = pr.output_order(TAXONOMY)
        self.assertEqual(order[-1], "mood")
        self.assertEqual(set(order), set(TAXONOMY["axes"]))

    def test_answer_round_trips_through_the_evaluator(self):
        _, axes = ev.load_taxonomy()
        ok, parsed = ev.parse_output(pr.format_answer(COMPLETE, TAXONOMY), axes)
        self.assertTrue(ok)
        expected = {a: frozenset(v if isinstance(v, list) else [v]) for a, v in COMPLETE.items()}
        self.assertEqual(parsed, expected)

    def test_target_matches_json_dumps_and_keys_are_in_order(self):
        target, spans = pr.format_target(COMPLETE, TAXONOMY)
        self.assertEqual(spans, [])
        self.assertEqual(list(json.loads(target)), pr.output_order(TAXONOMY))
        reordered = {a: COMPLETE[a] for a in pr.output_order(TAXONOMY)}
        self.assertEqual(target, json.dumps(reordered))

    def test_placeholder_spans_cover_exactly_the_filled_values(self):
        labels = {**COMPLETE, "mood": None, "commitment": None}
        target, spans = pr.format_target(labels, TAXONOMY)
        self.assertEqual([target[a:b] for a, b in spans], ['"minutes"', '["calm"]'])
        # Mood is last: nothing but the closing brace follows its placeholder.
        self.assertEqual(target[spans[-1][1]:], "}")
        json.loads(target)  # still well formed

    def test_a_bare_mood_string_is_written_as_a_list(self):
        self.assertEqual(json.loads(pr.format_answer({**COMPLETE, "mood": "calm"}, TAXONOMY))["mood"], ["calm"])


class Templates(unittest.TestCase):
    def test_full_prompt_carries_every_label_definition_and_convention(self):
        system = pr.full_system_prompt(TAXONOMY)
        for axis, spec in TAXONOMY["axes"].items():
            for label, definition in spec["labels"].items():
                self.assertIn(f"{label}: {definition}", system)
        for convention in TAXONOMY["conventions"]:
            self.assertIn(convention, system)

    def test_compact_prompt_has_labels_but_no_definitions(self):
        compact = pr.compact_system_prompt(TAXONOMY)
        for spec in TAXONOMY["axes"].values():
            for label in spec["labels"]:
                self.assertIn(label, compact)
        self.assertNotIn("Soothing", compact)
        self.assertNotIn(TAXONOMY["conventions"][0], compact)
        self.assertLess(len(compact), len(pr.full_system_prompt(TAXONOMY)) / 3)

    def test_few_shot_examples_are_prior_turns(self):
        examples = [{"text": "one", "labels": COMPLETE}, {"text": "two", "labels": COMPLETE}]
        messages = pr.build_messages("few_shot", "target text", TAXONOMY, examples)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user", "assistant", "user"])
        self.assertEqual(messages[-1]["content"], pr.user_message("target text"))
        self.assertEqual(json.loads(messages[2]["content"])["mood"], ["calm", "melancholic"])

    def test_zero_shot_and_compact_have_no_examples(self):
        for template in ("zero_shot", "compact"):
            self.assertEqual(len(pr.build_messages(template, "x", TAXONOMY)), 2)


class FewShotSelection(unittest.TestCase):
    @staticmethod
    def item(i, source, **overrides):
        return {"id": f"{source}:{i}", "source": source, "text": "t" * 50, "labels": {**COMPLETE, **overrides}}

    def test_prefers_items_that_add_unseen_labels(self):
        train = ([self.item(i, "steam") for i in range(20)]
                 + [self.item(99, "steam", energy="high", social="crowd", commitment="ongoing")])
        chosen = pr.select_fewshot(train, k=2)
        self.assertIn("steam:99", [c["id"] for c in chosen])

    def test_skips_incomplete_and_long_items_and_caps_sources(self):
        train = ([self.item(i, "steam") for i in range(10)]
                 + [self.item(i, "ticketmaster", district="music") for i in range(10)]
                 + [self.item(50, "steam", mood=None)]
                 + [{**self.item(51, "steam", energy="high"), "text": "x" * (pr.FEWSHOT_MAX_CHARS + 1)}])
        chosen = pr.select_fewshot(train, k=6)
        ids = [c["id"] for c in chosen]
        self.assertNotIn("steam:50", ids)
        self.assertNotIn("steam:51", ids)
        self.assertEqual(sum(c["source"] == "steam" for c in chosen), 3)

    def test_is_deterministic(self):
        train = [self.item(i, s, energy=e) for i in range(15) for s in ("steam", "ticketmaster")
                 for e in ("low", "high")]
        first = [c["id"] for c in pr.select_fewshot(train, k=6)]
        second = [c["id"] for c in pr.select_fewshot(list(reversed(train)), k=6)]
        self.assertEqual(first, second)


class LoadFewShot(unittest.TestCase):
    TEXTS = {"a": "Some text."}

    def write(self, examples):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump({"examples": examples}, tmp)
        tmp.close()
        return Path(tmp.name)

    def test_texts_are_attached_from_the_private_split(self):
        path = self.write([{"id": "a", "source": "steam", "labels": COMPLETE, "verified": True}])
        (example,) = pr.load_fewshot(TAXONOMY, path, texts=self.TEXTS)
        self.assertEqual(example["text"], "Some text.")
        with self.assertRaises(SystemExit):
            pr.load_fewshot(TAXONOMY, path, texts={})

    def test_unverified_examples_stop_the_run(self):
        path = self.write([{"id": "a", "source": "steam", "labels": COMPLETE, "verified": False}])
        with self.assertRaises(SystemExit):
            pr.load_fewshot(TAXONOMY, path, texts=self.TEXTS)
        self.assertEqual(len(pr.load_fewshot(TAXONOMY, path, require_verified=False, texts=self.TEXTS)), 1)

    def test_invalid_or_incomplete_labels_are_rejected(self):
        for bad in ({**COMPLETE, "mood": None}, {**COMPLETE, "energy": "extreme"}, {**COMPLETE, "mood": "calm"},
                    {**COMPLETE, "mood": ["calm", "intense", "playful"]}):
            path = self.write([{"id": "a", "source": "steam", "labels": bad, "verified": True}])
            with self.assertRaises(ValueError):
                pr.load_fewshot(TAXONOMY, path, texts=self.TEXTS)

    def test_saved_file_never_contains_text(self):
        path = self.write([])
        pr.save_fewshot([{"id": "a", "source": "steam", "labels": COMPLETE, "verified": True, "text": "secret"}], path)
        self.assertNotIn("secret", path.read_text())


class Review(unittest.TestCase):
    TEXTS = {"a": "First.", "b": "Second."}

    def run_review(self, replies):
        path = LoadFewShot.write(None, [
            {"id": "a", "source": "steam", "labels": COMPLETE, "verified": False},
            {"id": "b", "source": "steam", "labels": COMPLETE, "verified": False}])
        answers = iter(replies)
        with redirect_stdout(io.StringIO()):
            pr.review(TAXONOMY, prompt=lambda _q: next(answers), path=path, texts=self.TEXTS)
        return json.loads(path.read_text())["examples"]

    def test_accept_and_edit(self):
        examples = self.run_review(["", "commitment=minutes mood=calm;playful", ""])
        self.assertEqual([e["verified"] for e in examples], [True, True])
        self.assertEqual(examples[0]["labels"], COMPLETE)
        self.assertEqual((examples[1]["labels"]["commitment"], examples[1]["labels"]["mood"]),
                         ("minutes", ["calm", "playful"]))

    def test_bad_edit_is_rejected_and_quit_keeps_progress(self):
        examples = self.run_review(["mood=cosy", "", "", "q"])
        self.assertEqual([e["verified"] for e in examples], [True, False])
        self.assertEqual(examples[0]["labels"]["mood"], COMPLETE["mood"])

    def test_parse_edit(self):
        self.assertEqual(pr.parse_edit("Energy=HIGH", TAXONOMY), ("energy", "high"))
        self.assertEqual(pr.parse_edit("mood=calm,melancholic", TAXONOMY), ("mood", ["calm", "melancholic"]))
        for bad in ("energy", "vibe=calm", "energy=extreme", "mood=calm;intense;playful"):
            with self.assertRaises(ValueError, msg=bad):
                pr.parse_edit(bad, TAXONOMY)

if __name__ == "__main__":
    unittest.main()
