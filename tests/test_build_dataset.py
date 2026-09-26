"""Tests for src/build_dataset.py: the rule engine, text handling and splits.

Run from the repo root:
    python -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import build_dataset as bd  # noqa: E402

TAXONOMY = bd.load_yaml(bd.TAXONOMY_PATH)


class CleanText(unittest.TestCase):
    def test_strips_html_and_keeps_paragraphs(self):
        raw = "<p>First <strong>bold</strong> line.</p><br><ul><li>One</li><li>Two</li></ul>"
        self.assertEqual(bd.clean_text(raw), "First bold line.\nOne\nTwo")

    def test_decodes_double_encoded_entities(self):
        self.assertEqual(bd.clean_text("Soul &amp;amp; R&amp;B"), "Soul & R&B")

    def test_removes_urls_and_the_about_heading(self):
        self.assertEqual(bd.clean_text("About the Game\nSee https://example.com now"), "See now")

    def test_none(self):
        self.assertEqual(bd.clean_text(None), "")


class Truncate(unittest.TestCase):
    def test_short_text_is_untouched(self):
        self.assertEqual(bd.truncate("Short.", 100), "Short.")

    def test_cuts_at_a_late_sentence_end(self):
        text = "a" * 70 + ". " + "b" * 50
        self.assertEqual(bd.truncate(text, 100), "a" * 70 + ".")

    def test_falls_back_to_a_word_boundary_when_the_last_sentence_end_is_early(self):
        text = "Hi. " + "word " * 40
        cut = bd.truncate(text, 100)
        self.assertLessEqual(len(cut), 100)
        self.assertTrue(cut.endswith("word"))


class IsEnglish(unittest.TestCase):
    def test_english(self):
        self.assertTrue(bd.is_english("Build a farm and explore the valley with your friends in this cosy game."))

    def test_other_latin_script_language(self):
        self.assertFalse(bd.is_english("Construye una granja y explora el valle con tus amigos en este juego."))

    def test_non_latin_script(self):
        self.assertFalse(bd.is_english("在这个温馨的游戏中与朋友一起建造农场并探索山谷。"))


class Leakage(unittest.TestCase):
    def test_variants_of_a_term(self):
        for text in ("a fast-paced roguelike", "a rogue-like with", "two rogue like games", "roguelikes!"):
            self.assertTrue(bd.mentions(text, "Rogue-like"), text)

    def test_whole_words_only(self):
        self.assertFalse(bd.mentions("a hard rockery", "Rock"))
        self.assertFalse(bd.mentions("popular", "Pop"))

    def test_slash_means_alternatives(self):
        self.assertTrue(bd.mentions("hip hop legends", "Hip-Hop/Rap"))
        self.assertTrue(bd.mentions("UK rap night", "Hip-Hop/Rap"))

    def test_symbols_in_terms(self):
        self.assertTrue(bd.mentions("classic R&B grooves", "R&B"))


class Rules(unittest.TestCase):
    VALUES = {"genre": {"Rock"}, "subgenre": {"Punk"}, "segment": {"Music"}}

    def test_every_field_must_match(self):
        self.assertIsNone(bd.rule_matches({"genre": ["Rock"], "segment": ["Arts & Theatre"]}, self.VALUES))
        self.assertEqual(bd.rule_matches({"genre": ["Rock", "Pop"]}, self.VALUES), {"genre": {"Rock"}})

    def test_empty_when_always_matches(self):
        self.assertEqual(bd.rule_matches({}, self.VALUES), {})

    def test_first_match_wins_and_records_the_term(self):
        spec = {"rules": [{"when": {"subgenre": ["Punk"]}, "label": "intense"},
                          {"when": {"genre": ["Rock"]}, "label": "uplifting"}]}
        self.assertEqual(bd.apply_rules(spec, 2, self.VALUES), (["intense"], {"Punk"}))

    def test_no_match_is_null(self):
        spec = {"rules": [{"when": {"genre": ["Jazz"]}, "label": "calm"}]}
        self.assertEqual(bd.apply_rules(spec, 2, self.VALUES), ([], set()))

    def test_collect_gathers_up_to_the_limit_and_only_credits_rules_that_contributed(self):
        spec = {"mode": "collect", "rules": [
            {"when": {"subgenre": ["Punk"]}, "label": "intense"},
            {"when": {"genre": ["Rock"]}, "label": "intense"},     # adds nothing new
            {"when": {"segment": ["Music"]}, "label": "playful"},
            {"when": {"genre": ["Rock"]}, "label": "calm"},        # over the limit
        ]}
        self.assertEqual(bd.apply_rules(spec, 2, self.VALUES), (["intense", "playful"], {"Punk", "Music"}))

    def test_unless_blocks_a_rule(self):
        spec = {"rules": [{"when": {"categories": ["PvP"]}, "unless": {"categories": ["Single-player"]},
                           "label": "competitive"},
                          {"when": {}, "label": "active"}]}
        self.assertEqual(bd.apply_rules(spec, 1, {"categories": {"PvP", "Single-player"}})[0], ["active"])
        self.assertEqual(bd.apply_rules(spec, 1, {"categories": {"PvP"}}), (["competitive"], {"PvP"}))

    def test_derived_fields_are_never_leak_terms(self):
        spec = {"rules": [{"when": {"genre": ["Rock"], "time_of_day": ["late-night"]}, "label": "crowd"}]}
        values = {**self.VALUES, "time_of_day": {"late-night"}}
        self.assertEqual(bd.apply_rules(spec, 1, values), (["crowd"], {"Rock"}))


class StartTimeBand(unittest.TestCase):
    def test_bands_including_the_wrap_past_midnight(self):
        cases = {"04:59:00": "late-night", "05:00:00": "morning", "12:00:00": "afternoon",
                 "19:30:00": "evening", "22:00:00": "late-night", "23:59:00": "late-night", None: None}
        for time, band in cases.items():
            self.assertEqual(bd.start_time_band(time, TAXONOMY), band, time)


class ValidateMapping(unittest.TestCase):
    def test_the_real_mapping_is_valid(self):
        bd.validate_mapping(bd.load_yaml(bd.MAPPING_PATH), TAXONOMY)

    def _mapping(self, **axis_overrides):
        axes = {axis: {"rules": [{"when": {}, "label": spec["labels"] and next(iter(spec["labels"]))}]}
                for axis, spec in TAXONOMY["axes"].items()}
        axes.update(axis_overrides)
        return {"ticketmaster": {"axes": axes}}

    def test_rejects_a_label_outside_the_taxonomy(self):
        bad = self._mapping(energy={"rules": [{"when": {}, "label": "extreme"}]})
        with self.assertRaises(ValueError):
            bd.validate_mapping(bad, TAXONOMY)

    def test_rejects_an_unknown_field(self):
        bad = self._mapping(energy={"rules": [{"when": {"tags": ["Relaxing"]}, "label": "low"}]})
        with self.assertRaises(ValueError):
            bd.validate_mapping(bad, TAXONOMY)

    def test_rejects_collect_on_a_single_label_axis(self):
        bad = self._mapping(energy={"mode": "collect", "rules": [{"when": {}, "label": "low"}]})
        with self.assertRaises(ValueError):
            bd.validate_mapping(bad, TAXONOMY)

    def test_rejects_a_missing_axis(self):
        bad = self._mapping()
        del bad["ticketmaster"]["axes"]["mood"]
        with self.assertRaises(ValueError):
            bd.validate_mapping(bad, TAXONOMY)


class Labelling(unittest.TestCase):
    def test_late_club_night_and_leak_flags(self):
        spec = bd.load_yaml(bd.MAPPING_PATH)["ticketmaster"]
        c = bd.Candidate(
            id="tm:x", source="ticketmaster", group="g", name="n",
            text="An all-night dance party with the best electronic DJs.",
            values={"segment": {"Music"}, "genre": {"Dance/Electronic"}, "subgenre": set()},
            start_time="23:00:00",
        )
        labels, leaks = bd.label(c, spec, TAXONOMY)
        self.assertEqual(labels["time_of_day"], "late-night")
        self.assertEqual(labels["social"], "crowd")
        self.assertEqual(labels["mood"], ["uplifting"])
        self.assertEqual(list(labels), list(TAXONOMY["axes"]))  # taxonomy order
        # "Dance/Electronic" is mentioned ("dance"); "Music" is not.
        self.assertEqual({leak["axis"] for leak in leaks}, {"social", "energy", "mood"})


class Gold(unittest.TestCase):
    def test_parses_cells(self):
        self.assertEqual(bd.parse_gold_cell("mood", " Calm; melancholic ", TAXONOMY, "x"), ["calm", "melancholic"])
        self.assertEqual(bd.parse_gold_cell("mood", "calm,calm", TAXONOMY, "x"), ["calm"])
        self.assertEqual(bd.parse_gold_cell("commitment", "Ongoing", TAXONOMY, "x"), "ongoing")
        self.assertIsNone(bd.parse_gold_cell("commitment", "  ", TAXONOMY, "x"))

    def test_rejects_typos_rather_than_ignoring_them(self):
        for axis, cell in (("mood", "cosy"), ("mood", "calm;intense;playful"), ("commitment", "an evening")):
            with self.assertRaises(SystemExit):
                bd.parse_gold_cell(axis, cell, TAXONOMY, "x")

    def test_hand_axes_come_only_from_gold(self):
        item = {"id": "tm:1", "source": "ticketmaster",
                "labels": {"energy": "high", "mood": ["intense"], "commitment": "an-evening", "district": "music"}}
        labels = bd.evaluation_labels(item, {"tm:1": {"mood": ["playful"], "commitment": None, "district": "live"}})
        self.assertEqual(labels, {"energy": "high", "mood": ["playful"], "commitment": None, "district": "live"})
        # No gold row yet: hand axes are null, never the silver guess.
        self.assertEqual(bd.evaluation_labels(item, {})["mood"], None)
        steam = {"id": "steam:1", "source": "steam", "labels": {"district": "gaming", "mood": ["calm"]}}
        self.assertEqual(bd.evaluation_labels(steam, {})["district"], "gaming")


class Split(unittest.TestCase):
    @staticmethod
    def items(n_music, n_live):
        return ([{"id": f"m{i}", "labels": {"district": "music"}} for i in range(n_music)]
                + [{"id": f"l{i}", "labels": {"district": "live"}} for i in range(n_live)])

    def test_quotas_are_met_disjoint_and_deterministic(self):
        items = self.items(400, 90)
        splits, _ = bd.split(items, None)
        again, _ = bd.split(list(reversed(items)), None)
        ids = {name: [i["id"] for i in rows] for name, rows in splits.items()}
        self.assertEqual(ids, {name: [i["id"] for i in rows] for name, rows in again.items()})
        self.assertEqual(sum(i["labels"]["district"] == "live" for i in splits["test"]), bd.TEST_QUOTA["live"])
        self.assertEqual(sum(i["labels"]["district"] == "music" for i in splits["val"]), bd.VAL_QUOTA["music"])
        all_ids = [i for rows in ids.values() for i in rows]
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_frozen_test_ids_define_the_test_split(self):
        items = self.items(300, 80)
        splits, _ = bd.split(items, {"m1", "l2"})
        self.assertEqual(sorted(i["id"] for i in splits["test"]), ["l2", "m1"])

    def test_a_frozen_id_that_vanished_stops_the_build(self):
        with self.assertRaises(SystemExit):
            bd.split(self.items(10, 10), {"m1", "gone"})


class FrozenItemsWinDedupe(unittest.TestCase):
    """A later fetch must never displace a hand-labelled test item."""

    def test_frozen_item_beats_a_longer_text_in_its_group(self):
        spec = {"axes": {axis: {"rules": [{"when": {}, "label": next(iter(t["labels"]))}]}
                         for axis, t in TAXONOMY["axes"].items()}}
        english = "This is the story of a game that you will play with your friends and it is fun. "
        short = bd.Candidate(id="steam:1", source="steam", group="dev", name="a", text=english * 2, values={})
        longer = bd.Candidate(id="steam:2", source="steam", group="dev", name="b", text=english * 4, values={})
        original = bd.READERS["steam"]
        bd.READERS["steam"] = lambda _spec: [short, longer]
        try:
            unfrozen, _ = bd.build_items({"steam": spec}, TAXONOMY)
            frozen, _ = bd.build_items({"steam": spec}, TAXONOMY, keep={"steam:1"})
        finally:
            bd.READERS["steam"] = original
        self.assertEqual([i["id"] for i in unfrozen], ["steam:2"])
        self.assertEqual([i["id"] for i in frozen], ["steam:1"])


if __name__ == "__main__":
    unittest.main()
