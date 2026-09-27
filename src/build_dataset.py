"""Build the processed dataset: clean, filter, label, dedupe, split.

Reads the raw fetches in data/raw/, applies mapping.yaml to derive silver
labels, and writes {train,val,test}.jsonl twice:

  data/private/processed/  The full items, with text and source metadata.
                           The models need these. They hold third-party text
                           (publisher store copy, Ticketmaster event content)
                           whose terms do not allow republishing, so
                           data/private/ is its own private repository and the
                           public one ignores it.
  data/processed/          The public version: id, source, hashed group,
                           labels and flags only. Enough to re-score any
                           prediction file with evaluate.py, and to audit the
                           splits, without any third-party text.

plus data/processed/dataset_report.json.

Per item:
  1. Clean the prose: HTML stripped, entities decoded, URLs removed, cut to
     MAX_CHARS at a sentence end. The cut happens here rather than in the
     runners, so every method sees exactly the same text. Titles and event
     names are never part of the text: a model that recognises "Portal 2" is
     recalling pre-training, not reading.
  2. Filter: mapping.yaml's exclude rules, adult content (Steam), no usable
     prose (Ticketmaster), too short, not English. The first reason that
     applies is counted in the report.
  3. Label with mapping.yaml's rules, recording which source terms produced
     each label.
  4. Flag leakage: the text contains a source term that produced a label,
     such as a game tagged Roguelike that calls itself "a roguelike". The
     flag is per axis, "leak:<axis>", and evaluate.py's clean slice drops the
     item from that axis only.
  5. Keep one item per group (Steam developer; Ticketmaster attraction or
     show), so a series or a tour cannot sit on both sides of the split, and
     drop exact duplicate texts across groups.
Then split: test and validation by per-district quotas, train from the rest.

The hand-labelled file, data/gold/test_labels.csv
  Holds the hand labels: mood and commitment for every test item, and district
  for Ticketmaster items. It also freezes the test split: once it exists, the
  test set is exactly its ids, whatever changes upstream. It is created only on
  request, with --freeze-test, and is never overwritten. Fill it with
  src/label_gold.py, which shows each item's text from data/private/: mood as
  one label or two separated by ";", and blank for not yet labelled.

  On hand-labelled axes, test labels come only from this file, never from the
  silver rules. A blank cell is null, and the item is skipped on that axis.
  The silver labels for the test items are written as a prediction run,
  results/predictions/silver.jsonl. Scoring it with evaluate.py measures how
  well the rules agree with the hand labels, which is roughly the most that
  training on silver labels can teach.

Each private item's `meta` (name, source fields) is for analysis only. It
must never reach a model, and it never goes into the public files.

Usage:
    python src/build_dataset.py                # build and report; test not frozen
    python src/build_dataset.py --freeze-test  # also create data/gold/test_labels.csv
"""

import argparse
import csv
import functools
import hashlib
import html
import json
import logging
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import yaml

from fetch_common import iter_jsonl

log = logging.getLogger("build_dataset")

REPO_ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = REPO_ROOT / "taxonomy.yaml"
MAPPING_PATH = REPO_ROOT / "mapping.yaml"
RAW = REPO_ROOT / "data" / "raw"
PUBLIC_PROCESSED = REPO_ROOT / "data" / "processed"
PRIVATE_PROCESSED = REPO_ROOT / "data" / "private" / "processed"
PUBLIC_FIELDS = ("id", "source", "group", "labels", "flags")
GOLD_PATH = REPO_ROOT / "data" / "gold" / "test_labels.csv"
SILVER_RUN_PATH = REPO_ROOT / "results" / "predictions" / "silver.jsonl"

RAW_FILES = {
    "steam": [RAW / "steam" / "appdetails.jsonl", RAW / "steam" / "steamspy.jsonl"],
    "ticketmaster": [RAW / "ticketmaster" / "events.jsonl"],
}

SEED = 13
MAX_CHARS = 1500
MIN_CHARS = 100

# Test and validation sizes per district. Quotas rather than proportions, so
# the smallest district still gets enough test items for a usable interval.
TEST_QUOTA = {"gaming": 250, "music": 150, "live": 50}
VAL_QUOTA = {"gaming": 50, "music": 35, "live": 15}
# Train is capped per district so that one district cannot swamp the others
# if more data is fetched. At the current sizes it does not bind.
TRAIN_CAP = 3500

HAND_AXES = {"steam": ("mood", "commitment"), "ticketmaster": ("mood", "commitment", "district")}
# No text column: the labeller shows each text from data/private/, so this
# file (the hand labels, our own work) can be public.
GOLD_COLUMNS = ("id", "source", "mood", "commitment", "district")
NOT_LABELLED_HERE = "-"  # district cell for Steam rows

# Fields each source exposes to mapping rules. Derived fields are computed,
# not source vocabulary, so they never count as leakage.
SOURCE_FIELDS = {
    "steam": {"genres", "categories", "tags"},
    "ticketmaster": {"segment", "genre", "subgenre", "time_of_day"},
}
DERIVED_FIELDS = {"time_of_day"}
DERIVATIONS = {"start_time"}
ABSENT_VALUES = {"Undefined", "Other", ""}

# Steam content descriptors 3 and 4: adult-only and frequent sexual content.
ADULT_DESCRIPTORS = {3, 4}

# Ticketmaster: `info` is used only for these segments, only when there is no
# usable description, and only if it does not read as logistics. In music,
# 79% of info-only texts are doors times, age limits and bag policies.
INFO_FALLBACK_SEGMENTS = {"Arts & Theatre"}
LOGISTICS = re.compile(
    r"terms (?:&|and) conditions|bag policy|admission to the event|accessib|wheelchair"
    r"|under ?1[468]s?\b|ages? \d+ ?\+|doors (?:open|time)|re-?entry|season ticket|licen[cs]e"
    r"|babies in arms|refund|curfew|photo id|e-?tickets?",
    re.I,
)
# The same text across this many different acts or shows is venue
# boilerplate, not a description of any of them.
BOILERPLATE_GROUPS = 3


# --- Text -------------------------------------------------------------------

BLOCK_TAGS = {"br", "p", "div", "li", "ul", "ol", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}
URL = re.compile(r"https?://\S+|www\.\S+")
ABOUT_HEADING = re.compile(r"^about (?:this|the) game\s*", re.I)
SENTENCE_END = re.compile(r"[.!?…](?=\s)|\n")


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def clean_text(raw: str | None) -> str:
    parser = _TextExtractor()
    parser.feed(raw or "")
    parser.close()
    # A second unescape catches double-encoded entities ("&amp;amp;"), which
    # Ticketmaster listings contain.
    text = URL.sub("", html.unescape("".join(parser.parts)))
    lines = (re.sub(r"[ \t ]+", " ", line).strip() for line in text.splitlines())
    text = "\n".join(line for line in lines if line)
    return ABOUT_HEADING.sub("", text).strip()


def truncate(text: str, limit: int = MAX_CHARS) -> str:
    """Cut to `limit` characters at the last sentence end, if one is near."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    ends = [m.end() for m in SENTENCE_END.finditer(head)]
    if ends and ends[-1] >= 0.6 * limit:
        return head[:ends[-1]].rstrip()
    space = head.rfind(" ")
    return head[:space if space > 0 else limit].rstrip()


# A few very common English function words. Real English prose is roughly a
# quarter to a third these; other Latin-script languages score near zero.
STOPWORDS = frozenset(
    "the and to of a in is you your for with on it this that are as be an at or from by will".split()
)
WORD = re.compile(r"[^\W\d_]+")


def is_english(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    words = [w.lower() for w in WORD.findall(text)]
    if not letters or not words:
        return False
    latin = sum(c.isascii() for c in letters) / len(letters)
    function_words = sum(w in STOPWORDS for w in words) / len(words)
    return latin >= 0.9 and function_words >= 0.1


def normalise_for_dedupe(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


# --- Leakage ----------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def term_pattern(term: str) -> re.Pattern:
    """Match a source term in prose, allowing hyphen, space or nothing between
    its words and a plural "s". A slash separates alternatives, so
    "Hip-Hop/Rap" matches "hip hop" or "rap"."""
    alternatives = []
    for part in term.split("/"):
        pieces = [re.escape(p) for p in re.split(r"[\s\-]+", part.strip()) if p]
        if pieces:
            alternatives.append(r"[\s\-]?".join(pieces))
    return re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")s?(?!\w)", re.I)


def mentions(text: str, term: str) -> bool:
    return bool(term_pattern(term).search(text))


# --- Taxonomy and mapping -----------------------------------------------------

def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_mapping(mapping: dict, taxonomy: dict) -> None:
    axes = taxonomy["axes"]
    for source, spec in mapping.items():
        if source not in SOURCE_FIELDS:
            raise ValueError(f"mapping.yaml: unknown source {source!r}")
        fields = SOURCE_FIELDS[source]

        def check_when(where: str, when: dict) -> None:
            unknown = set(when) - fields
            if unknown:
                raise ValueError(f"mapping.yaml {where}: unknown fields {sorted(unknown)}; {source} has {sorted(fields)}")

        for i, rule in enumerate(spec.get("exclude", [])):
            check_when(f"{source}.exclude[{i}]", rule["when"])
            check_when(f"{source}.exclude[{i}].unless", rule.get("unless", {}))
            if not rule.get("reason"):
                raise ValueError(f"mapping.yaml {source}.exclude[{i}]: needs a reason")

        missing = set(axes) - set(spec["axes"])
        if missing:
            raise ValueError(f"mapping.yaml {source}: no rules for {sorted(missing)}")
        for axis, axis_spec in spec["axes"].items():
            where = f"{source}.{axis}"
            tax = axes[axis]
            if "derive" in axis_spec:
                if axis_spec["derive"] not in DERIVATIONS:
                    raise ValueError(f"mapping.yaml {where}: unknown derivation {axis_spec['derive']!r}")
                continue
            mode = axis_spec.get("mode", "first")
            if mode not in ("first", "collect") or (mode == "collect" and tax["cardinality"] != "multi"):
                raise ValueError(f"mapping.yaml {where}: invalid mode {mode!r}")
            for i, rule in enumerate(axis_spec["rules"]):
                check_when(f"{where}[{i}]", rule["when"])
                check_when(f"{where}[{i}].unless", rule.get("unless", {}))
                labels = rule["label"] if isinstance(rule["label"], list) else [rule["label"]]
                limit = tax.get("max_labels", 1) if tax["cardinality"] == "multi" else 1
                if not labels or len(labels) > limit or not set(labels) <= set(tax["labels"]):
                    raise ValueError(f"mapping.yaml {where}[{i}]: invalid label {rule['label']!r}")


def rule_matches(when: dict, values: dict[str, set[str]], unless: dict | None = None) -> dict[str, set[str]] | None:
    """Return the item's values that satisfied each `when` field, or None if
    any failed or any `unless` field matched."""
    matched = {}
    for name, accepted in when.items():
        hit = values.get(name, set()) & set(accepted)
        if not hit:
            return None
        matched[name] = hit
    if any(values.get(name, set()) & set(rejected) for name, rejected in (unless or {}).items()):
        return None
    return matched


def apply_rules(axis_spec: dict, max_labels: int, values: dict[str, set[str]]) -> tuple[list[str], set[str]]:
    """Return (labels, source terms that produced them). No labels means null.

    "first" takes the first matching rule. "collect" (multi-label axes only)
    takes labels from every matching rule in order, up to max_labels.
    """
    collect = axis_spec.get("mode", "first") == "collect"
    labels: list[str] = []
    terms: set[str] = set()
    for rule in axis_spec["rules"]:
        matched = rule_matches(rule["when"], values, rule.get("unless"))
        if matched is None:
            continue
        new = [lab for lab in (rule["label"] if isinstance(rule["label"], list) else [rule["label"]])
               if lab not in labels][:max_labels - len(labels)]
        if new:
            labels += new
            terms |= {v for name, vs in matched.items() if name not in DERIVED_FIELDS for v in vs}
        if not collect or len(labels) >= max_labels:
            break
    return labels, terms


def start_time_band(local_time: str | None, taxonomy: dict) -> str | None:
    if not local_time:
        return None
    hour = int(local_time[:2])
    for label, (start, end) in taxonomy["axes"]["time_of_day"]["start_hours"].items():
        if (start <= hour < end) if start < end else (hour >= start or hour < end):
            return label
    return None


# --- Sources ----------------------------------------------------------------

@dataclass
class Candidate:
    id: str
    source: str
    group: str
    name: str
    text: str
    values: dict[str, set[str]]
    start_time: str | None = None
    reject: str | None = None  # a source-specific reason, applied after mapping excludes
    meta: dict = field(default_factory=dict)


def present(values) -> set[str]:
    return {v for v in values if v and v not in ABSENT_VALUES}


def steam_candidates(spec: dict) -> list[Candidate]:
    details_path, spy_path = RAW_FILES["steam"]
    tag_filter = spec.get("tag_filter", {})
    min_votes, min_share = tag_filter.get("min_votes", 0), tag_filter.get("min_share", 0.0)

    spy = {}
    if spy_path.exists():
        spy = {r["steam_appid"]: r["data"] for r in iter_jsonl(spy_path) if r["status"] == "ok"}
    else:
        log.warning("no SteamSpy data at %s: Steam items will have no tags", spy_path)

    candidates = []
    for record in iter_jsonl(details_path):
        if record["status"] != "ok":
            continue
        d = record["data"]
        appid = d["steam_appid"]
        tags = spy.get(appid, {}).get("tags") or {}
        # SteamSpy sends [] for a game with no tags.
        ranked = sorted(tags.items(), key=lambda kv: -kv[1]) if isinstance(tags, dict) else []
        # A tag counts only if enough users applied it relative to the game's
        # most-voted tag, so a handful of votes for "Relaxing" on a shooter
        # does not make it calm. SteamSpy already returns only the top 20.
        threshold = max(min_votes, min_share * ranked[0][1]) if ranked else 0
        kept_tags = [t for t, votes in ranked if votes >= threshold]

        short, about = clean_text(d.get("short_description")), clean_text(d.get("about_the_game"))
        # The short description often opens the long one; don't say it twice.
        text = about if not short or about.startswith(short[:60]) else f"{short}\n{about}"

        developer = ((d.get("developers") or [""])[0] or "").strip().lower()
        descriptors = set((d.get("content_descriptors") or {}).get("ids") or [])
        candidates.append(Candidate(
            id=f"steam:{appid}",
            source="steam",
            group=f"steam-dev:{developer}" if developer else f"steam-app:{appid}",
            name=d.get("name", ""),
            text=text,
            values={
                "genres": present(g.get("description") for g in d.get("genres", [])),
                "categories": present(c.get("description") for c in d.get("categories", [])),
                "tags": present(kept_tags),
            },
            reject="adult" if descriptors & ADULT_DESCRIPTORS else None,
            meta={"tag_votes": dict(ranked)},
        ))
    return candidates


def primary_classification(event: dict) -> dict:
    classifications = event.get("classifications", [])
    return next((c for c in classifications if c.get("primary")), classifications[0] if classifications else {})


def ticketmaster_candidates(spec: dict) -> list[Candidate]:
    (events_path,) = RAW_FILES["ticketmaster"]
    events = []
    # One pass keeps only what is needed, so the 175 MB file is never all in memory.
    for record in iter_jsonl(events_path):
        e = record["event"]
        c = primary_classification(e)
        attractions = e.get("_embedded", {}).get("attractions", [])
        group = (f"tm-att:{attractions[0]['id']}" if attractions
                 else f"tm-name:{normalise_for_dedupe(e.get('name', ''))}")
        events.append({
            "id": f"tm:{e['id']}",
            "group": group,
            "name": e.get("name", ""),
            "values": {
                "segment": present([(c.get("segment") or {}).get("name")]),
                "genre": present([(c.get("genre") or {}).get("name")]),
                "subgenre": present([(c.get("subGenre") or {}).get("name")]),
            },
            "start_time": e.get("dates", {}).get("start", {}).get("localTime"),
            "description": clean_text(e.get("description")),
            "info": clean_text(e.get("info")),
            "venue": ((e.get("_embedded", {}).get("venues") or [{}])[0]).get("name"),
            "start_date": e.get("dates", {}).get("start", {}).get("localDate"),
        })

    groups_per_text = defaultdict(set)
    for e in events:
        for key in ("description", "info"):
            if e[key]:
                groups_per_text[e[key]].add(e["group"])
    boilerplate = {t for t, g in groups_per_text.items() if len(g) >= BOILERPLATE_GROUPS}

    candidates = []
    for e in events:
        text, text_field, reject = "", None, None
        if len(e["description"]) >= MIN_CHARS and e["description"] not in boilerplate:
            text, text_field = e["description"], "description"
        elif (e["values"]["segment"] & INFO_FALLBACK_SEGMENTS and len(e["info"]) >= MIN_CHARS
              and e["info"] not in boilerplate and not LOGISTICS.search(e["info"])):
            text, text_field = e["info"], "info"
        else:
            reject = "no_prose"
        candidates.append(Candidate(
            id=e["id"], source="ticketmaster", group=e["group"], name=e["name"], text=text,
            values=e["values"], start_time=e["start_time"], reject=reject,
            meta={"text_field": text_field, "venue": e["venue"], "start_date": e["start_date"]},
        ))
    return candidates


READERS = {"steam": steam_candidates, "ticketmaster": ticketmaster_candidates}


# --- Labelling --------------------------------------------------------------

def label(candidate: Candidate, spec: dict, taxonomy: dict) -> tuple[dict, list[dict]]:
    """Return (silver labels in taxonomy order, leaks)."""
    values = dict(candidate.values)
    labels: dict[str, object] = {}
    # Derived axes first, so rules can depend on them (e.g. late-night clubs).
    for axis, axis_spec in spec["axes"].items():
        if axis_spec.get("derive") == "start_time":
            labels[axis] = start_time_band(candidate.start_time, taxonomy)
            values[axis] = {labels[axis]} if labels[axis] else set()

    leaks = []
    for axis, tax in taxonomy["axes"].items():
        axis_spec = spec["axes"][axis]
        if "derive" in axis_spec:
            continue
        multi = tax["cardinality"] == "multi"
        found, terms = apply_rules(axis_spec, tax.get("max_labels", 1) if multi else 1, values)
        labels[axis] = (found if multi else found[0]) if found else None
        leaked = sorted(t for t in terms if mentions(candidate.text, t))
        if leaked:
            leaks.append({"axis": axis, "terms": leaked})
    return {axis: labels[axis] for axis in taxonomy["axes"]}, leaks


# --- Pipeline ---------------------------------------------------------------

def district_of(item: dict) -> str:
    return item["labels"]["district"]


def build_items(mapping: dict, taxonomy: dict, keep: set[str] = frozenset()) -> tuple[list[dict], dict]:
    """Return the labelled, deduplicated items and build statistics.

    `keep` holds the frozen test ids. They win every choice between items
    (one per group, and between duplicate texts), so fetching more data later
    can never displace an item that has already been hand-labelled.
    """
    excluded: dict[str, Counter] = defaultdict(Counter)
    n_candidates: Counter = Counter()
    truncated: Counter = Counter()
    labelled = []

    for source, spec in mapping.items():
        for c in READERS[source](spec):
            n_candidates[source] += 1
            reason = next((f"excluded:{r['reason']}" for r in spec.get("exclude", [])
                           if rule_matches(r["when"], c.values, r.get("unless")) is not None), None)
            reason = reason or c.reject
            if reason is None and len(c.text) < MIN_CHARS:
                reason = "too_short"
            if reason is None and not is_english(c.text):
                reason = "not_english"
            if reason:
                excluded[source][reason] += 1
                continue
            text = truncate(c.text)
            truncated[source] += text != c.text
            c.text = text
            silver, leaks = label(c, spec, taxonomy)
            labelled.append({
                "id": c.id,
                "source": source,
                "group": c.group,
                "text": c.text,
                "labels": silver,
                "flags": sorted({f"leak:{leak['axis']}" for leak in leaks}),
                "leaks": leaks,
                "meta": {"name": c.name, "fields": {k: sorted(v) for k, v in c.values.items()},
                         "start_time": c.start_time, **c.meta},
            })

    # One item per group: a frozen test item if there is one, otherwise the
    # longest text, ties broken by id. Deterministic, and favours the fullest
    # description.
    def preference(item: dict) -> tuple:
        return item["id"] not in keep, -len(item["text"]), item["id"]

    best: dict[str, dict] = {}
    for item in sorted(labelled, key=preference):
        if item["group"] in best:
            excluded[item["source"]]["same_group"] += 1
        else:
            best[item["group"]] = item

    items, seen_texts = [], set()
    for item in sorted(best.values(), key=lambda i: (i["id"] not in keep, i["id"])):
        key = normalise_for_dedupe(item["text"])
        if key in seen_texts:
            excluded[item["source"]]["duplicate_text"] += 1
            continue
        seen_texts.add(key)
        items.append(item)
    items.sort(key=lambda i: i["id"])

    stats = {"candidates": n_candidates, "excluded": excluded, "truncated": truncated}
    return items, stats


def shuffled(items: list[dict], key: str) -> list[dict]:
    ordered = sorted(items, key=lambda i: i["id"])
    random.Random(f"{SEED}:{key}").shuffle(ordered)
    return ordered


def split(items: list[dict], frozen_test_ids: set[str] | None) -> tuple[dict[str, list[dict]], Counter]:
    by_district = defaultdict(list)
    for item in items:
        by_district[district_of(item)].append(item)

    test, rest = [], defaultdict(list)
    if frozen_test_ids is not None:
        missing = frozen_test_ids - {i["id"] for i in items}
        if missing:
            raise SystemExit(
                f"{len(missing)} test ids in {GOLD_PATH} are no longer produced by the build "
                f"(e.g. {sorted(missing)[:3]}). The test split is frozen by that file; "
                "restore the upstream data or rules that produced them, or deliberately "
                "delete the file to re-draw the test set (losing its hand labels)."
            )
        for district, pool in sorted(by_district.items()):
            for item in pool:
                (test if item["id"] in frozen_test_ids else rest[district]).append(item)
    else:
        for district, pool in sorted(by_district.items()):
            order = shuffled(pool, f"test:{district}")
            quota = TEST_QUOTA.get(district, 0)
            if len(order) < quota:
                log.warning("%s: only %d items for a test quota of %d", district, len(order), quota)
            test += order[:quota]
            rest[district] = order[quota:]

    val, train, capped = [], [], Counter()
    for district, pool in sorted(rest.items()):
        order = shuffled(pool, f"val:{district}")
        quota = VAL_QUOTA.get(district, 0)
        val += order[:quota]
        remaining = order[quota:]
        train += remaining[:TRAIN_CAP]
        capped[district] = max(0, len(remaining) - TRAIN_CAP)
    return {"train": train, "val": val, "test": test}, capped


# --- Gold labels ------------------------------------------------------------

def parse_gold_cell(axis: str, cell: str, taxonomy: dict, item_id: str):
    cell = (cell or "").strip()
    if not cell:
        return None
    tax = taxonomy["axes"][axis]
    if tax["cardinality"] == "multi":
        labels = list(dict.fromkeys(p.strip().lower() for p in re.split(r"[;,]", cell) if p.strip()))
        ok = tax.get("min_labels", 1) <= len(labels) <= tax.get("max_labels", 1) and set(labels) <= set(tax["labels"])
        value = labels
    else:
        value = cell.lower()
        ok = value in tax["labels"]
    if not ok:
        raise SystemExit(f"{GOLD_PATH}: {item_id} has {axis}={cell!r}, which is not valid under taxonomy.yaml")
    return value


def read_gold(taxonomy: dict) -> dict[str, dict]:
    with GOLD_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    gold = {}
    for row in rows:
        source = row["source"]
        gold[row["id"]] = {axis: parse_gold_cell(axis, row.get(axis), taxonomy, row["id"])
                           for axis in HAND_AXES[source]}
    return gold


def write_gold_sheet(test: list[dict]) -> None:
    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Shuffled so the sources and districts are interleaved, which keeps the
    # labeller from settling into one mode for a long run of similar items.
    # utf-8-sig so that spreadsheet software detects the encoding.
    with GOLD_PATH.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(GOLD_COLUMNS)
        for item in shuffled(test, "gold-sheet"):
            district = "" if "district" in HAND_AXES[item["source"]] else NOT_LABELLED_HERE
            writer.writerow([item["id"], item["source"], "", "", district])
    log.info("wrote %s with %d rows to label; the test split is now frozen", GOLD_PATH, len(test))


def evaluation_labels(item: dict, gold: dict[str, dict]) -> dict:
    """Test labels: silver, except hand-labelled axes, which come only from gold."""
    labels = dict(item["labels"])
    hand = gold.get(item["id"], {})
    for axis in HAND_AXES[item["source"]]:
        labels[axis] = hand.get(axis)
    return labels


# --- Output -----------------------------------------------------------------

def public_view(item: dict) -> dict:
    """The fields that may be published. Group keys embed developer and event
    names, so they are replaced by a hash: still usable to check that no group
    straddles two splits, without revealing what the group is."""
    public = {k: item[k] for k in PUBLIC_FIELDS}
    public["group"] = "g:" + hashlib.sha256(item["group"].encode()).hexdigest()[:16]
    return public


def write_both(name: str, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda i: i["id"])
    write_jsonl(PRIVATE_PROCESSED / f"{name}.jsonl", rows)
    write_jsonl(PUBLIC_PROCESSED / f"{name}.jsonl", [public_view(r) for r in rows])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def report(items: list[dict], splits: dict, capped: Counter, stats: dict, taxonomy: dict,
           gold: dict | None, mapping: dict) -> dict:
    by_source = defaultdict(list)
    for item in items:
        by_source[item["source"]].append(item)

    coverage, distribution, leakage, lengths = {}, {}, {}, {}
    for source, rows in by_source.items():
        coverage[source], distribution[source], leakage[source] = {}, {}, {}
        for axis in taxonomy["axes"]:
            values = [r["labels"][axis] for r in rows]
            coverage[source][axis] = sum(v is not None for v in values) / len(rows)
            distribution[source][axis] = dict(Counter(
                "+".join(v) if isinstance(v, list) else str(v) for v in values).most_common())
            terms = Counter(t for r in rows for leak in r["leaks"] if leak["axis"] == axis for t in leak["terms"])
            leakage[source][axis] = {
                "rate": sum(f"leak:{axis}" in r["flags"] for r in rows) / len(rows),
                "top_terms": terms.most_common(8),
            }
        leakage[source]["any_axis"] = sum(bool(r["flags"]) for r in rows) / len(rows)
        chars = sorted(len(r["text"]) for r in rows)
        lengths[source] = {"p10": chars[len(chars) // 10], "median": statistics.median(chars),
                           "p90": chars[9 * len(chars) // 10], "max": chars[-1]}

    raw = {str(p.relative_to(REPO_ROOT)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
           for source in mapping for p in RAW_FILES[source] if p.exists()}
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"seed": SEED, "max_chars": MAX_CHARS, "min_chars": MIN_CHARS, "test_quota": TEST_QUOTA,
                   "val_quota": VAL_QUOTA, "train_cap": TRAIN_CAP, "hand_axes": HAND_AXES},
        "provenance": {"taxonomy_sha256": sha256(TAXONOMY_PATH), "mapping_sha256": sha256(MAPPING_PATH), "raw": raw},
        "sources": {s: {"candidates": stats["candidates"][s], "excluded": dict(stats["excluded"][s].most_common()),
                        "truncated": stats["truncated"][s], "kept": len(by_source[s])} for s in mapping},
        "splits": {name: {"n": len(rows), "by_district": dict(Counter(map(district_of, rows)).most_common()),
                          "by_source": dict(Counter(r["source"] for r in rows).most_common())}
                   for name, rows in splits.items()},
        "train_dropped_by_cap": dict(capped),
        "silver_coverage": coverage,
        "silver_distribution": distribution,
        "leakage": leakage,
        "text_chars": lengths,
        "gold": None if gold is None else {
            "rows": len(gold),
            "filled": {axis: sum(g.get(axis) is not None for g in gold.values())
                       for axis in ("mood", "commitment", "district")},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--freeze-test", action="store_true",
                        help=f"create {GOLD_PATH.relative_to(REPO_ROOT)} for hand-labelling, freezing the test split")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    taxonomy, mapping = load_yaml(TAXONOMY_PATH), load_yaml(MAPPING_PATH)
    validate_mapping(mapping, taxonomy)
    for source in SOURCE_FIELDS.keys() - mapping.keys():
        log.warning("mapping.yaml has no %s section: %s is left out of this build", source, source)
    if args.freeze_test and mapping.keys() != SOURCE_FIELDS.keys():
        raise SystemExit("refusing to freeze the test split while a source is missing from mapping.yaml")

    gold = read_gold(taxonomy) if GOLD_PATH.exists() else None
    items, stats = build_items(mapping, taxonomy, keep=set(gold or {}))
    splits, capped = split(items, set(gold) if gold is not None else None)

    if args.freeze_test:
        if GOLD_PATH.exists():
            raise SystemExit(f"{GOLD_PATH} already exists; it is never overwritten")
        write_gold_sheet(splits["test"])
        gold = read_gold(taxonomy)

    for name in ("train", "val"):
        write_both(name, splits[name])
    test = sorted(splits["test"], key=lambda i: i["id"])
    write_both("test", [{**item, "labels": evaluation_labels(item, gold or {})} for item in test])
    write_jsonl(SILVER_RUN_PATH, [{"id": item["id"], "output": json.dumps(item["labels"])} for item in test])
    SILVER_RUN_PATH.with_suffix(".meta.json").write_text(json.dumps(
        {"method": "silver rules from mapping.yaml", "mapping_sha256": sha256(MAPPING_PATH)}, indent=2) + "\n", encoding="utf-8")

    summary = report(items, splits, capped, stats, taxonomy, gold, mapping)
    (PUBLIC_PROCESSED / "dataset_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for source, s in summary["sources"].items():
        log.info("%s: %d candidates, %d kept; excluded %s", source, s["candidates"], s["kept"], s["excluded"])
    for name, s in summary["splits"].items():
        log.info("%s: %d items %s", name, s["n"], s["by_district"])
    if gold is None:
        log.info("test split not frozen; hand-labelled axes are null in test.jsonl until it is")


if __name__ == "__main__":
    main()
