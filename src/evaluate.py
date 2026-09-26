"""Score a run's predictions against the held-out test split.

Every method (majority, zero-shot, few-shot, fine-tune) is scored by this one
script, unmodified. Its sha256 is written into every results file, alongside
those of the test split and the taxonomy, so anyone can check that all runs
were scored by the same code on the same items.

Test split: data/processed/test.jsonl, one item per line. This is the public
version of the split, without text: scoring needs only ids, labels and flags,
so anyone can re-score the published predictions.
    {"id": "steam:620", "source": "steam",
     "labels": {"energy": "medium", "mood": ["playful", "intense"],
                "commitment": null, ...},
     "flags": ["genre_leak"]}
  A null label means that axis has no ground truth for the item. The item is
  left out of that axis only, and each axis reports how many items it was
  scored on. Flags mark contamination, and a "clean" slice without it is
  scored too. A flag "leak:<axis>" (the text states the source term that
  produced that axis's label, such as a game tagged Roguelike calling itself
  a roguelike) removes the item from that axis only in the clean slice. Any
  other flag removes the item from every axis there.

Predictions: results/predictions/<run>.jsonl, one line per test item
    {"id": "steam:620", "output": "<raw generated text>",
     "latency_s": 0.84, "input_tokens": 912, "output_tokens": 61,
     "cost_usd": 0.00031}
  `output` is the raw text, not a parsed object. Parsing happens here, the
  same way for every method, so that no method gains from a more forgiving
  parser in its own runner. The majority baseline writes its constant answer
  as JSON text in the same field. The cost and latency fields are optional
  and are summarised when present. An optional <run>.meta.json beside the
  predictions (model, prompt version, ...) is copied into the results.

Scoring rules
  - A test item with no prediction is wrong on every axis. Dropping it would
    reward a method for failing silently.
  - The output is read as the first JSON object in the text, so a Markdown
    fence or a preamble does not fail an otherwise valid answer. If no object
    parses, the item is wrong on every axis and counts as a parse failure.
  - Keys and values are trimmed and lower-cased; nothing else is forgiven. A
    value outside the taxonomy is wrong on that axis and counts towards its
    invalid rate. A mood with no labels or more than max_labels is invalid,
    not truncated.
  - Accuracy is exact match, which for mood means the whole set must match.
    Mood also reports mean Jaccard overlap, which gives partial credit.
  - Macro-F1 averages per-label F1 over the labels present in the gold data.
    Averaging over every label either side used (scikit-learn's default)
    would let the denominator change from run to run, so two runs' figures
    would not be comparable. It is the headline metric: on a skewed axis the
    majority baseline has high accuracy but low macro-F1.
  - Intervals are 95% percentile bootstraps over items, with fixed seeds.
    `compare` draws the same resampled items for both runs (a paired
    bootstrap). That is far tighter than checking whether two separate
    intervals overlap, because it removes the shared difficulty of the items.

Usage:
    python src/evaluate.py score results/predictions/fewshot.jsonl
    python src/evaluate.py compare results/predictions/fewshot.jsonl results/predictions/lora.jsonl
"""

import argparse
import hashlib
import json
import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = REPO_ROOT / "taxonomy.yaml"
DEFAULT_TEST_PATH = REPO_ROOT / "data" / "processed" / "test.jsonl"
RESULTS_DIR = REPO_ROOT / "results"

N_BOOTSTRAP = 1000
SEED = 0
USAGE_FIELDS = ("latency_s", "input_tokens", "output_tokens", "cost_usd")

# A prediction for one axis, and a gold label, are both sets of labels: a
# single-label axis is a set of one. An invalid or missing prediction is the
# empty set, which matches nothing. One representation means one code path.
Labels = frozenset[str]
NO_ANSWER: Labels = frozenset()


@dataclass(frozen=True)
class Axis:
    name: str
    labels: tuple[str, ...]
    multi: bool
    min_labels: int
    max_labels: int


def load_taxonomy(path: Path = TAXONOMY_PATH) -> tuple[int, list[Axis]]:
    spec = yaml.safe_load(path.read_text())
    axes = []
    for name, a in spec["axes"].items():
        multi = a["cardinality"] == "multi"
        axes.append(Axis(
            name=name,
            labels=tuple(a["labels"]),
            multi=multi,
            min_labels=a.get("min_labels", 1) if multi else 1,
            max_labels=a.get("max_labels", 1) if multi else 1,
        ))
    return spec["version"], axes


# --- Parsing -----------------------------------------------------------------

_decoder = json.JSONDecoder()


def extract_json_object(text: str) -> dict | None:
    """Return the first JSON object in `text`, or None."""
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = _decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def normalise_key(key: str) -> str:
    return key.strip().lower().replace("-", "_").replace(" ", "_")


def parse_value(axis: Axis, value) -> Labels | None:
    """Return the value as a label set, or None if it is invalid."""
    if axis.multi and isinstance(value, str):
        value = [value]
    if axis.multi:
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            return None
        labels = frozenset(v.strip().lower() for v in value)
        if not axis.min_labels <= len(labels) <= axis.max_labels:
            return None
    else:
        if not isinstance(value, str):
            return None
        labels = frozenset([value.strip().lower()])
    return labels if labels <= set(axis.labels) else None


def parse_output(output: str, axes: list[Axis]) -> tuple[bool, dict[str, Labels]]:
    """Return (parsed, {axis: labels}), with NO_ANSWER where invalid."""
    obj = extract_json_object(output)
    if obj is None:
        return False, {a.name: NO_ANSWER for a in axes}
    obj = {normalise_key(k): v for k, v in obj.items() if isinstance(k, str)}
    return True, {a.name: parse_value(a, obj.get(a.name)) or NO_ANSWER for a in axes}


def parse_gold(item: dict, axes: list[Axis]) -> dict[str, Labels | None]:
    gold = {}
    for axis in axes:
        value = item["labels"].get(axis.name)
        if value is None:
            gold[axis.name] = None
            continue
        labels = parse_value(axis, value)
        if labels is None:
            raise ValueError(f"{item['id']}: gold {axis.name}={value!r} is not valid under the taxonomy")
        gold[axis.name] = labels
    return gold


# --- Loading -----------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    # Strict, unlike the fetchers: processed data and predictions must be whole.
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_predictions(path: Path, test_ids: set[str]) -> dict[str, dict]:
    predictions = {}
    for n, record in enumerate(read_jsonl(path), 1):
        item_id = record["id"]
        if item_id in predictions:
            raise ValueError(f"{path}, record {n}: duplicate prediction for {item_id}")
        if item_id not in test_ids:
            raise ValueError(f"{path}, record {n}: {item_id} is not in the test split")
        predictions[item_id] = record
    return predictions


@dataclass
class Run:
    """A run's predictions aligned to the test items, in test-file order."""
    name: str
    predicted: list[dict[str, Labels]]
    n_missing: int
    n_unparsed: int
    records: list[dict | None]


def align(name: str, items: list[dict], predictions: dict[str, dict], axes: list[Axis]) -> Run:
    predicted, records, n_missing, n_unparsed = [], [], 0, 0
    for item in items:
        record = predictions.get(item["id"])
        records.append(record)
        if record is None:
            n_missing += 1
            predicted.append({a.name: NO_ANSWER for a in axes})
            continue
        parsed, labels = parse_output(record.get("output") or "", axes)
        n_unparsed += not parsed
        predicted.append(labels)
    return Run(name, predicted, n_missing, n_unparsed, records)


# --- Metrics -----------------------------------------------------------------

Pair = tuple[Labels, Labels]  # (gold, predicted)


def label_counts(axis: Axis, pairs: list[Pair]) -> dict[str, dict[str, int]]:
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in axis.labels}
    for gold, pred in pairs:
        for label in gold & pred:
            counts[label]["tp"] += 1
        for label in pred - gold:
            counts[label]["fp"] += 1
        for label in gold - pred:
            counts[label]["fn"] += 1
    return counts


def f1(c: dict[str, int]) -> float:
    # 2tp / (2tp + fp + fn) equals the harmonic mean of precision and recall,
    # and stays defined when a label is never predicted.
    denominator = 2 * c["tp"] + c["fp"] + c["fn"]
    return 2 * c["tp"] / denominator if denominator else 0.0


def headline(axis: Axis, pairs: list[Pair]) -> dict[str, float]:
    counts = label_counts(axis, pairs)
    present = [c for c in counts.values() if c["tp"] + c["fn"] > 0]
    result = {
        "accuracy": sum(g == p for g, p in pairs) / len(pairs),
        "macro_f1": sum(map(f1, present)) / len(present),
    }
    if axis.multi:
        result["jaccard"] = sum(len(g & p) / len(g | p) for g, p in pairs) / len(pairs)
    return result


def detail(axis: Axis, pairs: list[Pair]) -> dict:
    counts = label_counts(axis, pairs)
    labels = {}
    for label, c in counts.items():
        predicted, support = c["tp"] + c["fp"], c["tp"] + c["fn"]
        labels[label] = {
            "support": support,
            "predicted": predicted,
            # None rather than 0 where undefined, so "never predicted" and
            # "always wrong" read differently.
            "precision": c["tp"] / predicted if predicted else None,
            "recall": c["tp"] / support if support else None,
            "f1": f1(c) if support or predicted else None,
        }
    result = {"invalid_rate": sum(p == NO_ANSWER for _, p in pairs) / len(pairs), "labels": labels}
    if not axis.multi:
        confusion: dict[str, dict[str, int]] = {}
        for gold, pred in pairs:
            row = confusion.setdefault(next(iter(gold)), {})
            key = next(iter(pred)) if pred else "<invalid>"
            row[key] = row.get(key, 0) + 1
        result["confusion"] = confusion
    return result


def percentile(sorted_values: list[float], q: float) -> float:
    position = q * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def interval(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [percentile(ordered, 0.025), percentile(ordered, 0.975)]


def bootstrap(axis: Axis, runs: list[list[Pair]], key: str, n_resamples: int) -> list[list[dict]]:
    """Resample items once per replicate and apply the same draw to every run.

    Each replicate recomputes the statistic from scratch, including which
    labels are present for macro-F1. Seeding from `key` makes every interval
    reproducible, and independent of which other slices or axes were scored.
    """
    rng = random.Random(f"{SEED}:{key}")
    n = len(runs[0])
    replicates: list[list[dict]] = [[] for _ in runs]
    for _ in range(n_resamples):
        draw = [rng.randrange(n) for _ in range(n)]
        for r, pairs in enumerate(runs):
            replicates[r].append(headline(axis, [pairs[i] for i in draw]))
    return replicates


# --- Slices ------------------------------------------------------------------

LEAK_PREFIX = "leak:"


def item_level_flags(item: dict) -> list[str]:
    return [f for f in item.get("flags", []) if not f.startswith(LEAK_PREFIX)]


def slices(items: list[dict]) -> dict[str, list[int]]:
    result = {"all": list(range(len(items)))}
    if any(item.get("flags") for item in items):
        result["clean"] = [i for i, item in enumerate(items) if not item_level_flags(item)]
    sources = sorted({item["source"] for item in items})
    if len(sources) > 1:
        for source in sources:
            result[f"source={source}"] = [i for i, item in enumerate(items) if item["source"] == source]
    return result


def axis_indices(items: list[dict], indices: list[int], slice_name: str, axis: Axis) -> list[int]:
    """The slice's items that count for one axis."""
    if slice_name != "clean":
        return indices
    leak = f"{LEAK_PREFIX}{axis.name}"
    return [i for i in indices if leak not in items[i].get("flags", [])]


def axis_pairs(axis: Axis, gold: list[dict], run: Run, indices: list[int]) -> list[Pair]:
    return [(gold[i][axis.name], run.predicted[i][axis.name]) for i in indices if gold[i][axis.name] is not None]


# --- Commands ----------------------------------------------------------------

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(test_path: Path, taxonomy_version: int, n_items: int) -> dict:
    return {
        "evaluate_sha256": sha256(Path(__file__)),
        "test_sha256": sha256(test_path),
        "taxonomy_sha256": sha256(TAXONOMY_PATH),
        "taxonomy_version": taxonomy_version,
        "n_test_items": n_items,
    }


def usage_summary(records: list[dict | None]) -> dict:
    summary = {}
    for field in USAGE_FIELDS:
        values = sorted(r[field] for r in records if r is not None and r.get(field) is not None)
        if values:
            summary[field] = {
                "n": len(values),
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "p95": percentile(values, 0.95),
                "total": sum(values),
            }
    return summary


def score(items: list[dict], axes: list[Axis], run: Run, n_resamples: int) -> dict:
    gold = [parse_gold(item, axes) for item in items]
    result = {}
    for slice_name, indices in slices(items).items():
        slice_axes = {}
        for axis in axes:
            pairs = axis_pairs(axis, gold, run, axis_indices(items, indices, slice_name, axis))
            if not pairs:
                slice_axes[axis.name] = {"n": 0}
                continue
            point = headline(axis, pairs)
            (replicates,) = bootstrap(axis, [pairs], f"{slice_name}:{axis.name}", n_resamples)
            entry = {"n": len(pairs)}
            for metric, value in point.items():
                entry[metric] = value
                entry[f"{metric}_ci"] = interval([r[metric] for r in replicates])
            entry.update(detail(axis, pairs))
            slice_axes[axis.name] = entry
        result[slice_name] = {"n_items": len(indices), "axes": slice_axes}
    return result


def compare(items: list[dict], axes: list[Axis], a: Run, b: Run, n_resamples: int) -> dict:
    gold = [parse_gold(item, axes) for item in items]
    result = {}
    for slice_name, indices in slices(items).items():
        slice_axes = {}
        for axis in axes:
            axis_items = axis_indices(items, indices, slice_name, axis)
            pairs_a = axis_pairs(axis, gold, a, axis_items)
            pairs_b = axis_pairs(axis, gold, b, axis_items)
            if not pairs_a:
                slice_axes[axis.name] = {"n": 0}
                continue
            point_a, point_b = headline(axis, pairs_a), headline(axis, pairs_b)
            reps_a, reps_b = bootstrap(axis, [pairs_a, pairs_b], f"{slice_name}:{axis.name}", n_resamples)
            entry = {"n": len(pairs_a), a.name: point_a, b.name: point_b, "difference": {}}
            for metric in point_a:
                diffs = [rb[metric] - ra[metric] for ra, rb in zip(reps_a, reps_b)]
                lo, hi = interval(diffs)
                entry["difference"][metric] = {
                    "value": point_b[metric] - point_a[metric],
                    "ci": [lo, hi],
                    # The interval excludes zero: the difference is not
                    # plausibly resampling noise at the 95% level.
                    "clear": lo > 0 or hi < 0,
                }
            slice_axes[axis.name] = entry
        result[slice_name] = {"n_items": len(indices), "axes": slice_axes}
    return result


def load_run(path: Path, items: list[dict], axes: list[Axis]) -> Run:
    predictions = load_predictions(path, {item["id"] for item in items})
    return align(path.stem, items, predictions, axes)


def load_items(test_path: Path) -> list[dict]:
    items = read_jsonl(test_path)
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{test_path} contains duplicate ids")
    return items


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def fmt_ci(entry: dict, metric: str) -> str:
    lo, hi = entry[f"{metric}_ci"]
    return f"{entry[metric]:.3f} [{lo:.3f}, {hi:.3f}]"


def print_score(name: str, result: dict) -> None:
    for slice_name in ("all", "clean"):
        if slice_name not in result["slices"]:
            continue
        block = result["slices"][slice_name]
        print(f"\n{name}  slice={slice_name}  items={block['n_items']}")
        print(f"  {'axis':12s} {'n':>5s}  {'macro-F1 [95% CI]':24s}  {'accuracy [95% CI]':24s}  invalid")
        for axis, e in block["axes"].items():
            if not e["n"]:
                print(f"  {axis:12s} {0:5d}  (no gold labels)")
                continue
            print(f"  {axis:12s} {e['n']:5d}  {fmt_ci(e, 'macro_f1'):24s}  {fmt_ci(e, 'accuracy'):24s}  {e['invalid_rate']:.1%}")
    c = result["coverage"]
    print(f"\n  missing {c['n_missing']}, unparsed {c['n_unparsed']} of {c['n_test_items']} test items")


def print_compare(a: str, b: str, result: dict) -> None:
    for slice_name in ("all", "clean"):
        if slice_name not in result:
            continue
        print(f"\n{b} minus {a}  slice={slice_name}")
        print(f"  {'axis':12s} {'n':>5s}  {'macro-F1 difference [95% CI]':30s}  clear?")
        for axis, e in result[slice_name]["axes"].items():
            if not e["n"]:
                continue
            d = e["difference"]["macro_f1"]
            span = f"{d['value']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]"
            print(f"  {axis:12s} {e['n']:5d}  {span:30s}  {'yes' if d['clear'] else 'no'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    commands = parser.add_subparsers(dest="command", required=True)
    p_score = commands.add_parser("score", help="score one run; writes results/<run>.json")
    p_score.add_argument("predictions", type=Path)
    p_compare = commands.add_parser("compare", help="paired comparison; writes results/compare_<a>_vs_<b>.json")
    p_compare.add_argument("a", type=Path, help="reference run, e.g. few-shot")
    p_compare.add_argument("b", type=Path, help="run compared against it, e.g. the fine-tune")
    args = parser.parse_args()

    taxonomy_version, axes = load_taxonomy()
    items = load_items(args.test)
    base = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_bootstrap": args.n_bootstrap,
        "provenance": provenance(args.test, taxonomy_version, len(items)),
    }

    if args.command == "score":
        run = load_run(args.predictions, items, axes)
        meta_path = args.predictions.with_suffix(".meta.json")
        result = {
            "run": run.name,
            **base,
            "predictions_sha256": sha256(args.predictions),
            "meta": json.loads(meta_path.read_text()) if meta_path.exists() else None,
            "coverage": {
                "n_test_items": len(items),
                "n_missing": run.n_missing,
                "n_unparsed": run.n_unparsed,
                "parse_failure_rate": (run.n_missing + run.n_unparsed) / len(items),
            },
            "usage": usage_summary(run.records),
            "slices": score(items, axes, run, args.n_bootstrap),
        }
        write_json(RESULTS_DIR / f"{run.name}.json", result)
        print_score(run.name, result)
    else:
        a, b = load_run(args.a, items, axes), load_run(args.b, items, axes)
        result = {
            "a": a.name,
            "b": b.name,
            **base,
            "predictions_sha256": {a.name: sha256(args.a), b.name: sha256(args.b)},
            "slices": compare(items, axes, a, b, args.n_bootstrap),
        }
        write_json(RESULTS_DIR / f"compare_{a.name}_vs_{b.name}.json", result)
        print_compare(a.name, b.name, result["slices"])


if __name__ == "__main__":
    main()
