"""Prompt templates and the one output format every method shares.

Three templates:
  zero_shot  The whole taxonomy in the system prompt: the conventions and
             every label's definition, verbatim from taxonomy.yaml. This is
             the fair baseline: the model is given the rules in writing,
             not left to guess from bare label names.
  few_shot   zero_shot plus K fixed worked examples, given as earlier
             user/assistant turns so the model sees the exact answer format.
             The realistic alternative to training.
  compact    Axis names and label lists only: no definitions, conventions or
             examples. The fine-tune is trained and served with this. Moving
             the conventions out of the prompt and into the weights is the
             point of fine-tuning, so the fine-tune pays for them once, in
             training, instead of in the prompt tokens of every request.

Every template asks for the same answer: one JSON object with the keys in
OUTPUT_ORDER, and mood always a list. The multi-label axis (mood) comes last.
In training, an axis with no silver label still needs a value in the target
string, so a placeholder is written and its tokens are masked out of the loss.
The model still reads the placeholder as context for everything after it, so
the axis most often null goes last, where nothing comes after it but the
closing brace.

The few-shot examples are fixed, the same for every item, and chosen once
from train (never test) to cover as many labels as possible. Their ids and
silver labels are written to data/gold/fewshot_examples.json, for you to check
and correct by hand with `review`, because a prompt's examples are meant to be
right. The few-shot baseline refuses to run until every example is verified.
The file holds no text, so it can be published; texts are looked up in the
private train split when the examples are loaded.
Fixed examples also make a stable prompt prefix, which prompt caching can
reuse across requests; the cost write-up accounts for that.

Usage:
    python src/prompts.py select              # choose examples; writes the file above
    python src/prompts.py review              # check and correct them, one at a time
    python src/prompts.py show few_shot       # print a rendered prompt, with its size
"""

import argparse
import json
import math
import random
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = REPO_ROOT / "taxonomy.yaml"
# The private splits: the prompts need the text.
TRAIN_PATH = REPO_ROOT / "data" / "private" / "processed" / "train.jsonl"
TEST_PATH = REPO_ROOT / "data" / "private" / "processed" / "test.jsonl"
FEWSHOT_PATH = REPO_ROOT / "data" / "gold" / "fewshot_examples.json"

TEMPLATES = ("zero_shot", "few_shot", "compact")
FEWSHOT_K = 12
# Examples are drawn from texts no longer than this. It keeps the prompt near
# 5k tokens; examples demonstrate the conventions, so they need not be long.
FEWSHOT_MAX_CHARS = 800
SEED = 13

PLATFORM = (
    "Wayfinder is an entertainment platform organised as a city of districts: "
    "sport, gaming, music, social, commerce and live. Every item on it, from any "
    "district, is described on the same seven axes, so that items from different "
    "districts can be compared."
)


def load_taxonomy(path: Path = TAXONOMY_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def output_order(taxonomy: dict) -> list[str]:
    axes = taxonomy["axes"]
    single = [a for a, spec in axes.items() if spec["cardinality"] == "single"]
    multi = [a for a, spec in axes.items() if spec["cardinality"] == "multi"]
    return single + multi


def is_multi(taxonomy: dict, axis: str) -> bool:
    return taxonomy["axes"][axis]["cardinality"] == "multi"


# --- Answer format -----------------------------------------------------------

def format_answer(labels: dict, taxonomy: dict) -> str:
    """The canonical answer string for a complete set of labels."""
    target, _ = format_target(labels, taxonomy)
    return target


def format_target(labels: dict, taxonomy: dict) -> tuple[str, list[tuple[int, int]]]:
    """Return (target string, character spans of placeholder values).

    A null label is written as the axis's first label, so the target is
    always well formed, and its span is returned so that train_lora.py can
    mask those tokens out of the loss. The output is byte-for-byte what
    json.dumps would produce for the same object.
    """
    parts, spans, position = [], [], 1  # after the opening brace
    for i, axis in enumerate(output_order(taxonomy)):
        value = labels.get(axis)
        placeholder = value is None
        if placeholder:
            value = next(iter(taxonomy["axes"][axis]["labels"]))
        if is_multi(taxonomy, axis) and isinstance(value, str):
            value = [value]
        key = ("" if i == 0 else ", ") + json.dumps(axis) + ": "
        encoded = json.dumps(value)
        position += len(key)
        if placeholder:
            spans.append((position, position + len(encoded)))
        parts.append(key + encoded)
        position += len(encoded)
    return "{" + "".join(parts) + "}", spans


def answer_skeleton(taxonomy: dict) -> str:
    fields = []
    for axis in output_order(taxonomy):
        fields.append(f'"{axis}": ' + (f'["<{axis} label>", ...]' if is_multi(taxonomy, axis) else f'"<{axis} label>"'))
    return "{" + ", ".join(fields) + "}"


def format_instructions(taxonomy: dict) -> str:
    multi = [a for a in output_order(taxonomy) if is_multi(taxonomy, a)]
    limits = {a: taxonomy["axes"][a].get("max_labels", 1) for a in multi}
    lists = "; ".join(f"{a} is a list of 1 to {n} labels" for a, n in limits.items())
    return (
        "Answer with one JSON object and nothing else, with exactly these keys in this order:\n"
        f"{answer_skeleton(taxonomy)}\n"
        f"Use only the labels listed. {lists[0].upper() + lists[1:]}; every other value is a single label."
    )


# --- Templates --------------------------------------------------------------

def full_system_prompt(taxonomy: dict) -> str:
    lines = [PLATFORM, "", "Read the description and label it using the definitions and conventions below.", "",
             "Conventions:"]
    lines += [f"- {c}" for c in taxonomy["conventions"]]
    lines += ["", "Axes:"]
    for axis in output_order(taxonomy):
        spec = taxonomy["axes"][axis]
        lines += ["", f"{axis}: {spec['question']}"]
        lines += [f"  {label}: {definition}" for label, definition in spec["labels"].items()]
        if spec.get("note"):
            lines.append(f"  Note: {spec['note']}")
    lines += ["", format_instructions(taxonomy)]
    return "\n".join(lines)


def compact_system_prompt(taxonomy: dict) -> str:
    lines = ["Label the description on the Wayfinder taxonomy."]
    for axis in output_order(taxonomy):
        spec = taxonomy["axes"][axis]
        labels = " | ".join(spec["labels"])
        if is_multi(taxonomy, axis):
            labels = f"1 to {spec.get('max_labels', 1)} of: {labels}"
        lines.append(f"{axis}: {labels}")
    lines += ["", format_instructions(taxonomy)]
    return "\n".join(lines)


def user_message(text: str) -> str:
    return f"Description:\n{text}"


def build_messages(template: str, text: str, taxonomy: dict, examples: list[dict] | None = None) -> list[dict]:
    """Chat messages for one item. The runner applies the model's chat template."""
    if template not in TEMPLATES:
        raise ValueError(f"unknown template {template!r}")
    system = compact_system_prompt(taxonomy) if template == "compact" else full_system_prompt(taxonomy)
    messages = [{"role": "system", "content": system}]
    if template == "few_shot":
        if not examples:
            raise ValueError("few_shot needs examples")
        for ex in examples:
            messages += [{"role": "user", "content": user_message(ex["text"])},
                         {"role": "assistant", "content": format_answer(ex["labels"], taxonomy)}]
    messages.append({"role": "user", "content": user_message(text)})
    return messages


# --- Few-shot examples -------------------------------------------------------

def label_pairs(labels: dict) -> set[tuple[str, str]]:
    return {(axis, label) for axis, value in labels.items() if value is not None
            for label in (value if isinstance(value, list) else [value])}


def select_fewshot(train: list[dict], k: int = FEWSHOT_K, max_chars: int = FEWSHOT_MAX_CHARS) -> list[dict]:
    """Greedily pick k fully labelled train items that cover the most labels.

    Each pick maximises the number of (axis, label) pairs not yet covered, so
    rare labels such as late-night, crowd or minutes appear at least once
    where the data has them. Sources are capped at an equal share. Ties go to
    the first candidate in a seeded shuffle, so the choice is reproducible.
    """
    pool = [i for i in train if all(v is not None for v in i["labels"].values()) and len(i["text"]) <= max_chars]
    pool.sort(key=lambda i: i["id"])
    random.Random(f"{SEED}:fewshot").shuffle(pool)
    sources = sorted({i["source"] for i in pool})
    cap = math.ceil(k / len(sources))

    chosen, covered, per_source = [], set(), dict.fromkeys(sources, 0)
    while len(chosen) < k:
        candidates = [i for i in pool if i not in chosen and per_source[i["source"]] < cap]
        if not candidates:
            break
        best = max(candidates, key=lambda i: len(label_pairs(i["labels"]) - covered))
        chosen.append(best)
        covered |= label_pairs(best["labels"])
        per_source[best["source"]] += 1
    # Present in a mixed order, not greedy order, so the examples are not
    # sorted from most to least unusual.
    random.Random(f"{SEED}:fewshot-order").shuffle(chosen)
    return chosen


def validate_example_labels(labels: dict, taxonomy: dict, where: str) -> None:
    for axis, spec in taxonomy["axes"].items():
        value = labels.get(axis)
        values = value if isinstance(value, list) else [value]
        ok = value is not None and all(v in spec["labels"] for v in values)
        if is_multi(taxonomy, axis):
            ok = ok and isinstance(value, list) and spec.get("min_labels", 1) <= len(set(values)) <= spec.get("max_labels", 1)
        elif isinstance(value, list):
            ok = False
        if not ok:
            raise ValueError(f"{where}: {axis}={value!r} is not a valid, complete label")


def load_fewshot(taxonomy: dict, path: Path = FEWSHOT_PATH, require_verified: bool = True,
                 texts: dict[str, str] | None = None) -> list[dict]:
    """The examples with their texts attached, from the private train split
    unless `texts` is given."""
    examples = json.loads(path.read_text())["examples"]
    for ex in examples:
        validate_example_labels(ex["labels"], taxonomy, f"{path.name} {ex['id']}")
    unverified = [ex["id"] for ex in examples if not ex.get("verified")]
    if require_verified and unverified:
        raise SystemExit(f"{len(unverified)} few-shot examples in {path} are not verified yet, e.g. {unverified[:3]}; "
                         "run: python src/prompts.py review")
    if texts is None:
        texts = {i["id"]: i["text"] for i in read_jsonl(TRAIN_PATH)}
    missing = [ex["id"] for ex in examples if ex["id"] not in texts]
    if missing:
        raise SystemExit(f"few-shot examples not in the private train split: {missing}")
    return [{**ex, "text": texts[ex["id"]]} for ex in examples]


def save_fewshot(examples: list[dict], path: Path = FEWSHOT_PATH) -> None:
    """Write examples without their text, which stays in the private split."""
    note = ("Few-shot examples: ids and labels only. Check and correct them with `python src/prompts.py review`, "
            "which shows each text; labels start as the silver rules' guesses.")
    body = {"note": note, "examples": [{k: ex[k] for k in ("id", "source", "labels", "verified")} for ex in examples]}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def parse_edit(token: str, taxonomy: dict) -> tuple[str, object]:
    """Parse "axis=value" (mood: "a;b"), raising ValueError with a message."""
    if "=" not in token:
        raise ValueError(f"'{token}' is not axis=value")
    axis, value = (part.strip().lower() for part in token.split("=", 1))
    if axis not in taxonomy["axes"]:
        raise ValueError(f"unknown axis '{axis}'")
    spec = taxonomy["axes"][axis]
    if is_multi(taxonomy, axis):
        labels = list(dict.fromkeys(v.strip() for v in value.replace(",", ";").split(";") if v.strip()))
        if not spec.get("min_labels", 1) <= len(labels) <= spec.get("max_labels", 1) or not set(labels) <= set(spec["labels"]):
            raise ValueError(f"'{value}' is not 1 to {spec.get('max_labels', 1)} {axis} labels")
        return axis, labels
    if value not in spec["labels"]:
        raise ValueError(f"'{value}' is not a {axis} label")
    return axis, value


def review(taxonomy: dict, prompt=input, path: Path = FEWSHOT_PATH, texts: dict[str, str] | None = None) -> None:
    """Show each unverified example's text and labels; confirm or correct them.

    Enter accepts the labels as shown. "axis=value" edits one (several may be
    given on one line, e.g. "commitment=minutes mood=calm;melancholic"), after
    which the example is shown again. "q" quits; each example is saved as soon
    as it is verified.
    """
    examples = load_fewshot(taxonomy, path, require_verified=False, texts=texts)
    order = output_order(taxonomy)
    for n, ex in enumerate(examples, 1):
        if ex.get("verified"):
            continue
        while True:
            print("\033[2J\033[H", end="")
            print(f"Example {n} of {len(examples)}  |  {ex['source']}  |  {ex['id']}\n{'-' * 80}\n{ex['text']}\n{'-' * 80}")
            for axis in order:
                value = ex["labels"][axis]
                print(f"  {axis:12s} {';'.join(value) if isinstance(value, list) else value}")
            reply = prompt("Enter = correct as shown; axis=value to change; q = quit\n> ").strip()
            if reply.lower() == "q":
                return
            if not reply:
                ex["verified"] = True
                save_fewshot(examples, path)
                break
            try:
                edits = dict(parse_edit(token, taxonomy) for token in reply.split())
            except ValueError as e:
                prompt(f"  {e}. Enter to continue.")
                continue
            ex["labels"] = {**ex["labels"], **edits}
    print(f"{sum(e.get('verified', False) for e in examples)} of {len(examples)} examples verified.")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


# --- CLI --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("select", help=f"choose few-shot examples from train; writes {FEWSHOT_PATH.relative_to(REPO_ROOT)}")
    commands.add_parser("review", help="check and correct the few-shot examples, one at a time")
    p_show = commands.add_parser("show", help="print a rendered prompt for one test item")
    p_show.add_argument("template", choices=TEMPLATES)
    args = parser.parse_args()
    taxonomy = load_taxonomy()

    if args.command == "select":
        if FEWSHOT_PATH.exists():
            raise SystemExit(f"{FEWSHOT_PATH} exists; it holds hand-checked labels and is never overwritten")
        chosen = select_fewshot(read_jsonl(TRAIN_PATH))
        FEWSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_fewshot([{"id": i["id"], "source": i["source"], "verified": False,
                       "labels": {a: i["labels"][a] for a in output_order(taxonomy)}} for i in chosen])
        covered = set().union(*(label_pairs(i["labels"]) for i in chosen))
        possible = {(a, lab) for a, spec in taxonomy["axes"].items() for lab in spec["labels"]}
        print(f"wrote {len(chosen)} examples to {FEWSHOT_PATH}")
        print(f"they cover {len(covered)} of {len(possible)} labels; missing: {sorted(possible - covered)}")
        return

    if args.command == "review":
        review(taxonomy)
        return

    examples = load_fewshot(taxonomy, require_verified=False) if args.template == "few_shot" else None
    item = read_jsonl(TEST_PATH)[0]
    messages = build_messages(args.template, item["text"], taxonomy, examples)
    for m in messages:
        print(f"--- {m['role']} ---\n{m['content']}\n")
    chars = sum(len(m["content"]) for m in messages)
    # Roughly 4 characters per token for English; run_baseline.py counts exactly.
    print(f"[{args.template}: {len(messages)} messages, {chars} characters, roughly {chars // 4} tokens]")


if __name__ == "__main__":
    main()
