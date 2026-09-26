"""Hand-label the test split in the terminal.

Works through data/gold/test_labels.csv one item at a time, in the file's
(shuffled) order, starting at the first item not yet labelled. It shows the
text the models see, looked up in the private test split, and nothing else:
no silver labels, no source fields, no titles. The CSV holds no text, so it
can be published. The file is saved after every item, so you can stop whenever you
like and carry on later.

At each prompt, type label numbers or names:
  mood          one or two, e.g. "1", "1 4" or "calm melancholic"
  commitment    one
  district      one (Ticketmaster items only)
  Enter         keep the current value, if there is one
  ?             show this axis's definitions and the conventions
  s             skip this item for now; it comes round again at the end
  b             go back to the previous item
  q             quit (everything finished so far is already saved)

Only the first --target items (250 by default) are labelled. The file order
is shuffled, so the first N items are a random sample of the test set, and
stopping at a fixed N rather than at a convenient moment keeps it one. Items
beyond the target stay blank, and evaluate.py leaves them out of the
hand-labelled axes.

Usage:
    python src/label_gold.py              # the first 250 items
    python src/label_gold.py --target 450 # or more, later
"""

import argparse
import csv
import json
import re
import shutil
import sys
import textwrap
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = REPO_ROOT / "taxonomy.yaml"
GOLD_PATH = REPO_ROOT / "data" / "gold" / "test_labels.csv"
PRIVATE_TEST_PATH = REPO_ROOT / "data" / "private" / "processed" / "test.jsonl"

AXES = ("mood", "commitment", "district")
NOT_LABELLED_HERE = "-"  # written by build_dataset.py in Steam rows' district cell
MAX_WIDTH = 100
# Agreed on 2026-09-25: about four hours of labelling in total was not
# available, and 250 items gives roughly a 6-point interval per run.
DEFAULT_TARGET = 250


class Back(Exception):
    pass


class Skip(Exception):
    pass


class Quit(Exception):
    pass


# --- File -------------------------------------------------------------------

def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames), list(reader)


def save_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    # Write a temporary file and swap it in, so a crash mid-write can never
    # leave a half-written file of hand labels.
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def load_texts(path: Path = PRIVATE_TEST_PATH) -> dict[str, str]:
    if not path.exists():
        raise SystemExit(f"{path} is missing: clone the private data repo into data/private/ "
                         "or run src/build_dataset.py")
    with path.open() as f:
        return {item["id"]: item["text"] for item in map(json.loads, f) if item}


def axes_for(row: dict) -> list[str]:
    return [a for a in AXES if row[a].strip() != NOT_LABELLED_HERE]


def is_complete(row: dict) -> bool:
    return all(row[a].strip() for a in axes_for(row))


def next_incomplete(rows: list[dict], start: int) -> int:
    """The first incomplete row at or after `start`, wrapping round to pick up
    skipped rows; len(rows) when everything is labelled."""
    for i in list(range(start, len(rows))) + list(range(0, min(start, len(rows)))):
        if not is_complete(rows[i]):
            return i
    return len(rows)


# --- Input ------------------------------------------------------------------

def parse(axis: str, raw: str, taxonomy: dict) -> str:
    """Turn typed numbers or names into the cell's canonical form.

    Raises ValueError with a message to show, rather than accepting anything
    the build would later reject.
    """
    spec = taxonomy["axes"][axis]
    labels = list(spec["labels"])
    limit = spec.get("max_labels", 1) if spec["cardinality"] == "multi" else 1
    chosen = []
    # Hyphens are part of labels ("an-evening"), so split on spaces and
    # punctuation only.
    for token in re.split(r"[\s,;]+", raw.strip().lower()):
        if not token:
            continue
        if token.isdigit() and 1 <= int(token) <= len(labels):
            label = labels[int(token) - 1]
        elif token in labels:
            label = token
        else:
            raise ValueError(f"'{token}' is not one of the {axis} options")
        if label not in chosen:
            chosen.append(label)
    if not 1 <= len(chosen) <= limit:
        raise ValueError(f"{axis} takes {'one' if limit == 1 else f'one to {limit}'} label{'s' if limit > 1 else ''}")
    return ";".join(chosen)


def show_help(axis: str, taxonomy: dict, width: int) -> None:
    spec = taxonomy["axes"][axis]
    print()
    print(textwrap.fill(f"{axis}: {spec['question']}", width))
    for i, (label, definition) in enumerate(spec["labels"].items(), 1):
        print(textwrap.fill(f"{i}) {label}: {definition}", width, subsequent_indent="     "))
    if spec.get("note"):
        print(textwrap.fill(f"Note: {spec['note']}", width))
    print("\nConventions:")
    for convention in taxonomy["conventions"]:
        print(textwrap.fill(f"- {convention}", width, subsequent_indent="  "))
    print()


def ask(axis: str, row: dict, taxonomy: dict, width: int, prompt=input) -> str:
    spec = taxonomy["axes"][axis]
    options = "  ".join(f"{i}) {label}" for i, label in enumerate(spec["labels"], 1))
    multi = spec["cardinality"] == "multi"
    current = row[axis].strip()
    question = f"{axis}{' (one or two)' if multi else ''}  [{options}]"
    if current:
        question += f"  Enter keeps: {current}"
    while True:
        raw = prompt(question + "\n> ").strip()
        command = raw.lower()
        if command == "?":
            show_help(axis, taxonomy, width)
            continue
        if command in ("b", "s", "q"):
            raise {"b": Back, "s": Skip, "q": Quit}[command]
        if not raw:
            if current:
                return current
            continue
        try:
            return parse(axis, raw, taxonomy)
        except ValueError as e:
            print(f"  {e}. Try again, or ? for help.")


# --- Session ----------------------------------------------------------------

def render(row: dict, text: str, index: int, rows: list[dict], pace: float | None, width: int) -> None:
    done = sum(map(is_complete, rows))
    remaining = len(rows) - done
    eta = f"  ~{remaining * pace / 60:.0f} min left at this pace" if pace else ""
    print("\033[2J\033[H", end="")  # clear the screen
    print(f"Item {index + 1} of {len(rows)}  |  {done} labelled  |  {row['source']}{eta}")
    print("-" * width)
    for paragraph in text.splitlines():
        print(textwrap.fill(paragraph, width))
    print("-" * width)


def run(path: Path, taxonomy: dict, texts: dict[str, str], prompt=input, target: int | None = None) -> None:
    fieldnames, rows = read_rows(path)
    missing = [r["id"] for r in rows if r["id"] not in texts]
    if missing:
        raise SystemExit(f"{len(missing)} labelling rows have no text in the private test split, e.g. {missing[:3]}")
    # Navigate within the first `target` rows only; save the whole file. The
    # slice holds the same row dicts, so answers land in `rows` as well.
    work = rows[:target] if target else rows
    width = min(MAX_WIDTH, shutil.get_terminal_size().columns)
    index = next_incomplete(work, 0)
    started, finished = time.monotonic(), 0

    while index < len(work):
        row = work[index]
        pace = (time.monotonic() - started) / finished if finished else None
        render(row, texts[row["id"]], index, work, pace, width)
        try:
            answers = {axis: ask(axis, row, taxonomy, width, prompt) for axis in axes_for(row)}
        except Back:
            index = max(0, index - 1)
            continue
        except Skip:
            index = next_incomplete(work, index + 1)
            if index == work.index(row):  # the only item left
                break
            continue
        except Quit:
            break
        row.update(answers)
        save_rows(path, fieldnames, rows)
        finished += 1
        index = next_incomplete(work, index + 1)

    done = sum(map(is_complete, work))
    print(f"\n{done} of {len(work)} items labelled; saved to {path}.")
    if done == len(work):
        print("Target reached. Rebuild with: .venv/bin/python src/build_dataset.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET,
                        help="label the first this-many items of the file")
    args = parser.parse_args()
    if not GOLD_PATH.exists():
        raise SystemExit(f"{GOLD_PATH} does not exist; create it with build_dataset.py --freeze-test")
    taxonomy = yaml.safe_load(TAXONOMY_PATH.read_text())
    try:
        run(GOLD_PATH, taxonomy, load_texts(), target=args.target)
    except (KeyboardInterrupt, EOFError):
        # The item in progress is dropped; every finished item is on disk.
        print("\nStopped. Everything finished so far is saved.")
        sys.exit(0)


if __name__ == "__main__":
    main()
