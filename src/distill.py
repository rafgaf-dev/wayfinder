"""Distil few-shot labels into training data for a second fine-tune.

The silver labels for mood and commitment come from Steam tags and
Ticketmaster genres, and they disagree with the hand labels far more often
than they agree. This builds a second training set in which those two axes
are labelled instead by the few-shot model (the "teacher"): the same prompt,
examples, model and greedy decoding as the few-shot baseline. A fine-tune on
it (the "student") answers a different question from the silver fine-tune:
can training reproduce few-shot quality at the compact prompt's cost? It
cannot be expected to beat its teacher. Comparing the two fine-tunes
measures what label quality is worth, since they differ only in labels.

  label   The teacher labels every train and validation item. Raw outputs go
          to results/predictions/teacher-{train,val}.jsonl (labels only, no
          text), appended as they arrive, so a dead session resumes.
  build   Writes {train,val}_distilled.jsonl, private (with text) and public
          (labels only). Each item keeps its silver labels except on
          TEACHER_AXES, where it takes the teacher's answer. The other axes
          stay silver deliberately: the test set scores them against silver
          labels, so teaching the student anything else there would be
          marked down as error. An invalid teacher answer becomes null,
          which training masks. The 12 few-shot examples are in train; they
          keep their hand-checked labels rather than the teacher's copy.

Unlike the baselines, the teacher decodes in batches (left-padded). Its
per-item latency is not a result, only its total cost, which is recorded:
distillation's GPU time is part of what the fine-tune "pays once". Batched
greedy decoding in fp16 can differ from unbatched in rare near-ties, which is
harmless for training labels.

Then train and predict with train_lora.py:
    python src/train_lora.py train --labels data/private/processed/train_distilled.jsonl \\
        --val-labels data/private/processed/val_distilled.jsonl --output-dir <dir>/lora-distilled
    python src/train_lora.py predict --output-dir <dir>/lora-distilled --run-name lora-distilled

Usage:
    python src/distill.py label train
    python src/distill.py label val
    python src/distill.py build
"""

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import build_dataset as bd
import evaluate as ev
import prompts
import run_baseline as rb

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIVATE = REPO_ROOT / "data" / "private" / "processed"
PUBLIC = REPO_ROOT / "data" / "processed"
REPORT_PATH = REPO_ROOT / "results" / "distill_report.json"

TEACHER_AXES = ("mood", "commitment")
SPLITS = ("train", "val")
BATCH_SIZE = 8


def teacher_path(split: str) -> Path:
    return rb.PREDICTIONS_DIR / f"teacher-{split}.jsonl"


# --- Label ------------------------------------------------------------------

def generate_batch(model, tokenizer, device: str, batch: list[list[dict]], config) -> tuple[list[str], int, int]:
    """Greedy answers for a batch of chat prompts; also total input and output tokens."""
    import torch
    texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, generation_config=config)
    width = inputs["input_ids"].shape[1]  # left padding: every answer starts here
    answers, n_out = [], 0
    for row in output[:, width:]:
        generated = row[row != tokenizer.pad_token_id] if tokenizer.pad_token_id is not None else row
        n_out += int(generated.shape[0])
        answers.append(tokenizer.decode(generated, skip_special_tokens=True))
    return answers, int(inputs["attention_mask"].sum()), n_out


def label(split: str, batch_size: int = BATCH_SIZE, limit: int | None = None,
          model_id: str = rb.MODEL_ID, require_verified: bool = True) -> None:
    taxonomy = prompts.load_taxonomy()
    examples = prompts.load_fewshot(taxonomy, require_verified=require_verified)
    example_ids = {e["id"] for e in examples}
    items = [i for i in rb.read_jsonl(PRIVATE / f"{split}.jsonl") if i["id"] not in example_ids]
    items = items[:limit] if limit else items

    out_path = teacher_path(split)
    meta_path = out_path.with_suffix(".meta.json")
    done = {r["id"] for r in rb.read_jsonl(out_path)} if out_path.exists() else set()
    previous = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    # Longest first, so similar lengths share a batch (less padding) and any
    # memory problem shows up in the first batch rather than an hour in.
    todo = sorted((i for i in items if i["id"] not in done), key=lambda i: (-len(i["text"]), i["id"]))
    print(f"teacher, {split}: {len(done)} done, {len(todo)} to go")
    if not todo:
        return

    model, tokenizer, device = rb.load_model(model_id)
    tokenizer.padding_side = "left"
    config = rb.generation_config(model, tokenizer)
    system = prompts.build_messages("few_shot", "", taxonomy, examples)[0]["content"]
    totals = previous.get("totals", {"wall_s": 0.0, "input_tokens": 0, "output_tokens": 0, "items": 0})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as out:
        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            messages = [prompts.build_messages("few_shot", i["text"], taxonomy, examples) for i in batch]
            rb.synchronise(device)
            began = time.perf_counter()
            answers, n_in, n_out = generate_batch(model, tokenizer, device, messages, config)
            rb.synchronise(device)
            totals["wall_s"] += time.perf_counter() - began
            totals["input_tokens"] += n_in
            totals["output_tokens"] += n_out
            totals["items"] += len(batch)
            for item, answer in zip(batch, answers):
                out.write(json.dumps({"id": item["id"], "output": answer}, ensure_ascii=False) + "\n")
            out.flush()
            meta_path.write_text(json.dumps({
                "role": "teacher", "method": "few_shot", "split": split, "model": model_id,
                "model_revision": getattr(model.config, "_commit_hash", None),
                "system_prompt_sha256": rb.sha256_text(system), "fewshot_ids": sorted(example_ids),
                "decoding": {**rb.decoding_settings(), "batch_size": batch_size}, "device": device,
                "totals": totals, "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, indent=2) + "\n")
            n = start + len(batch)
            if (start // batch_size) % 25 == 0 or n == len(todo):
                rate = totals["items"] / totals["wall_s"]
                print(f"  {n}/{len(todo)}  {rate:.2f} items/s, ~{(len(todo) - n) / rate / 60:.0f} min left")


# --- Build ------------------------------------------------------------------

def to_label(axis: ev.Axis, labels: frozenset):
    """The evaluator's parsed set back to a label (a list for mood), or None."""
    if not labels:
        return None
    ordered = [lab for lab in axis.labels if lab in labels]
    return ordered if axis.multi else ordered[0]


def distil_labels(item: dict, teacher_output: str | None, example_labels: dict | None,
                  axes: list[ev.Axis]) -> dict:
    labels = dict(item["labels"])
    if example_labels is not None:
        for axis in TEACHER_AXES:
            labels[axis] = example_labels[axis]
        return labels
    parsed = ev.parse_output(teacher_output or "", axes)[1]
    by_name = {a.name: a for a in axes}
    for axis in TEACHER_AXES:
        labels[axis] = to_label(by_name[axis], parsed[axis])
    return labels


def example_labels(taxonomy: dict, require_verified: bool = True) -> dict[str, dict]:
    """The few-shot examples' hand-checked labels, which become training labels."""
    examples = json.loads(prompts.FEWSHOT_PATH.read_text())["examples"]
    if require_verified and not all(e.get("verified") for e in examples):
        raise SystemExit("few-shot examples are not all verified; run: python src/prompts.py review")
    for e in examples:
        prompts.validate_example_labels(e["labels"], taxonomy, f"few-shot {e['id']}")
    return {e["id"]: e["labels"] for e in examples}


def build(require_verified: bool = True) -> None:
    taxonomy = prompts.load_taxonomy()
    _, axes = ev.load_taxonomy()
    examples = example_labels(taxonomy, require_verified)
    report = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "teacher_axes": TEACHER_AXES,
              "splits": {}}

    for split in SPLITS:
        teacher = {r["id"]: r["output"] for r in rb.read_jsonl(teacher_path(split))} if teacher_path(split).exists() else {}
        items = rb.read_jsonl(PRIVATE / f"{split}.jsonl")
        kept = [i for i in items if i["id"] in teacher or i["id"] in examples]
        if len(kept) < len(items):
            print(f"{split}: {len(items) - len(kept)} items have no teacher label yet and are left out")
        distilled = [{**i, "labels": distil_labels(i, teacher.get(i["id"]), examples.get(i["id"]), axes)} for i in kept]
        bd.write_both(f"{split}_distilled", distilled)

        stats = {"items": len(distilled), "from_examples": sum(i["id"] in examples for i in kept)}
        for axis in TEACHER_AXES:
            key = lambda v: "+".join(v) if isinstance(v, list) else str(v)
            new = [i["labels"][axis] for i in distilled]
            old = {i["id"]: i["labels"][axis] for i in kept}
            silver_present = [(o, n) for (o, n) in zip(old.values(), new) if o is not None]
            stats[axis] = {
                "teacher_null": sum(v is None for v in new),
                "distribution": dict(Counter(map(key, new)).most_common()),
                "silver_distribution": dict(Counter(map(key, old.values())).most_common()),
                "agreement_with_silver_where_present": (
                    sum(key(o) == key(n) for o, n in silver_present) / len(silver_present) if silver_present else None),
            }
        report["splits"][split] = stats
        print(f"{split}: {stats['items']} items; " + "; ".join(
            f"{a}: {stats[a]['teacher_null']} null, agrees with silver on "
            f"{stats[a]['agreement_with_silver_where_present'] or 0:.0%}" for a in TEACHER_AXES))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {PRIVATE}/{{train,val}}_distilled.jsonl, public views, and {REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    p_label = commands.add_parser("label", help="label a split with the few-shot teacher")
    p_label.add_argument("split", choices=SPLITS)
    p_label.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p_label.add_argument("--limit", type=int, help="first N items only (smoke tests)")
    commands.add_parser("build", help="write the distilled training and validation files")
    args = parser.parse_args()
    if args.command == "label":
        label(args.split, args.batch_size, args.limit)
    else:
        build()


if __name__ == "__main__":
    main()
