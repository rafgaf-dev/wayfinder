"""Run the baselines on the test split and write their predictions.

  majority   The most common silver label per axis in train. A floor: a method
             that cannot beat it has learned nothing. Runs anywhere.
  zero_shot  Qwen2.5-3B-Instruct with the full taxonomy in the prompt.
  few_shot   The same, plus the 12 hand-checked examples as earlier turns.

The model runs need a GPU (a free Colab T4 is enough) and requirements-gpu.txt.
Predictions go to results/predictions/<run>.jsonl in the format evaluate.py
reads, next to a <run>.meta.json that records exactly what produced them.
Model runs append one line per item, so if a Colab session dies, running the
same command again resumes where it stopped.

Every model run decodes the same way, including the fine-tune, whose
predictions come from generate_predictions() below, called by train_lora.py:
  - Greedy decoding, with a GenerationConfig built from scratch. Qwen's own
    generation_config.json sets sampling parameters and repetition_penalty
    1.05. The penalty would still apply under greedy decoding, and it
    penalises exactly what a JSON answer must repeat (quotes, commas, the
    shared label vocabulary), so it is switched off explicitly.
  - No constrained decoding. A method that fails to produce valid JSON is
    wrong, and the evaluator reports how often that happens.
  - Batch size 1, timed with the device synchronised around each call and
    after a short warm-up, so latency_s is the real time for one request.
    Batching would raise throughput for every method alike; it would not
    change which method is cheaper per item.
  - fp16 weights: the T4 has no bf16 support.

Usage:
    python src/run_baseline.py majority
    python src/run_baseline.py zero_shot
    python src/run_baseline.py few_shot
    python src/run_baseline.py zero_shot --limit 20   # smoke test, separate run name
"""

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import prompts

REPO_ROOT = Path(__file__).resolve().parents[1]
# The majority baseline needs labels only, so it reads the public splits and
# anyone can reproduce it. The model runs need the text, in the private ones.
PUBLIC_TRAIN_PATH = REPO_ROOT / "data" / "processed" / "train.jsonl"
PUBLIC_TEST_PATH = REPO_ROOT / "data" / "processed" / "test.jsonl"
TEST_PATH = REPO_ROOT / "data" / "private" / "processed" / "test.jsonl"
PREDICTIONS_DIR = REPO_ROOT / "results" / "predictions"

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
# The answer is about 60 tokens. The headroom lets a rambling model finish a
# sentence and fail on its own terms instead of being cut off mid-answer.
MAX_NEW_TOKENS = 160
WARMUP_ITEMS = 3

# Fields that must match for a resumed run to append to an existing one.
RESUME_KEYS = ("method", "model", "model_revision", "system_prompt_sha256", "fewshot_ids", "decoding")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --- Majority ---------------------------------------------------------------

def majority_labels(train: list[dict], taxonomy: dict) -> dict:
    """The most common non-null silver value per axis. For mood, the most
    common whole set, so the answer is a set some item actually had."""
    labels = {}
    for axis in taxonomy["axes"]:
        values = Counter(tuple(sorted(v)) if isinstance(v, list) else v
                         for i in train if (v := i["labels"][axis]) is not None)
        top = values.most_common(1)[0][0]
        labels[axis] = list(top) if isinstance(top, tuple) else top
    return labels


def run_majority(taxonomy: dict) -> None:
    train, test = read_jsonl(PUBLIC_TRAIN_PATH), read_jsonl(PUBLIC_TEST_PATH)
    labels = majority_labels(train, taxonomy)
    answer = prompts.format_answer(labels, taxonomy)
    path = PREDICTIONS_DIR / "majority.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in test:
            f.write(json.dumps({"id": item["id"], "output": answer}) + "\n")
    path.with_suffix(".meta.json").write_text(json.dumps({
        "method": "majority", "labels": labels, "source": "most common silver label per axis in train",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"majority answer for all {len(test)} items: {answer}\nwrote {path}")


# --- Model runs -------------------------------------------------------------

def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def synchronise(device: str) -> None:
    import torch
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def load_model(model_id: str = MODEL_ID):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device, dtype = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    model.eval()
    return model, tokenizer, device


def generation_config(model, tokenizer):
    from transformers import GenerationConfig
    return GenerationConfig(
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS,
        repetition_penalty=1.0,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )


def decoding_settings() -> dict:
    return {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "repetition_penalty": 1.0, "batch_size": 1}


def generate_one(model, tokenizer, device: str, messages: list[dict], config) -> dict:
    import torch
    # Render the chat template to text, then tokenise, rather than asking
    # apply_chat_template for tensors: its return type differs between
    # transformers versions, and this way the exact prompt is inspectable.
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    n_input = inputs["input_ids"].shape[1]

    synchronise(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, generation_config=config)
    synchronise(device)
    latency = time.perf_counter() - started

    generated = output[0, n_input:]
    return {
        "output": tokenizer.decode(generated, skip_special_tokens=True),
        "latency_s": round(latency, 4),
        "input_tokens": int(n_input),
        # Includes the end-of-sequence token, which the model did have to generate.
        "output_tokens": int(generated.shape[0]),
    }


def generate_predictions(model, tokenizer, device: str, items: list[dict], make_messages,
                         out_path: Path, meta: dict) -> None:
    """Write one prediction per item to out_path, resuming if it exists.

    `make_messages(item)` returns the chat messages for one item. Shared by
    the baselines and the fine-tune so that every model run is decoded and
    timed identically.
    """
    import torch
    import transformers

    meta = {**meta, "decoding": decoding_settings(), "device": device,
            "device_name": torch.cuda.get_device_name() if device == "cuda" else device,
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__}}
    meta_path = out_path.with_suffix(".meta.json")
    done = {r["id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    if done:
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        mismatched = [k for k in RESUME_KEYS if previous.get(k) != meta.get(k)]
        if mismatched:
            raise SystemExit(f"{out_path} was produced with different {mismatched}; "
                             "use a new --run-name rather than mixing runs")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({**meta, "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                                    indent=2) + "\n", encoding="utf-8")

    config = generation_config(model, tokenizer)
    todo = [item for item in items if item["id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to go on {meta['device_name']}")
    for item in todo[:WARMUP_ITEMS]:  # first calls pay for kernel compilation and allocation
        generate_one(model, tokenizer, device, make_messages(item), config)

    with out_path.open("a", encoding="utf-8") as out:
        for n, item in enumerate(todo, 1):
            record = generate_one(model, tokenizer, device, make_messages(item), config)
            out.write(json.dumps({"id": item["id"], **record}, ensure_ascii=False) + "\n")
            out.flush()
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}  last: {record['latency_s']:.2f}s, "
                      f"{record['input_tokens']} in / {record['output_tokens']} out")


def run_model(template: str, taxonomy: dict, model_id: str, run_name: str, limit: int | None) -> None:
    examples = prompts.load_fewshot(taxonomy) if template == "few_shot" else None
    items = read_jsonl(TEST_PATH)[:limit] if limit else read_jsonl(TEST_PATH)
    model, tokenizer, device = load_model(model_id)
    system = prompts.build_messages(template, "", taxonomy, examples)[0]["content"]
    meta = {
        "method": template,
        "model": model_id,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "system_prompt_sha256": sha256_text(system),
        "fewshot_ids": [e["id"] for e in examples] if examples else None,
        "fewshot_verified": all(e.get("verified") for e in examples) if examples else None,
        "taxonomy_version": taxonomy["version"],
        "limit": limit,
    }
    generate_predictions(model, tokenizer, device, items,
                         lambda item: prompts.build_messages(template, item["text"], taxonomy, examples),
                         PREDICTIONS_DIR / f"{run_name}.jsonl", meta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("method", choices=("majority", "zero_shot", "few_shot"))
    parser.add_argument("--model", default=MODEL_ID, help="override only for local testing")
    parser.add_argument("--limit", type=int, help="run the first N test items only (smoke test)")
    parser.add_argument("--run-name", help="defaults to the method name; --limit runs get their own")
    args = parser.parse_args()
    taxonomy = prompts.load_taxonomy()

    if args.method == "majority":
        run_majority(taxonomy)
        return
    run_name = args.run_name or (f"{args.method}-limit{args.limit}" if args.limit else args.method)
    if args.model != MODEL_ID and not args.run_name:
        run_name += "-" + args.model.split("/")[-1]
    run_model(args.method, taxonomy, args.model, run_name, args.limit)


if __name__ == "__main__":
    main()
