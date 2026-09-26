"""QLoRA fine-tune of Qwen2.5-3B-Instruct on the compact prompt, then predict.

Two steps, run separately so that a Colab session dying after training does
not cost the adapter:
  train    Fine-tune; writes the LoRA adapter to <output-dir>/adapter.
  predict  Merge the adapter into the fp16 base and write
           results/predictions/<run>.jsonl, decoded and timed exactly like
           the baselines (run_baseline.generate_predictions).

Training choices, and where the practice differs from the theory:
  - QLoRA. The base weights are frozen in 4-bit NF4 with double quantisation
    (the quantisation constants are quantised too, saving about 0.4 bits per
    weight). Adapters (r=16, alpha=32) go on all seven projections: q, k, v,
    o in attention and gate, up, down in the MLP. The MLP holds about two
    thirds of the weights, and adapting only attention tends to underfit
    behavioural changes like an output convention.
  - fp16, not bf16: the T4 has no bf16. fp16's narrow range needs loss
    scaling (Trainer's fp16=True), and the LoRA weights themselves are kept
    in fp32. PyTorch's GradScaler refuses to unscale fp16 gradients, and fp32
    adapters also avoid overflow in the updates.
  - Gradient checkpointing recomputes activations during the backward pass
    instead of storing them: roughly 30% slower, and what lets a batch of 4
    sequences of up to ~700 tokens fit in 16 GB.
  - The loss covers the answer only. Prompt tokens are labelled -100, which
    the loss ignores. Within the answer, any token overlapping the
    placeholder for a null label is masked as well (spans from
    prompts.format_target), so an axis without a label teaches nothing,
    rather than teaching the placeholder. The end-of-turn token is trained,
    so the model learns to stop.
  - Prompt and answer are tokenised separately and joined, which is how
    generation sees them: the chat template ends with
    "<|im_start|>assistant\\n", and the model produces the answer's tokens
    from there. Tokenising the joined string could merge tokens across the
    boundary and train on a tokenisation the model never has to produce.
  - Batches are grouped by length, so padding (wasted compute) stays small.
  - The best checkpoint by validation loss is kept. Validation labels are
    silver, so this guards against overfitting the training labels; it
    cannot tell whether those labels are right. Only the test gold can.

Serving choice: the adapter is merged into the fp16 base rather than served
on top of the 4-bit base. The merged model has exactly the baselines'
architecture and precision, so latency and cost differ from theirs only
through prompt length. One caveat: the adapter learned to correct the
*quantised* weights, and merging applies it to the unquantised ones. That is
usually negligible. `predict --serve quantised` runs the other way, so the
difference can be measured rather than assumed.

Usage (on a GPU; see notebooks/train_colab.ipynb):
    python src/train_lora.py train --output-dir /content/drive/MyDrive/wayfinder/lora
    python src/train_lora.py predict --output-dir /content/drive/MyDrive/wayfinder/lora
Train on other labels (e.g. distilled) with --labels PATH; give the run its
own --output-dir and --run-name.
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import prompts
import run_baseline as rb

REPO_ROOT = Path(__file__).resolve().parents[1]
# The private splits: training and prediction need the text.
TRAIN_PATH = REPO_ROOT / "data" / "private" / "processed" / "train.jsonl"
VAL_PATH = REPO_ROOT / "data" / "private" / "processed" / "val.jsonl"
TEST_PATH = REPO_ROOT / "data" / "private" / "processed" / "test.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "lora"

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
EPOCHS = 2
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRAD_ACCUMULATION = 4  # effective batch of 16
WARMUP = 0.03  # a fraction of total steps
MAX_LENGTH = 1024  # the longest example is ~700 tokens; longer is a bug, not a case to truncate
EVAL_STEPS = 100
SEED = 13
END_OF_TURN = "<|im_end|>"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- Data -------------------------------------------------------------------

def encode(item: dict, tokenizer, taxonomy: dict, end_id: int) -> dict:
    """Token ids and loss labels for one training example."""
    messages = prompts.build_messages("compact", item["text"], taxonomy)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    target, spans = prompts.format_target(item["labels"], taxonomy)

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer = tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
    answer_labels = []
    for token, (start, end) in zip(answer["input_ids"], answer["offset_mapping"]):
        # Mask any token that overlaps a placeholder, even partly. That can
        # hide a neighbouring quote or space as well, which costs nothing:
        # every other example teaches the JSON structure.
        overlaps = any(start < span_end and end > span_start for span_start, span_end in spans)
        answer_labels.append(-100 if overlaps else token)

    input_ids = prompt_ids + answer["input_ids"] + [end_id]
    labels = [-100] * len(prompt_ids) + answer_labels + [end_id]
    if len(input_ids) > MAX_LENGTH:
        raise ValueError(f"{item['id']}: {len(input_ids)} tokens exceeds MAX_LENGTH={MAX_LENGTH}")
    return {"input_ids": input_ids, "labels": labels}


def encode_all(items: list[dict], tokenizer, taxonomy: dict) -> tuple[list[dict], dict]:
    end_id = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    encoded = [encode(item, tokenizer, taxonomy, end_id) for item in items]
    answer_tokens = sum(sum(1 for i, l in zip(e["input_ids"], e["labels"]) if l != -100) for e in encoded)
    masked = sum(len(prompts.format_target(item["labels"], taxonomy)[1]) for item in items)
    lengths = sorted(len(e["input_ids"]) for e in encoded)
    stats = {
        "examples": len(encoded),
        "trained_tokens": answer_tokens,
        "placeholder_values_masked": masked,
        "tokens_median": lengths[len(lengths) // 2],
        "tokens_max": lengths[-1],
    }
    return encoded, stats


class Collator:
    """Right-pads a batch: pad tokens in input_ids, -100 in labels, 0 in the mask."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch
        width = max(len(b["input_ids"]) for b in batch)

        def pad(seq, value):
            return seq + [value] * (width - len(seq))

        return {
            "input_ids": torch.tensor([pad(b["input_ids"], self.pad_id) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
            "attention_mask": torch.tensor([pad([1] * len(b["input_ids"]), 0) for b in batch]),
        }


# --- Model ------------------------------------------------------------------

def load_for_training(model_id: str, quantise: bool):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if quantise:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, dtype=torch.float16,
                                                     device_map={"": 0})
        # Casts the remaining non-quantised layers (norms) to fp32 and
        # enables gradient checkpointing with input gradients, which frozen
        # 4-bit weights otherwise block.
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        # Unquantised fp32, for smoke tests on a CPU or Mac only: bitsandbytes
        # 4-bit needs CUDA.
        device, _ = rb.pick_device()
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    model.config.use_cache = False  # the KV cache is useless in training and clashes with checkpointing
    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
        target_modules=list(LORA_TARGETS), task_type="CAUSAL_LM",
    ))
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    return model


def train(args) -> None:
    import torch
    from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(SEED)
    taxonomy = prompts.load_taxonomy()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_items = rb.read_jsonl(args.labels)[:args.limit_train] if args.limit_train else rb.read_jsonl(args.labels)
    val_items = rb.read_jsonl(VAL_PATH)
    train_set, train_stats = encode_all(train_items, tokenizer, taxonomy)
    val_set, _ = encode_all(val_items, tokenizer, taxonomy)
    print(f"train: {train_stats}")

    quantise = not args.no_4bit
    if quantise and not torch.cuda.is_available():
        raise SystemExit("4-bit QLoRA needs a CUDA GPU; use --no-4bit only for smoke tests")
    model = load_for_training(args.model, quantise)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {trainable:,} of {total:,} ({trainable / total:.2%})")

    output_dir = Path(args.output_dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=EPOCHS,
        max_steps=args.max_steps or -1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP,
        weight_decay=0.0,
        optim="adamw_torch",
        fp16=quantise,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        train_sampling_strategy="group_by_length",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=SEED,
    )
    trainer = Trainer(model=model, args=training_args, data_collator=Collator(tokenizer.pad_token_id),
                      train_dataset=train_set, eval_dataset=val_set, processing_class=tokenizer)

    checkpoints = list((output_dir / "checkpoints").glob("checkpoint-*"))
    started = time.time()
    trainer.train(resume_from_checkpoint=bool(checkpoints) and args.resume)
    runtime = time.time() - started

    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "labels_file": str(Path(args.labels).relative_to(REPO_ROOT)) if Path(args.labels).is_relative_to(REPO_ROOT) else str(args.labels),
        "labels_sha256": sha256_file(Path(args.labels)),
        "quantised_4bit": quantise,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "targets": list(LORA_TARGETS)},
        "optimisation": {"epochs": EPOCHS, "max_steps": args.max_steps, "learning_rate": LEARNING_RATE,
                         "batch_size": BATCH_SIZE, "grad_accumulation": GRAD_ACCUMULATION, "warmup": WARMUP,
                         "scheduler": "cosine", "seed": SEED},
        "data": train_stats,
        "trainable_parameters": trainable,
        "train_runtime_s": round(runtime, 1),
        "best_eval_loss": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu/mps",
        "log_history": trainer.state.log_history,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"adapter saved to {adapter_dir}; best eval loss {trainer.state.best_metric}; {runtime / 60:.1f} min")


# --- Predict ----------------------------------------------------------------

def load_for_serving(model_id: str, adapter_dir: Path, serve: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if serve == "merged":
        device, dtype = rb.pick_device()
        base = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
        model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    else:
        if not torch.cuda.is_available():
            raise SystemExit("--serve quantised needs a CUDA GPU")
        device = "cuda"
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.float16)
        base = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, dtype=torch.float16,
                                                    device_map={"": 0})
        model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.config.use_cache = True
    model.eval()
    return model, tokenizer, device


def predict(args) -> None:
    taxonomy = prompts.load_taxonomy()
    output_dir = Path(args.output_dir)
    summary = json.loads((output_dir / "training_summary.json").read_text())
    model, tokenizer, device = load_for_serving(summary["model"], output_dir / "adapter", args.serve)
    items = rb.read_jsonl(TEST_PATH)[:args.limit] if args.limit else rb.read_jsonl(TEST_PATH)
    system = prompts.build_messages("compact", "", taxonomy)[0]["content"]
    run_name = args.run_name or ("lora" if args.serve == "merged" else "lora-quantised")
    if args.limit and not args.run_name:
        run_name += f"-limit{args.limit}"
    meta = {
        "method": "lora",
        "model": summary["model"],
        "model_revision": getattr(model.config, "_commit_hash", None),
        "system_prompt_sha256": rb.sha256_text(system),
        "fewshot_ids": None,
        "serve": args.serve,
        "taxonomy_version": taxonomy["version"],
        "limit": args.limit,
        "training": {k: summary[k] for k in ("labels_file", "labels_sha256", "lora", "optimisation", "data",
                                              "trainable_parameters", "train_runtime_s", "best_eval_loss",
                                              "device_name")},
    }
    rb.generate_predictions(model, tokenizer, device, items,
                            lambda item: prompts.build_messages("compact", item["text"], taxonomy),
                            rb.PREDICTIONS_DIR / f"{run_name}.jsonl", meta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    p_train = commands.add_parser("train")
    p_train.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p_train.add_argument("--labels", type=Path, default=TRAIN_PATH, help="training items with labels")
    p_train.add_argument("--model", default=rb.MODEL_ID)
    p_train.add_argument("--resume", action="store_true", help="continue from the latest checkpoint")
    p_train.add_argument("--eval-steps", type=int, default=EVAL_STEPS)
    p_train.add_argument("--max-steps", type=int, help="stop early (smoke tests)")
    p_train.add_argument("--limit-train", type=int, help="first N training items only (smoke tests)")
    p_train.add_argument("--no-4bit", action="store_true", help="unquantised; smoke tests on CPU/Mac only")

    p_predict = commands.add_parser("predict")
    p_predict.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p_predict.add_argument("--serve", choices=("merged", "quantised"), default="merged")
    p_predict.add_argument("--run-name")
    p_predict.add_argument("--limit", type=int, help="first N test items only (smoke tests)")

    args = parser.parse_args()
    {"train": train, "predict": predict}[args.command](args)


if __name__ == "__main__":
    main()
