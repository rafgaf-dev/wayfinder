# Wayfinder

**Does fine-tuning a small model earn its place over prompting, for mapping content from different sources
onto one shared taxonomy?**

Wayfinder is a measured answer to that question for a digital entertainment platform organised as a "virtual
city" of districts (sport, gaming, music, social, commerce, live). Content arrives from each district in its
own vocabulary: Steam genres, Ticketmaster event classifications and so on. Cross-district discovery needs one
taxonomy, so that a calm puzzle game and an ambient album come out measurably close. A language model can
already read the descriptions. What it lacks are *this taxonomy's conventions*: for example, that a football
match is `passive` and `spectator` for the person holding the ticket. That makes the problem one of behaviour,
not knowledge, which is where fine-tuning is supposed to help.

So the repository compares five methods on the same frozen test set, scored by the same unmodified evaluation
script: a majority-class floor, zero-shot prompting, few-shot prompting, and two QLoRA fine-tunes of the same
3B model. It reports, per axis, which won and whether the difference justifies the cost.

## The answer

**On the axes scored against independent hand labels, neither fine-tune beats few-shot prompting, and the two
fine-tunes don't differ from each other. What fine-tuning buys is cost: the same quality with a prompt 7×
shorter and responses 2.4× faster.**

Macro-F1 with 95% bootstrap intervals, on the hand-labelled axes (the fair test):

| Axis (gold items) | Majority | Zero-shot | Few-shot | Fine-tune, silver labels | Fine-tune, distilled labels |
|---|---|---|---|---|---|
| mood (250) | 0.10 [0.09, 0.12] | 0.46 [0.41, 0.50] | 0.45 [0.39, 0.49] | 0.47 [0.43, 0.52] | 0.44 [0.39, 0.48] |
| commitment (250) | 0.26 [0.24, 0.27] | 0.36 [0.31, 0.41] | 0.40 [0.34, 0.47] | 0.39 [0.34, 0.45] | 0.39 [0.32, 0.45] |
| district, Ticketmaster (106) | 0.00 | 0.53 [0.42, 0.68] | 0.76 [0.52, 0.89] | 0.61 [0.56, 0.65] | 0.61 [0.56, 0.65] |

Paired bootstrap differences (same resampled items for both methods) show **no clear difference** between
few-shot and either fine-tune on any of these axes. For example, silver fine-tune minus few-shot: mood +0.03
[−0.03, +0.08], commitment −0.01 [−0.08, +0.05]. Distilled minus silver: mood −0.03 [−0.08, +0.01],
commitment −0.01 [−0.07, +0.06].

Cost per item, all measured on an NVIDIA T4:

| | Prompt tokens | Latency, mean (p95) | One-off cost |
|---|---|---|---|
| Few-shot | 3,316 | 6.72 s (7.08 s) | none |
| Zero-shot | 1,311 | 3.50 s (4.07 s) | none |
| Fine-tune, silver | **451** | **2.82 s** (3.37 s) | 2.6 h training |
| Fine-tune, distilled | **451** | 2.09 s (2.23 s)¹ | 2.8 h teacher labelling + 2.2 h training |

¹ Measured on an AWS T4, which ran faster than Colab's T4 throughout (the teacher, for example, ran at 0.43
against 0.38 items/s). The like-for-like latency comparison is the other three rows, all on Colab. Prompt
tokens don't depend on hardware at all.

Every answer from every method parsed: there were 0 invalid outputs across 2,250 predictions.

### What I would ship

**A fine-tune: specifically the distilled one, if the catalogue is large and keeps growing.** Quality is the
same as few-shot on the axes that were independently labelled, and each item costs a seventh of the prompt
tokens. At 3.9 seconds saved per item, the silver fine-tune's one-off cost is recovered after about 2,400 items
and the distilled one's after about 4,600, which is trivial at the scale of a platform catalogue.

Of the two fine-tunes, I'd ship the **distilled** one although it costs more to build. It behaves like its
few-shot teacher (see *Commitment* below), and the teacher's prompt is the one artefact a taxonomy owner can
read and edit. Changing a convention then means editing the prompt and re-running a roughly 5-hour GPU job.
The silver fine-tune scores the same on average, but it learned rules that disagree with human judgement on
Steam commitment 69% of the time, and that shows in the errors it makes.

**Below a few thousand items, or while the taxonomy is still changing week to week, I'd ship few-shot
prompting instead.** It has no training step, and a convention changes the moment the prompt does.

**What I would not claim:** that fine-tuning improves quality here. It didn't, whether trained on rule-based
labels or on labels distilled from few-shot. The axis I expected it to win, commitment, is hard for every method.

## The taxonomy

Seven axes, all single-label except mood, which takes one or two labels
([`taxonomy.yaml`](taxonomy.yaml), with a definition for every label and four cross-axis conventions):

| Axis | Labels |
|---|---|
| energy | low, medium, high |
| social | solo, small-group, crowd, spectator |
| engagement | passive, active, competitive |
| commitment | minutes, an-evening, ongoing |
| time_of_day | morning, afternoon, evening, late-night, any |
| district | sport, gaming, music, social, commerce, live |
| mood | calm, uplifting, intense, melancholic, playful (one or two) |

## Data

| Source | Fetched | Kept | Main exclusions |
|---|---|---|---|
| Steam: store API and SteamSpy user tags | 4,000 games | 3,552 | 267 adult (content descriptors and tags), 105 extra games from a developer already kept, 67 not English |
| Ticketmaster Discovery API, GB, 120 days | 17,363 events | 1,420 | 13,973 without usable prose, 1,513 repeat performances of the same act or show, 393 sport |

- **The model sees the description only.** Titles and event names are left out: a model that recognises
  "Portal 2" is recalling pre-training, not reading. Texts are cut to 1,500 characters in the dataset itself, so
  every method sees identical input.
- **One item per group** (Steam developer; Ticketmaster act or show), so a series or a touring show can't sit on
  both sides of the split.
- **Splits:** 4,422 train, 100 validation, 450 test. The test set is chosen per district (250 gaming, 150 music,
  50 live) so the smallest district still has a usable sample.
- **Sport was dropped.** Ticketmaster's GB listings had prose for only about 6 distinct sporting acts or
  fixtures in 120 days, much of it talks, tours or terms and conditions. `sport`, `social` and `commerce` stay
  in the label space because the product has them; no training data exists for them.
- **Leakage.** An item is flagged, per axis, when its text contains the source term that produced a label (a
  game tagged Roguelike calling itself "a roguelike"). 30% of Steam items and 63% of Ticketmaster items are
  flagged on at least one axis, mostly the word "music". Scores on the clean slice, which drops each flagged
  item from the affected axis only, move by at most 0.02 on the hand-labelled axes, so leakage doesn't drive
  the results.

### Labels: silver and gold

- **Silver labels** come from rules that map each source's vocabulary to the taxonomy
  ([`mapping.yaml`](mapping.yaml)). They exist for every item on most axes, but for mood on only 22% of Steam
  and 41% of Ticketmaster items, and for commitment on only 24% of Steam items. Where the rules have no
  basis, the label is left null and masked out of training, rather than guessed.
- **Gold labels:** 250 test items hand-labelled for mood and commitment, plus district for Ticketmaster items.
  Labelling used the text alone, with no silver labels or model output in view. Four hours for all 450
  weren't available, so the protocol was: **the first 250 rows of the shuffled labelling file, in order, with
  none skipped**. Because the file was shuffled, that's a random sample of the test set. It gives intervals of
  about ±6 points per run.

How well the silver rules agree with the hand labels, where the rules produced a label:

| Axis | Source | Rules gave a label | Agrees with hand label |
|---|---|---|---|
| commitment | Steam | 39 of 144 | **31%** |
| commitment | Ticketmaster | 106 of 106 (always `an-evening`) | 82% |
| mood | Steam | 33 of 144 | 48% |
| mood | Ticketmaster | 44 of 106 | 48% |
| district | Ticketmaster | 106 of 106 | 91% (so 9% label noise in the source's segments) |

Steam tags describe *how* a game is played (roguelike, sandbox), not how long it asks for. That gap is the
reason for the second, distilled fine-tune.

## Methods

Every model method uses **Qwen2.5-3B-Instruct**, so the method is the only thing that changes. Decoding is
identical for all of them: greedy, with the repetition penalty set to 1.0 explicitly, because Qwen's defaults
would penalise the repeated tokens a JSON answer needs. There's no constrained decoding (invalid JSON counts as
wrong), batch size is 1, and the device is synchronised around each call for timing.

| Method | Prompt | Notes |
|---|---|---|
| Majority | none | The most common silver label per axis in train. |
| Zero-shot | Full taxonomy: platform context, conventions, every definition | The rules written down, not bare label names. |
| Few-shot | Zero-shot plus 12 fixed examples as earlier turns | Chosen from train to cover the most labels, then **checked by hand** (3 labels corrected). |
| Fine-tune, silver | **Compact**: axis names and label lists only | QLoRA on the silver labels. |
| Fine-tune, distilled | Compact | QLoRA on labels from the few-shot model (the teacher) for mood and commitment. |

**Why the compact prompt:** moving the conventions from the prompt into the weights is the whole point of
fine-tuning. A fine-tune served with the full taxonomy prompt would pay the same prompt tokens as zero-shot on
every request.

**QLoRA** ([`src/train_lora.py`](src/train_lora.py)):
- 4-bit NF4 base with double quantisation.
- LoRA r=16, alpha 32, on all seven projections (q, k, v, o, gate, up, down): 29.9M trainable parameters.
- fp16 with loss scaling, because the T4 has no bf16. Adapters are kept in fp32.
- 2 epochs, learning rate 2e-4 with a cosine schedule, effective batch 16, gradient checkpointing.
- The loss covers answer tokens only. **An axis with a null label is masked out of the loss per axis:** the
  answer string still contains a placeholder for it, but those tokens teach nothing. Mood comes last in the
  answer so that the axis most often null never conditions another.
- The best checkpoint by validation loss is kept.
- For serving, the adapter is merged into the fp16 base, so the fine-tunes run with exactly the baselines'
  architecture and precision.

**Distillation** ([`src/distill.py`](src/distill.py)): the few-shot model labelled all 4,422 training items.
Only mood and commitment were replaced. The other axes stay silver, because the test set scores them against
silver labels, and teaching anything else there would be marked down as error. The 12 few-shot examples kept
their hand-checked labels. The teacher produced a valid answer for 4,409 of 4,410 items, and agrees with the
silver rules on only 38% of mood labels.

**Evaluation** ([`src/evaluate.py`](src/evaluate.py), written and tested before any model ran):
- **Macro-F1 is the headline metric,** averaged over the labels present in the gold data, so the denominator
  can't change between runs. Accuracy and mood's Jaccard overlap are reported too.
- Parsing happens in the evaluator, identically for every method.
- A missing prediction is wrong, not dropped.
- Intervals are 95% percentile bootstraps, and comparisons use a paired bootstrap.
- Every results file records the sha256 of `evaluate.py`, the test split and the taxonomy. All five runs match.

## Results in detail

### Where prompting helps: conventions

Few-shot beats zero-shot clearly on engagement (+0.17 macro-F1), time of day (+0.16), social (+0.08) and
Ticketmaster district (+0.23). These are the axes defined by conventions particular to this taxonomy, and the
examples demonstrate them: a match is `passive`, comedy is `live` rather than `music`. It makes no difference
on mood or commitment.

### Commitment is hard for every method

Every method scores 0.36–0.40 macro-F1 against a floor of 0.26, and none reliably separates `minutes` from
`an-evening`:
- **Zero-shot** never predicts `ongoing` (0 of 42) and calls almost everything `an-evening`.
- **Few-shot** starts using `ongoing` but over-uses it: 46 `an-evening` items become `ongoing`.
- **The silver fine-tune** over-uses `ongoing` even more (52), which is the tag rules' bias, learned.
- **The distilled fine-tune** behaves like its teacher: 35 `an-evening` items become `ongoing`.

The two fine-tunes make *different* errors that average to the same macro-F1: silver is better on `ongoing`
(F1 0.38 against 0.30), distilled on `minutes` (0.15 against 0.11). My expectation that commitment would be
where fine-tuning earns its place was wrong. The limiting factor looks like signal in the descriptions,
not the method.

### A caveat on the silver-scored axes

On energy, social, engagement and time of day, the test labels are silver, and both fine-tunes score far
higher than prompting (for example, social 0.78 and 0.77 against 0.47). **That isn't evidence of better
understanding.** The fine-tunes learned the same rules that wrote those test labels. These axes show that
fine-tuning reproduces a labelling function; only the hand-labelled axes measure quality.

| Axis (silver test labels) | Majority | Zero-shot | Few-shot | Fine-tune, silver | Fine-tune, distilled |
|---|---|---|---|---|---|
| energy | 0.19 | 0.50 | 0.49 | 0.69 | 0.70 |
| social | 0.16 | 0.39 | 0.47 | 0.78 | 0.77 |
| engagement | 0.22 | 0.44 | 0.61 | 0.79 | 0.77 |
| time_of_day | 0.18 | 0.48 | 0.64 | 0.89 | 0.83 |

All scores, per slice and per label, with confusion matrices, are in [`results/`](results/).

## Limitations

- **One labeller, and no measure of agreement between labellers.** The gold labels are one person's reading
  of the taxonomy, so human-level agreement on commitment is unknown.
- **250 gold items.** Differences under about 5–6 points on the hand-labelled axes can't be distinguished from
  noise. Live has only about 25 gold items, and `social` only 4, which makes Ticketmaster district's macro-F1
  volatile.
- **Selection in the Ticketmaster data.** Only events with a written description were usable (10–42% by
  segment), which favours smaller venues and promoters.
- **Two source districts.** Sport, social and commerce have no training data.
- **One model size and one training seed.**
- **Latency comes from two T4 hosts** (Colab and AWS). Prompt tokens are the hardware-independent comparison.

## Reproducing

Public data here is ids, labels and flags only. The descriptions are third-party text (publisher store copy;
Ticketmaster event content, which its API terms don't allow republishing), so the full splits live in a
private repository cloned at `data/private/`.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover tests                    # 113 tests
.venv/bin/python src/evaluate.py score results/predictions/few_shot.jsonl
.venv/bin/python src/evaluate.py compare results/predictions/few_shot.jsonl results/predictions/lora.jsonl
```

Scoring and comparing work from the public files alone. Rebuilding from scratch:
- `src/fetch_steam.py` needs a Steam Web API key.
- `src/fetch_ticketmaster.py` needs a Ticketmaster key. **Ticketmaster lists only upcoming events, so the
  Ticketmaster half can't be fetched again as it was.**
- Then `src/build_dataset.py`.

Model runs need a GPU:
- [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb): Colab.
- [`docs/run_windows.md`](docs/run_windows.md): a local NVIDIA card.
- [`docs/run_aws.md`](docs/run_aws.md) with [`infra/aws/`](infra/aws/): Terraform for one T4 machine. It uses
  no SSH, keeps progress in S3, and shuts itself down.

## Repository layout

```
taxonomy.yaml, mapping.yaml   the taxonomy, and source-to-taxonomy rules
src/
  fetch_steam.py, fetch_ticketmaster.py, fetch_common.py
  build_dataset.py            clean, filter, label, dedupe, split; freezes the test set
  label_gold.py               terminal tool for hand labelling
  evaluate.py                 per-axis metrics, bootstrap intervals, paired comparisons
  prompts.py                  zero-shot, few-shot and compact prompts; the shared answer format
  run_baseline.py             majority, zero-shot, few-shot; generation shared by every model run
  train_lora.py               QLoRA training and predictions
  distill.py                  few-shot teacher labels, and the distilled training set
data/processed/               public splits: ids, labels, flags
data/gold/                    hand labels, and the checked few-shot examples
results/                      scores, comparisons, predictions, run metadata
tests/                        113 tests, including guards on public outputs and file encodings
notebooks/, docs/, infra/aws/ ways to run the GPU steps
```

## Licence

[MIT](LICENSE), covering everything in this repository: the code, the hand labels, the prompts and the results.
The third-party descriptions the models read are not part of it. They are kept in a private repository
under Steam's and Ticketmaster's terms.
