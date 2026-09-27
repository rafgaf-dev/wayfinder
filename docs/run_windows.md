# Running the model steps on a Windows PC

For when Colab's free GPU allowance runs out. These steps were written for an RTX 4060 Ti (8 GB) that also
drives the display, but work on any recent NVIDIA card. Everything is plain terminal commands, since the
scripts in `src/` do all the work. Run them in **PowerShell** from the repo folder.

Two rules keep the results comparable with the Colab runs:
- **Training uses the same settings as on the T4**, including fp16. The 4060 Ti supports bf16, but switching
  would make precision a second difference between the two fine-tunes.
- **The fine-tune's test predictions still run on Colab's T4** (step 5). Its latency and cost are compared
  with the baselines' figures, which were measured on the T4.

## 1. One-time setup

Install a current **NVIDIA driver**, **Git for Windows** (it includes Git Credential Manager, which handles
signing in to GitHub) and **Python 3.12 or 3.13** from python.org (tick "Add python.exe to PATH").

```powershell
git clone https://github.com/rafgaf-dev/wayfinder.git
cd wayfinder
# The private data repo: a browser window opens to sign in to GitHub the first time.
git clone https://github.com/rafgaf-dev/wayfinder-data.git data\private

py -3.13 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
# Install a CUDA build of torch FIRST, or pip installs a CPU-only one as a dependency. Get the exact command
# from https://pytorch.org/get-started/locally/ (Stable, Windows, Pip, CUDA 12.x); it looks like:
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-gpu.txt
.venv\Scripts\python -c "import torch, bitsandbytes; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
```

The last line must print `True` and the card's name. Calling `.venv\Scripts\python` directly avoids activating
the virtual environment, which PowerShell's script policy often blocks.

The first model step downloads Qwen2.5-3B-Instruct (about 6 GB) into `%USERPROFILE%\.cache\huggingface`.

## 2. Memory: the display shares the card

Windows reserves part of the card's memory for the display, and newer NVIDIA drivers can quietly spill over
into system RAM when the card is full. That makes a run crawl instead of failing with a clear error.
- Close anything else using the GPU (browsers with hardware acceleration, games, video).
- Recommended while testing: NVIDIA Control Panel → Manage 3D settings → **CUDA – Sysmem Fallback Policy →
  Prefer No Sysmem Fallback**. A run that doesn't fit then stops with "CUDA out of memory" instead of slowing
  down. Set it back afterwards if you like.
- If the CPU has integrated graphics, plugging the monitor into the motherboard frees the card entirely.
- `nvidia-smi` in a second PowerShell window shows memory use while a step runs.

To keep the PC awake overnight while it's plugged in, set Settings → System → Power → "Sleep" to Never for
the duration, or run `powercfg /change standby-timeout-ac 0` (and change it back afterwards).

## 3. Finish the teacher labelling

The teacher (few-shot Qwen, fp16) labels the training items. The Colab run's progress carries over.

1. From Google Drive, `MyDrive/wayfinder/predictions/`, copy `teacher-train.jsonl` and `teacher-train.meta.json`
   into `results\predictions\`.
2. Check how many items are already done (about 1,200 when this was written):
   ```powershell
   (Get-Content results\predictions\teacher-train.jsonl).Count
   ```
3. Continue with small batches, which the fp16 model needs on 8 GB:
   ```powershell
   .venv\Scripts\python src\distill.py label train --batch-size 2
   ```
   Within a minute it prints a line like `8/3202  0.40 items/s, ~130 min left`.
   - **"CUDA out of memory"**: run it again with `--batch-size 1`. Nothing is lost: every finished batch is
     already saved.
   - **Far slower than about 0.3 items/s**: the card is probably spilling into system RAM (see step 2). Stop
     with Ctrl+C and either free memory or finish on Colab once the allowance resets.
   - Ctrl+C is always safe. Re-running the same command resumes.
4. Label the 100 validation items:
   ```powershell
   .venv\Scripts\python src\distill.py label val --batch-size 2
   ```
5. Copy `teacher-train.*` and `teacher-val.*` from `results\predictions\` back to the Drive folder, so that
   Colab and the Mac have them.

Each run's `.meta.json` records per-session device, batch size, items and time, so the teacher's cost can be
reported per GPU.

## 4. Build the distilled labels and train the student

```powershell
.venv\Scripts\python src\distill.py build
.venv\Scripts\python src\train_lora.py train --labels data\private\processed\train_distilled.jsonl --val-labels data\private\processed\val_distilled.jsonl --output-dir outputs\lora-distilled
```

- **"CUDA out of memory"** at the start: add `--batch-size 2 --grad-accumulation 8`. That's the same effective
  batch of 16, so the run stays comparable with the silver fine-tune; the script warns if the effective
  batch ever differs.
- If Colab already saved checkpoints for this run, copy `MyDrive/wayfinder/lora-distilled/checkpoints/` into
  `outputs\lora-distilled\checkpoints\` and add `--resume` to continue from the latest one.
- The progress bar shows the time remaining. Checkpoints are saved every 100 steps, so a stopped run resumes
  with `--resume`.

## 5. Predictions on Colab's T4

Upload `outputs\lora-distilled\adapter\` (the whole folder) and `outputs\lora-distilled\training_summary.json`
to `MyDrive/wayfinder/lora-distilled/` on Drive. Then, in the Colab notebook, run the setup cells and the
distilled **predict** cell. It takes about 20 minutes, and its latency is then measured on the same GPU as
every other method.
