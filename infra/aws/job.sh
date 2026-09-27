#!/usr/bin/env bash
# The remaining model steps, run on the EC2 GPU machine by its boot script:
# finish the teacher's labels, build the distilled splits, train the student,
# and predict the test set. Every step resumes from what is already in S3, so
# if the machine dies, `terraform apply` a new one and it carries on.
#
# Progress goes to S3 every 5 minutes and at the end; the final marker is
# s3://$BUCKET/DONE on success or s3://$BUCKET/FAILED otherwise, after which
# the machine shuts down. Never add `set -x` here: it would print the token.
set -euo pipefail
# shellcheck source=/dev/null
source /etc/wayfinder.env # BUCKET, REGION, TOKEN_PARAMETER, DATA_REPO

S3="s3://${BUCKET}"
REPO="${HOME}/wayfinder"
LOG="${HOME}/job.log"
PY="${REPO}/.venv/bin/python"
export AWS_DEFAULT_REGION="${REGION}"
exec > >(tee -a "${LOG}") 2>&1
cd "${REPO}"
echo "job started $(date -u +%FT%TZ) at commit $(git rev-parse --short HEAD)"

sync_up() {
  aws s3 sync results/predictions "${S3}/predictions" --only-show-errors || true
  if [ -d outputs ]; then aws s3 sync outputs "${S3}/outputs" --only-show-errors || true; fi
  aws s3 cp "${LOG}" "${S3}/logs/job.log" --only-show-errors || true
}

finish() {
  local status=$?
  if [ -n "${SYNC_PID:-}" ]; then kill "${SYNC_PID}" 2>/dev/null || true; fi
  sync_up
  local marker=DONE
  if [ "${status}" -ne 0 ]; then marker=FAILED; fi
  echo "${marker} $(date -u +%FT%TZ) exit=${status}" | aws s3 cp - "${S3}/${marker}" || true
  echo "${marker}; shutting down in 1 minute"
  sudo shutdown -h +1
}
trap finish EXIT

# --- Code and data -----------------------------------------------------------

if [ ! -d data/private/.git ]; then
  token="$(aws ssm get-parameter --name "${TOKEN_PARAMETER}" --with-decryption \
    --query Parameter.Value --output text)"
  git clone --depth 1 "https://x-access-token:${token}@github.com/${DATA_REPO}.git" data/private
  # Do not leave the token in the clone's config on disk.
  git -C data/private remote set-url origin "https://github.com/${DATA_REPO}.git"
  unset token
fi

if [ ! -x "${PY}" ]; then
  python3.12 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  # On Linux, the default torch wheel from PyPI bundles CUDA.
  .venv/bin/pip install --quiet torch
  .venv/bin/pip install --quiet -r requirements.txt -r requirements-gpu.txt
  # Not used here, and an old version breaks peft's adapter loading (see the notebook).
  .venv/bin/pip uninstall --quiet -y torchao 2>/dev/null || true
fi
"${PY}" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, torch.cuda.get_device_name())"

# --- Wait for the inputs ---------------------------------------------------

# The teacher's partial output is uploaded after the machine exists; starting
# before it arrives would re-label from scratch.
echo "waiting for ${S3}/READY"
until aws s3 ls "${S3}/READY" >/dev/null 2>&1; do sleep 60; done
aws s3 sync "${S3}/predictions" results/predictions --only-show-errors
aws s3 sync "${S3}/outputs" outputs --only-show-errors 2>/dev/null || true
echo "teacher items already done: $(wc -l < results/predictions/teacher-train.jsonl 2>/dev/null || echo 0)"

( while true; do sleep 300; sync_up; done ) &
SYNC_PID=$!

# --- Steps (each resumes or skips if already done) --------------------------

"${PY}" src/distill.py label train --batch-size 8
"${PY}" src/distill.py label val --batch-size 8
"${PY}" src/distill.py build

if [ ! -f outputs/lora-distilled/training_summary.json ]; then
  "${PY}" src/train_lora.py train \
    --labels data/private/processed/train_distilled.jsonl \
    --val-labels data/private/processed/val_distilled.jsonl \
    --output-dir outputs/lora-distilled --resume
fi

"${PY}" src/train_lora.py predict --output-dir outputs/lora-distilled --run-name lora-distilled
cp outputs/lora-distilled/training_summary.json results/predictions/lora-distilled.training_summary.json
cp results/distill_report.json results/predictions/distill_report.json
echo "all steps finished $(date -u +%FT%TZ)"
