#!/usr/bin/env bash
set -euo pipefail

# Run early-fusion experiments for all SWECO variable groups.
# Customize via env vars:
#   BACKBONE, EPOCHS, BATCH_SIZE, NUM_WORKERS, PRETRAINED, RUN_PREFIX

BACKBONE="${BACKBONE:-resnet18}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PRETRAINED="${PRETRAINED:-0}"
RUN_PREFIX="${RUN_PREFIX:-ablation_study}"

GROUPS=(
  geol
  edaph
  vege
  bioclim
  lulc_grasslands
  lulc_all
  hydro
  population
)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for group in "${GROUPS[@]}"; do
  ts="$(date +%Y%m%d_%H%M%S)"
  run_name="${RUN_PREFIX}_${group}_${ts}"
  args=(
    "${ROOT_DIR}/train.py"
    --mode fusion
    --group "${group}"
    --backbone "${BACKBONE}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --run_name "${run_name}"
  )
  if [[ "${PRETRAINED}" == "1" ]]; then
    args+=(--pretrained)
  fi
  echo "[info] running group=${group} backbone=${BACKBONE} epochs=${EPOCHS}"
  python "${args[@]}"
done
