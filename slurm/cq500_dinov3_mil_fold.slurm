#!/usr/bin/env bash
#SBATCH --job-name=cq500_dinov3_mil
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cq500_dinov3_mil_%A_%a.out
#SBATCH --error=logs/cq500_dinov3_mil_%A_%a.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/rhrjs0307/repos/capston/ct_brain_dino}
SPLIT_CSV=${SPLIT_CSV:-$PROJECT_ROOT/cq500_5fold_split_full.csv}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/cq500_dinov3_mil}
PYTHON=${PYTHON:-python}

FOLD=${SLURM_ARRAY_TASK_ID:-0}
STRATEGY=${STRATEGY:-frozen}
POOLER=${POOLER:-abmil}
MODEL_NAME=${MODEL_NAME:-facebook/dinov3-vitb16-pretrain-lvd1689m}

IMAGE_SIZE=${IMAGE_SIZE:-256}
MAX_SLICES=${MAX_SLICES:-0}
SLICE_CHUNK_SIZE=${SLICE_CHUNK_SIZE:-4}
BATCH_SIZE=${BATCH_SIZE:-1}
EPOCHS=${EPOCHS:-20}
NUM_WORKERS=${NUM_WORKERS:-4}

LR_HEAD=${LR_HEAD:-3e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}

mkdir -p logs "$OUTPUT_ROOT"

EXTRA_ARGS=()
if [[ "$STRATEGY" == "frozen" || "$STRATEGY" == "linear_probe" ]]; then
  if [[ "${CACHE_FROZEN_FEATURES:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--cache-frozen-features)
  fi
fi
if [[ "$STRATEGY" == "lora" ]]; then
  LR_BACKBONE=${LR_BACKBONE:-1e-4}
  EXTRA_ARGS+=(--lora-r "${LORA_R:-8}" --lora-alpha "${LORA_ALPHA:-16}" --lora-targets "${LORA_TARGETS:-auto}")
fi
if [[ "$STRATEGY" == "partial" ]]; then
  UNFREEZE_LAST_N_BLOCKS=${UNFREEZE_LAST_N_BLOCKS:-4}
  if [[ -z "${LR_BACKBONE+x}" ]]; then
    if [[ "$UNFREEZE_LAST_N_BLOCKS" -ge 6 ]]; then
      LR_BACKBONE=2e-6
    else
      LR_BACKBONE=5e-6
    fi
  fi
  EXTRA_ARGS+=(--unfreeze-last-n-blocks "$UNFREEZE_LAST_N_BLOCKS" --train-norm --gradient-checkpointing)
fi
LR_BACKBONE=${LR_BACKBONE:-1e-5}

"$PYTHON" -m cq500_dinov3_mil.train \
  --strategy "$STRATEGY" \
  --pooler "$POOLER" \
  --split-csv "$SPLIT_CSV" \
  --project-root "$PROJECT_ROOT" \
  --fold "$FOLD" \
  --output-dir "$OUTPUT_ROOT/fold${FOLD}" \
  --model-name "$MODEL_NAME" \
  --image-size "$IMAGE_SIZE" \
  --max-slices "$MAX_SLICES" \
  --slice-chunk-size "$SLICE_CHUNK_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --num-workers "$NUM_WORKERS" \
  --lr-backbone "$LR_BACKBONE" \
  --lr-head "$LR_HEAD" \
  --weight-decay "$WEIGHT_DECAY" \
  --amp --amp-dtype bf16 \
  --use-pos-weight \
  "${EXTRA_ARGS[@]}"
