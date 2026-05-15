#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-/data/rhrjs0307/repos/capston/ct_brain_dino}
OUTPUT_BASE=${OUTPUT_BASE:-$PROJECT_ROOT/outputs}
PARTIAL_FOLDS=${PARTIAL_FOLDS:-0-4}

mkdir -p "$SCRIPT_DIR/../logs"

submit_partial() {
  local n_blocks="$1"
  local lr_backbone="$2"
  local output_root="$OUTPUT_BASE/partial_last${n_blocks}_abmil"

  echo "[INFO] submit partial fine-tuning last${n_blocks}: output=${output_root}, lr_backbone=${lr_backbone}"
  STRATEGY=partial \
  POOLER=abmil \
  OUTPUT_ROOT="$output_root" \
  UNFREEZE_LAST_N_BLOCKS="$n_blocks" \
  LR_BACKBONE="$lr_backbone" \
  SLICE_CHUNK_SIZE="${SLICE_CHUNK_SIZE:-1}" \
  sbatch --array="$PARTIAL_FOLDS" "$SCRIPT_DIR/cq500_dinov3_mil_fold.slurm"
}

submit_partial 4 "${LAST4_LR_BACKBONE:-5e-6}"
submit_partial 6 "${LAST6_LR_BACKBONE:-2e-6}"
