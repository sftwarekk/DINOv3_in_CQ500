# CQ-500 DINOv3 MIL Experiments

Clean code-only runner for comparing DINOv3 on CQ-500 3D CT cases by encoding 2D slices and aggregating them with MIL.

This repo is intentionally separated from the local data/results folder. Put only code in GitHub; pass the CQ-500 paths at runtime.

## Supported Comparisons

- `frozen`: DINOv3 frozen, train only MIL head.
- `linear_probe`: DINOv3 frozen, mean-pool slices, train only a linear classifier.
- `lora`: DINOv3 frozen base with LoRA adapters plus MIL head.
- `partial`: unfreeze the last 4 or 6 transformer blocks plus MIL head.

MIL poolers:

- `abmil`: gated attention MIL.
- `transmil`: compact Transformer MIL.
- `mean`: masked mean pooling.

## Expected CQ-500 CSV

The existing project CSV works:

```text
name,label,series_dir,num_slices,fold,split
CQ500-CT-1,1,datase/256x256/extracted/...,36,0,train
```

The loader also accepts common aliases such as `case_id`, `target`, `path`, `case_dir`, and handles the existing `datase/...` typo by resolving it relative to `--project-root`.

## Install

```bash
conda create -n dinov3 python=3.10 -y
conda activate dinov3
pip install -e .
```

If the DINOv3 model is gated on Hugging Face, export a token:

```bash
export HF_TOKEN=...
```

## Single Fold Examples

Set these once on your server:

```bash
export PROJECT_ROOT=/data/rhrjs0307/repos/capston/ct_brain_dino
export SPLIT_CSV=$PROJECT_ROOT/cq500_5fold_split_full.csv
```

Frozen backbone + ABMIL:

```bash
python -m cq500_dinov3_mil.train \
  --strategy frozen \
  --pooler abmil \
  --split-csv "$SPLIT_CSV" \
  --project-root "$PROJECT_ROOT" \
  --fold 0 \
  --output-dir outputs/frozen_abmil/fold0 \
  --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
  --image-size 256 \
  --slice-chunk-size 4 \
  --cache-frozen-features \
  --epochs 20 \
  --amp --amp-dtype bf16 \
  --use-pos-weight
```

Linear probe:

```bash
python -m cq500_dinov3_mil.train \
  --strategy linear_probe \
  --pooler mean \
  --split-csv "$SPLIT_CSV" \
  --project-root "$PROJECT_ROOT" \
  --fold 0 \
  --output-dir outputs/linear_probe/fold0 \
  --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
  --image-size 256 \
  --slice-chunk-size 8 \
  --cache-frozen-features \
  --epochs 20 \
  --amp --amp-dtype bf16 \
  --use-pos-weight
```

LoRA + ABMIL:

```bash
python -m cq500_dinov3_mil.train \
  --strategy lora \
  --pooler abmil \
  --split-csv "$SPLIT_CSV" \
  --project-root "$PROJECT_ROOT" \
  --fold 0 \
  --output-dir outputs/lora_abmil/fold0 \
  --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-targets auto \
  --image-size 256 \
  --slice-chunk-size 1 \
  --gradient-checkpointing \
  --epochs 20 \
  --amp --amp-dtype bf16 \
  --use-pos-weight
```

Partial fine-tuning last4 + ABMIL:

```bash
python -m cq500_dinov3_mil.train \
  --strategy partial \
  --pooler abmil \
  --split-csv "$SPLIT_CSV" \
  --project-root "$PROJECT_ROOT" \
  --fold 0 \
  --output-dir outputs/partial_last4_abmil/fold0 \
  --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
  --unfreeze-last-n-blocks 4 \
  --train-norm \
  --gradient-checkpointing \
  --image-size 256 \
  --slice-chunk-size 1 \
  --epochs 20 \
  --lr-backbone 5e-6 \
  --lr-head 3e-4 \
  --amp --amp-dtype bf16 \
  --use-pos-weight
```

Partial fine-tuning last6 + ABMIL:

```bash
python -m cq500_dinov3_mil.train \
  --strategy partial \
  --pooler abmil \
  --split-csv "$SPLIT_CSV" \
  --project-root "$PROJECT_ROOT" \
  --fold 0 \
  --output-dir outputs/partial_last6_abmil/fold0 \
  --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
  --unfreeze-last-n-blocks 6 \
  --train-norm \
  --gradient-checkpointing \
  --image-size 256 \
  --slice-chunk-size 1 \
  --epochs 20 \
  --lr-backbone 2e-6 \
  --lr-head 3e-4 \
  --amp --amp-dtype bf16 \
  --use-pos-weight
```

## Evaluate a Saved Checkpoint

Training writes `checkpoints/best.pt`, `history.json`, validation predictions, and attention CSVs. To re-evaluate:

```bash
python -m cq500_dinov3_mil.evaluate \
  --checkpoint outputs/lora_abmil/fold0/checkpoints/best.pt \
  --split val \
  --output-dir outputs/lora_abmil/fold0/re_eval
```

## 5-Fold SLURM

Use the helper as an array wrapper:

```bash
mkdir -p logs
CONDA_ENV=dinov3 STRATEGY=lora POOLER=abmil OUTPUT_ROOT=outputs/lora_abmil \
  sbatch --array=0-4 slurm/cq500_dinov3_mil_fold.slurm
```

For all-slice LoRA on 12GB GPUs, keep `SLICE_CHUNK_SIZE=1`. If it still OOMs, reduce `IMAGE_SIZE=224` or run LoRA with a sampled-slice ablation.

Submit both partial fine-tuning presets:

```bash
PROJECT_ROOT=/data/rhrjs0307/repos/capston/ct_brain_dino \
  bash slurm/submit_partial_last4_last6.sh
```

## Summarize Folds

```bash
python scripts/summarize_folds.py \
  --run-root outputs/lora_abmil \
  --output-json outputs/lora_abmil_summary.json
```
