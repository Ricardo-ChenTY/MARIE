#!/usr/bin/env bash
set -euo pipefail

# Usage in Colab terminal:
#   bash Scripts/run_r2_sweep_50_cp_strict_colab.sh
#
# This script runs a quick R2 sweep on 50/50 cases for CP-strict Stage0-4:
#   r2_min_support_ratio = 1.0, 0.8, 0.6

CTRATE_CSV="/content/drive/MyDrive/Data/manifests/ctrate_manifest.csv"
RADGENOME_CSV="/content/drive/MyDrive/Data/manifests/radgenome_manifest.csv"
ENCODER_CKPT="/content/drive/MyDrive/Data/checkpoints/swinunetr.ckpt"
OUT_ROOT="/content/drive/MyDrive/Data/outputs_stage0_4_r2sweep_50_cp_strict"

RATIOS=("1.0" "0.8" "0.6")

mkdir -p "${OUT_ROOT}"

for ratio in "${RATIOS[@]}"; do
  OUT_DIR="${OUT_ROOT}/r2_ratio_${ratio}"
  mkdir -p "${OUT_DIR}"

  echo "=== Probe ckpt for ratio=${ratio} ==="
  python Scripts/ckpt_probe.py \
    --ckpt_path "${ENCODER_CKPT}" \
    --in_channels 1 \
    --out_channels 2 \
    --feature_size 48 \
    --save_report "${OUT_DIR}/ckpt_probe_report.json"

  echo "=== Run 50/50 cp_strict with r2_min_support_ratio=${ratio} ==="
  python run_mini_experiment.py \
    --ctrate_csv "${CTRATE_CSV}" \
    --radgenome_csv "${RADGENOME_CSV}" \
    --out_dir "${OUT_DIR}" \
    --max_cases 50 \
    --expected_cases_per_dataset 50 \
    --cp_strict \
    --encoder_ckpt "${ENCODER_CKPT}" \
    --text_encoder semantic \
    --text_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
    --text_encoder_device cuda \
    --device cuda \
    --token_budget_b 64 \
    --k_per_sentence 8 \
    --lambda_spatial 0.3 \
    --tau_iou 0.1 \
    --beta 0.1 \
    --r2_mode ratio \
    --r2_min_support_ratio "${ratio}"

  echo "=== Validate outputs for ratio=${ratio} ==="
  python validate_stage0_4_outputs.py \
    --out_dir "${OUT_DIR}" \
    --datasets ctrate,radgenome \
    --expected_cases_map ctrate=50,radgenome=50 \
    --save_report "${OUT_DIR}/validation_report.json"
done

echo "=== Sweep done: ${OUT_ROOT} ==="
