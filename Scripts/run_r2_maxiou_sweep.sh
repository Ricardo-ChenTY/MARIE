#!/usr/bin/env bash
set -euo pipefail

# Usage in Colab terminal:
#   bash Scripts/run_r2_maxiou_sweep.sh
#
# R2 max_iou 模式扫描说明：
#   max_iou 模式：只要 cited tokens 中 max(IoU) >= tau 就通过，比 ratio 模式宽松。
#   扫描 tau_iou ∈ {0.10, 0.05, 0.02}，共 3 组。
#   同时关闭 R4（阈值未校准），隔离 R4 干扰。
#
# 对比基准：run_r2_sweep_50_cp_strict_colab.sh (ratio 模式)
# 结果汇总：python Scripts/summarize_r2_sweep.py --sweep_root <OUT_ROOT> --glob "r2_maxiou_tau*"

CTRATE_CSV="/content/drive/MyDrive/Data/manifests/ctrate_manifest.csv"
RADGENOME_CSV="/content/drive/MyDrive/Data/manifests/radgenome_manifest.csv"
ENCODER_CKPT="/content/drive/MyDrive/Data/checkpoints/swinunetr.ckpt"
OUT_ROOT="/content/drive/MyDrive/Data/outputs_stage0_4_r2maxiou_sweep_50"

TAU_VALUES=("0.10" "0.05" "0.02")

mkdir -p "${OUT_ROOT}"

for tau in "${TAU_VALUES[@]}"; do
  tau_str="t$(echo "${tau}" | tr -d '.')"
  OUT_DIR="${OUT_ROOT}/r2_maxiou_tau${tau_str}"
  mkdir -p "${OUT_DIR}"

  echo "=== [max_iou, tau=${tau}] Run 50/50 ==="
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
    --resize_d 64 --resize_h 64 --resize_w 64 \
    --token_budget_b 64 \
    --k_per_sentence 8 \
    --lambda_spatial 0.3 \
    --tau_iou "${tau}" \
    --beta 0.1 \
    --r2_mode max_iou \
    --r4_disabled

  echo "=== [max_iou, tau=${tau}] Validate ==="
  python validate_stage0_4_outputs.py \
    --out_dir "${OUT_DIR}" \
    --datasets ctrate,radgenome \
    --expected_cases_map ctrate=50,radgenome=50 \
    --save_report "${OUT_DIR}/validation_report.json"

  echo "=== [max_iou, tau=${tau}] Done ==="
done

echo ""
echo "=== All 3 max_iou sweep runs done: ${OUT_ROOT} ==="
echo "Summarize with:"
echo "  python Scripts/summarize_r2_sweep.py \\"
echo "    --sweep_root ${OUT_ROOT} \\"
echo "    --glob 'r2_maxiou_tau*' \\"
echo "    --save_csv ${OUT_ROOT}/maxiou_sweep_summary.csv"
