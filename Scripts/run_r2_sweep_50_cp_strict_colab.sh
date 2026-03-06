#!/usr/bin/env bash
set -euo pipefail

# Usage in Colab terminal:
#   bash Scripts/run_r2_sweep_50_cp_strict_colab.sh
#
# 参数扫描说明：
#   tau_iou          x  r2_min_support_ratio  =  2 x 3 = 6 组
#   {0.10, 0.05}     x  {1.0, 0.8, 0.6}
#
# 为什么用 64^3 做 sweep：
#   IoU = token_vol / anatomy_vol，分子分母同比例缩放，
#   64^3 和 128^3 下 tau 的最优区间相同（resolution-invariant）。
#   用 64^3 速度约快 4-8x，参数确定后主实验再上 128^3。
#
# 语义 encoder 说明：
#   hash encoder（默认）是无语义的随机向量，routing 等同于随机，
#   必须用 semantic 才能让 router 有效工作。

CTRATE_CSV="/content/drive/MyDrive/Data/manifests/ctrate_manifest.csv"
RADGENOME_CSV="/content/drive/MyDrive/Data/manifests/radgenome_manifest.csv"
ENCODER_CKPT="/content/drive/MyDrive/Data/checkpoints/swinunetr.ckpt"
OUT_ROOT="/content/drive/MyDrive/Data/outputs_stage0_4_r2sweep_50_cp_strict"

# 2D sweep grid
TAU_VALUES=("0.10" "0.05")
RATIOS=("1.0" "0.8" "0.6")

mkdir -p "${OUT_ROOT}"

for tau in "${TAU_VALUES[@]}"; do
  # 生成目录用的 tau 字符串，去掉小数点 (0.10 -> t010, 0.05 -> t005)
  tau_str="t$(echo "${tau}" | tr -d '.')"

  for ratio in "${RATIOS[@]}"; do
    OUT_DIR="${OUT_ROOT}/r2_tau${tau_str}_ratio_${ratio}"
    mkdir -p "${OUT_DIR}"

    echo "=== [tau=${tau}, ratio=${ratio}] Probe ckpt ==="
    python Scripts/ckpt_probe.py \
      --ckpt_path "${ENCODER_CKPT}" \
      --in_channels 1 \
      --out_channels 2 \
      --feature_size 48 \
      --save_report "${OUT_DIR}/ckpt_probe_report.json"

    echo "=== [tau=${tau}, ratio=${ratio}] Run 50/50 ==="
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
      --r2_mode ratio \
      --r2_min_support_ratio "${ratio}"

    echo "=== [tau=${tau}, ratio=${ratio}] Validate ==="
    python validate_stage0_4_outputs.py \
      --out_dir "${OUT_DIR}" \
      --datasets ctrate,radgenome \
      --expected_cases_map ctrate=50,radgenome=50 \
      --save_report "${OUT_DIR}/validation_report.json"

    echo "=== [tau=${tau}, ratio=${ratio}] Done ==="
  done
done

echo ""
echo "=== All 6 sweep runs done: ${OUT_ROOT} ==="
echo "Summarize with:"
echo "  python Scripts/summarize_r2_sweep.py \\"
echo "    --sweep_root ${OUT_ROOT} \\"
echo "    --glob 'r2_tau*_ratio_*' \\"
echo "    --save_csv ${OUT_ROOT}/sweep_summary.csv"
