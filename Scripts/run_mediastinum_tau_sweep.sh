#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════
# Mediastinum 局部阈值 Sweep
# ═══════════════════════════════════════════════════════════════════
# 问题：mediastinum 在当前配置下全部被标记为违规
# 策略：围绕 0.0442 做局部密扫，测试 tau_iou 的 5 个点
# 阈值点：0.03, 0.035, 0.04, 0.045, 0.05
#
# 重点观察指标：
#   - R2_ANATOMY 总数
#   - LLM confirmed 数量
#   - mediastinum confirmed 比例
#   - 总体 violation_sentence_rate
# ═══════════════════════════════════════════════════════════════════

ROOT="/data/ProveTok_ACM"
OUT_ROOT="${ROOT}/outputs/mediastinum_tau_sweep_50"
CACHE_ROOT="${ROOT}/.cache"
HF_HOME_DIR="${ROOT}/.hf"

# 5 个阈值点围绕 0.0442
TAU_VALUES=("0.03" "0.035" "0.04" "0.045" "0.05")

mkdir -p "${OUT_ROOT}"
mkdir -p \
  "${CACHE_ROOT}/huggingface/hub" \
  "${CACHE_ROOT}/huggingface/transformers" \
  "${CACHE_ROOT}/sentence_transformers" \
  "${HF_HOME_DIR}"

export XDG_CACHE_HOME="${CACHE_ROOT}"
export HF_HOME="${HF_HOME_DIR}"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/huggingface/transformers"
export SENTENCE_TRANSFORMERS_HOME="${CACHE_ROOT}/sentence_transformers"

source "${ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate "${ROOT}/miniconda3/envs/provetok"

# 开始 sweep
for tau in "${TAU_VALUES[@]}"; do
  # 生成目录名：tau 去掉小数点 (0.03 -> tau003, 0.045 -> tau0045)
  tau_str="tau$(echo "${tau}" | tr -d '.')"
  OUT_DIR="${OUT_ROOT}/${tau_str}"

  echo ""
  echo "═══════════════════════════════════════════════════════════════════"
  echo "  Starting Run: tau_iou=${tau}"
  echo "  Output: ${OUT_DIR}"
  echo "═══════════════════════════════════════════════════════════════════"

  rm -rf "${OUT_DIR}"
  mkdir -p "${OUT_DIR}"

  # Step 1: Probe checkpoint
  echo "[${tau_str}] Step 1: Probing checkpoint..."
  python "${ROOT}/Scripts/ckpt_probe.py" \
    --ckpt_path "${ROOT}/checkpoints/swinunetr.ckpt" \
    --in_channels 1 \
    --out_channels 2 \
    --feature_size 48 \
    --save_report "${OUT_DIR}/ckpt_probe_report.json"

  # Step 2: Run mini experiment with current tau_iou
  echo "[${tau_str}] Step 2: Running mini experiment..."
  python "${ROOT}/run_mini_experiment.py" \
    --ctrate_csv "${ROOT}/manifests/ctrate_manifest.csv" \
    --radgenome_csv "${ROOT}/manifests/radgenome_manifest.csv" \
    --out_dir "${OUT_DIR}" \
    --max_cases 50 \
    --expected_cases_per_dataset 50 \
    --cp_strict \
    --encoder_ckpt "${ROOT}/checkpoints/swinunetr.ckpt" \
    --text_encoder semantic \
    --text_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
    --text_encoder_device cuda \
    --device cuda \
    --token_budget_b 64 \
    --k_per_sentence 8 \
    --lambda_spatial 0.3 \
    --tau_iou "${tau}" \
    --beta 0.1 \
    --r2_mode ratio \
    --r2_min_support_ratio 0.8 \
    --r4_disabled \
    --r5_fallback_disabled \
    --anatomy_spatial_routing \
    --r2_skip_bilateral \
    --r1_negation_exempt \
    --r1_skip_midline \
    --r1_min_same_side_ratio 0.6 \
    --llm_judge huggingface \
    --llm_judge_model "${ROOT}/models/Llama-3.1-8B-Instruct" \
    --llm_judge_hf_torch_dtype bfloat16 \
    --llm_judge_alpha 0.5 2>&1 | tee "${OUT_DIR}/run.log"

  # Step 3: Validate outputs
  echo "[${tau_str}] Step 3: Validating outputs..."
  python "${ROOT}/validate_stage0_4_outputs.py" \
    --out_dir "${OUT_DIR}" \
    --datasets ctrate,radgenome \
    --expected_cases_map ctrate=50,radgenome=50 \
    --save_report "${OUT_DIR}/validation_report.json" 2>&1 | tee -a "${OUT_DIR}/run.log"

  echo "[${tau_str}] ✓ Complete"
done

# 运行完成后，生成汇总分析
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  All 5 sweep runs completed!"
echo "═══════════════════════════════════════════════════════════════════"
echo "  Output directory: ${OUT_ROOT}"
echo ""
echo "运行分析脚本查看结果："
echo "  python analyze_outputs.py \\"
echo "    --mode sweep \\"
echo "    --sweep_root ${OUT_ROOT} \\"
echo "    --sweep_glob 'tau*'"
echo ""
echo "查看 mediastinum 详细情况："
echo "  # 在每个输出目录的 analysis_exports/ 中查看："
echo "  #   - anatomy_r2_breakdown.csv      (R2_ANATOMY 按解剖结构分组)"
echo "  #   - anatomy_all_violation_rate.csv (所有解剖结构的违规率)"
echo "  #   - sentence_detail.csv           (句子级详情，可筛选 mediastinum)"
echo ""
