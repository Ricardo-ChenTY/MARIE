#!/usr/bin/env bash
set -euo pipefail

CTRATE_CSV="/data/MARIE/manifests/ctrate_manifest.csv"
RADGENOME_CSV="/data/MARIE/manifests/radgenome_manifest.csv"
ENCODER_CKPT="/data/MARIE/checkpoints/swinunetr.ckpt"

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${PROJ_ROOT}/outputs/stage0_5_llama_200"
MODEL_DIR="${PROJ_ROOT}/models/Llama-3.1-8B-Instruct"
CACHE_ROOT="${PROJ_ROOT}/.cache"
HF_HOME_DIR="${PROJ_ROOT}/.hf"

mkdir -p "${OUT_DIR}"
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

if [ ! -d "${MODEL_DIR}" ]; then
  echo "❌ 模型目录不存在: ${MODEL_DIR}"
  echo ""
  echo "请先下载模型:"
  echo "  cd ${PROJ_ROOT}"
  echo "  huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \\"
  echo "    --local-dir models/Llama-3.1-8B-Instruct"
  exit 1
fi

echo "=========================================="
echo "Stage 0-5  |  200/200  |  Llama-3.1-8B"
echo "tau_iou = 0.04  |  token_budget_b = 128  |  shuffle_seed = 99"
echo "模型: ${MODEL_DIR}"
echo "OUT:  ${OUT_DIR}"
echo "=========================================="

python "${PROJ_ROOT}/Scripts/ckpt_probe.py" \
  --ckpt_path "${ENCODER_CKPT}" \
  --in_channels 1 \
  --out_channels 2 \
  --feature_size 48 \
  --save_report "${OUT_DIR}/ckpt_probe_report.json"

python "${PROJ_ROOT}/run_mini_experiment.py" \
  --ctrate_csv    "${CTRATE_CSV}" \
  --radgenome_csv "${RADGENOME_CSV}" \
  --out_dir       "${OUT_DIR}" \
  --max_cases 200 \
  --expected_cases_per_dataset 200 \
  --shuffle_seed 99 \
  --cp_strict \
  --encoder_ckpt  "${ENCODER_CKPT}" \
  --text_encoder  semantic \
  --text_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
  --text_encoder_device cuda \
  --device cuda \
  --token_budget_b 128 \
  --k_per_sentence 8 \
  --lambda_spatial 0.3 \
  --tau_iou 0.04 \
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
  --llm_judge_model "${MODEL_DIR}" \
  --llm_judge_hf_torch_dtype bfloat16 \
  --llm_judge_alpha 0.5 2>&1 | tee "${OUT_DIR}/run.log"

python "${PROJ_ROOT}/validate_stage0_4_outputs.py" \
  --out_dir "${OUT_DIR}" \
  --datasets ctrate,radgenome \
  --expected_cases_map ctrate=200,radgenome=200 \
  --save_report "${OUT_DIR}/validation_report.json" 2>&1 | tee -a "${OUT_DIR}/run.log"

echo ""
echo "✅ Stage 0-5 (200/200) 完成: ${OUT_DIR}"
echo ""
echo "分析结果:"
echo "  python analyze_outputs.py \\"
echo "    --out_dir ${OUT_DIR} \\"
echo "    --expected_cases_map ctrate=200,radgenome=200"
echo ""

