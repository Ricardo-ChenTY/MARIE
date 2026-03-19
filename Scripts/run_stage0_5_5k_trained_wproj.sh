#!/usr/bin/env bash
# ============================================================
# Stage 0-5 with Trained W_proj — 5k test split evaluation
#   Uses W_proj trained on 4000 train cases
#   Routing: spatial_filter_semantic_rerank (E1)
#
# 用法:
#   bash Scripts/run_stage0_5_5k_trained_wproj.sh         # 跑 test (default)
#   bash Scripts/run_stage0_5_5k_trained_wproj.sh valid    # 跑 valid
# ============================================================
set -euo pipefail

SPLIT="${1:-test}"

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANI_DIR="${PROJ_ROOT}/manifests"
ENCODER_CKPT="${PROJ_ROOT}/checkpoints/swinunetr.ckpt"
MODEL_DIR="${PROJ_ROOT}/models/Llama-3.1-8B-Instruct"
W_PROJ="${PROJ_ROOT}/outputs/wprojection_5k/w_proj_best.pt"
CACHE_ROOT="${PROJ_ROOT}/.cache"
HF_HOME_DIR="${PROJ_ROOT}/.hf"

CT_CSV="${MANI_DIR}/ctrate_5k_${SPLIT}_manifest.csv"
RG_CSV="${MANI_DIR}/radgenome_5k_${SPLIT}_manifest.csv"
OUT_DIR="${PROJ_ROOT}/outputs/stage0_5_5k_trained_wproj/${SPLIT}"

mkdir -p "${CACHE_ROOT}/huggingface/hub" "${CACHE_ROOT}/huggingface/transformers" "${CACHE_ROOT}/sentence_transformers" "${HF_HOME_DIR}"

export XDG_CACHE_HOME="${CACHE_ROOT}"
export HF_HOME="${HF_HOME_DIR}"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/huggingface/transformers"
export SENTENCE_TRANSFORMERS_HOME="${CACHE_ROOT}/sentence_transformers"

if [ ! -f "${W_PROJ}" ]; then
  echo "W_proj not found: ${W_PROJ}"
  echo "Please run Scripts/run_wprojection_train_5k.sh first"
  exit 1
fi

echo "=========================================="
echo "Split: ${SPLIT}  |  Trained W_proj + E1 Routing"
echo "W_proj: ${W_PROJ}"
echo "OUT: ${OUT_DIR}"
echo "=========================================="

mkdir -p "${OUT_DIR}"

python "${PROJ_ROOT}/run_mini_experiment.py" \
  --ctrate_csv    "${CT_CSV}" \
  --radgenome_csv "${RG_CSV}" \
  --out_dir       "${OUT_DIR}" \
  --max_cases 250 \
  --expected_cases_per_dataset 0 \
  --cp_strict \
  --encoder_ckpt  "${ENCODER_CKPT}" \
  --text_encoder  semantic \
  --text_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
  --text_encoder_device cuda \
  --device cuda \
  --shuffle_seed 42 \
  --token_budget_b 128 \
  --k_per_sentence 8 \
  --lambda_spatial 0.3 \
  --tau_iou 0.04 \
  --beta 0.1 \
  --r2_mode ratio \
  --r2_min_support_ratio 0.8 \
  --r4_disabled \
  --r5_fallback_disabled \
  --r2_skip_bilateral \
  --r1_negation_exempt \
  --r1_skip_midline \
  --r1_min_same_side_ratio 0.6 \
  --w_proj_path "${W_PROJ}" \
  --spatial_filter_semantic_rerank \
  --llm_judge huggingface \
  --llm_judge_model "${MODEL_DIR}" \
  --llm_judge_hf_torch_dtype bfloat16 \
  --llm_judge_alpha 0.5 \
  --stage3c_backend huggingface \
  --stage3c_model "${MODEL_DIR}" \
  --stage3c_temperature 0.3 \
  --stage3c_max_tokens 256 \
  --strict_laterality \
  --reroute_gamma 2.0 \
  --reroute_max_retry 1 2>&1 | tee "${OUT_DIR}/run.log"

echo ""
echo "Split ${SPLIT} 完成: ${OUT_DIR}"
