#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

CTRATE_CSV="${PROJ_ROOT}/manifests/ctrate_test.csv"
RADGENOME_CSV="${PROJ_ROOT}/manifests/radgenome_test.csv"
ENCODER_CKPT="${PROJ_ROOT}/checkpoints/swinunetr.ckpt"
W_PROJ_PATH="${PROJ_ROOT}/outputs_wprojection/w_proj.pt"
BASE_OUT="${PROJ_ROOT}/outputs/ablation_routing_2x2"
CACHE_ROOT="${PROJ_ROOT}/.cache"
HF_HOME_DIR="${PROJ_ROOT}/.hf"

mkdir -p "${BASE_OUT}"
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

CONDA_BASE="${PROJ_ROOT}/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate MARIE

COMMON_FLAGS=(
  --ctrate_csv    "${CTRATE_CSV}"
  --radgenome_csv "${RADGENOME_CSV}"
  --max_cases 90
  --expected_cases_per_dataset 90
  --cp_strict
  --encoder_ckpt  "${ENCODER_CKPT}"
  --text_encoder  semantic
  --text_encoder_model sentence-transformers/all-MiniLM-L6-v2
  --text_encoder_device cuda
  --device cuda
  --token_budget_b 128
  --k_per_sentence 8
  --lambda_spatial 0.3
  --tau_iou 0.04
  --beta 0.1
  --r2_mode ratio
  --r2_min_support_ratio 0.8
  --r4_disabled
  --r5_fallback_disabled
  --r2_skip_bilateral
  --r1_negation_exempt
  --r1_skip_midline
  --r1_min_same_side_ratio 0.6
)

echo "================================================================"
echo "Phase E1: Spatial Filter + Semantic Rerank (trained W_proj)"
echo "  90 ctrate + 90 radgenome = 180 test cases"
echo "  Stage 3c OFF, Stage 5 OFF (gold text)"
echo "================================================================"

E1_DIR="${BASE_OUT}/E1_trained_filter_rerank"
mkdir -p "${E1_DIR}"
echo ""
echo ">>> E1: trained W_proj + spatial filter + semantic rerank"
echo "    OUT: ${E1_DIR}"
python "${PROJ_ROOT}/run_mini_experiment.py" \
  "${COMMON_FLAGS[@]}" \
  --out_dir "${E1_DIR}" \
  --w_proj_path "${W_PROJ_PATH}" \
  --spatial_filter_semantic_rerank \
  2>&1 | tee "${E1_DIR}/run.log"
echo ">>> E1 done"

echo ""
echo "Results saved in: ${E1_DIR}/summary.csv"

