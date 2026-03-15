#!/usr/bin/env bash
# ============================================================
# Phase B2: E1 routing + Stage 3c ON (token-gated LLM generation)
#
# Base: E1 (trained W_proj + spatial filter + semantic rerank, k=8)
# Change: Stage 3c enabled (Llama-3.1-8B-Instruct via HuggingFace)
# Stage 5: OFF
#
# Usage:
#   tmux new-session -s B2 'bash Scripts/run_ablation_B2.sh 2>&1 | tee logs/B2.log'
# ============================================================
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ─── Paths ────────────────────────────────────────
CTRATE_CSV="${PROJ_ROOT}/manifests/ctrate_test.csv"
RADGENOME_CSV="${PROJ_ROOT}/manifests/radgenome_test.csv"
ENCODER_CKPT="${PROJ_ROOT}/checkpoints/swinunetr.ckpt"
W_PROJ_PATH="${PROJ_ROOT}/outputs_wprojection/w_proj.pt"
LLAMA_MODEL="${PROJ_ROOT}/models/Llama-3.1-8B-Instruct"
OUT_DIR="${PROJ_ROOT}/outputs/ablation_routing_2x2/B2_E1_stage3c"
CACHE_ROOT="${PROJ_ROOT}/.cache"
HF_HOME_DIR="${PROJ_ROOT}/.hf"
# ──────────────────────────────────────────────────

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

# Activate conda environment
CONDA_BASE="${PROJ_ROOT}/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate provetok

echo "================================================================"
echo "Phase B2: E1 routing + Stage 3c ON"
echo "  E1: trained W_proj + spatial filter + semantic rerank, k=8"
echo "  Stage 3c: Llama-3.1-8B-Instruct (HuggingFace)"
echo "  Stage 5: OFF"
echo "  90 ctrate + 90 radgenome = 180 test cases"
echo "================================================================"

python "${PROJ_ROOT}/run_mini_experiment.py" \
  --ctrate_csv    "${CTRATE_CSV}" \
  --radgenome_csv "${RADGENOME_CSV}" \
  --max_cases 90 \
  --expected_cases_per_dataset 90 \
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
  --r2_skip_bilateral \
  --r1_negation_exempt \
  --r1_skip_midline \
  --r1_min_same_side_ratio 0.6 \
  --w_proj_path "${W_PROJ_PATH}" \
  --spatial_filter_semantic_rerank \
  --stage3c_backend huggingface \
  --stage3c_model "${LLAMA_MODEL}" \
  --stage3c_temperature 0.3 \
  --stage3c_max_tokens 256 \
  --out_dir "${OUT_DIR}" \
  2>&1 | tee "${OUT_DIR}/run.log"

echo ""
echo ">>> B2 done"
echo "Results: ${OUT_DIR}/summary.csv"
