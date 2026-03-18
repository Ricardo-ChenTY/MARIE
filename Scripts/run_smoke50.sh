#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT="/data/ProveTok_ACM"
MODEL_DIR="${PROJ_ROOT}/models/Llama-3.1-8B-Instruct"
CACHE_ROOT="${PROJ_ROOT}/.cache"
HF_HOME_DIR="${PROJ_ROOT}/.hf"

mkdir -p "${CACHE_ROOT}/huggingface/hub" "${CACHE_ROOT}/huggingface/transformers" "${CACHE_ROOT}/sentence_transformers" "${HF_HOME_DIR}"

export XDG_CACHE_HOME="${CACHE_ROOT}"
export HF_HOME="${HF_HOME_DIR}"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/huggingface/transformers"
export SENTENCE_TRANSFORMERS_HOME="${CACHE_ROOT}/sentence_transformers"

cd "${PROJ_ROOT}"

python run_mini_experiment.py \
  --ctrate_csv manifests/ctrate_5k_test_manifest.csv \
  --radgenome_csv manifests/radgenome_5k_test_manifest.csv \
  --out_dir outputs/stage0_5_5k_smoke50 \
  --max_cases 25 \
  --cp_strict \
  --encoder_ckpt checkpoints/swinunetr.ckpt \
  --text_encoder semantic \
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
  --anatomy_spatial_routing \
  --r2_skip_bilateral \
  --r1_negation_exempt \
  --r1_skip_midline \
  --r1_min_same_side_ratio 0.6 \
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
  --reroute_max_retry 1
