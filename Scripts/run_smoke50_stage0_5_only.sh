#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/ProveTok_ACM"
OUT="${ROOT}/outputs/stage0_5_llama_50"
CACHE_ROOT="${ROOT}/.cache"
HF_HOME_DIR="${ROOT}/.hf"

mkdir -p \
  "${OUT}" \
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

rm -rf "${OUT}"
mkdir -p "${OUT}"

python "${ROOT}/Scripts/ckpt_probe.py" \
  --ckpt_path "${ROOT}/checkpoints/swinunetr.ckpt" \
  --in_channels 1 \
  --out_channels 2 \
  --feature_size 48 \
  --save_report "${OUT}/ckpt_probe_report.json"

python "${ROOT}/run_mini_experiment.py" \
  --ctrate_csv "${ROOT}/manifests/ctrate_manifest.csv" \
  --radgenome_csv "${ROOT}/manifests/radgenome_manifest.csv" \
  --out_dir "${OUT}" \
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
  --tau_iou 0.05 \
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
  --llm_judge_alpha 0.5 2>&1 | tee "${OUT}/run.log"

python "${ROOT}/validate_stage0_4_outputs.py" \
  --out_dir "${OUT}" \
  --datasets ctrate,radgenome \
  --expected_cases_map ctrate=50,radgenome=50 \
  --save_report "${OUT}/validation_report.json" 2>&1 | tee -a "${OUT}/run.log"
