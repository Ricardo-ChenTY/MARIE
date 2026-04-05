#!/usr/bin/env bash
set -euo pipefail

CTRATE_CSV="/data/MARIE/manifests/ctrate_manifest.csv"
RADGENOME_CSV="/data/MARIE/manifests/radgenome_manifest.csv"
ENCODER_CKPT="/data/MARIE/checkpoints/swinunetr.ckpt"

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${PROJ_ROOT}/outputs/stage0_4_450"
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

echo "=========================================="
echo "Stage 0-4 Baseline  |  450/450"
echo "OUT: ${OUT_DIR}"
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
  --max_cases 450 \
  --expected_cases_per_dataset 450 \
  --cp_strict \
  --encoder_ckpt  "${ENCODER_CKPT}" \
  --text_encoder  semantic \
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
  --r1_min_same_side_ratio 0.6

python "${PROJ_ROOT}/validate_stage0_4_outputs.py" \
  --out_dir "${OUT_DIR}" \
  --datasets ctrate,radgenome \
  --expected_cases_map ctrate=450,radgenome=450 \
  --save_report "${OUT_DIR}/validation_report.json"

echo ""
echo "✅ Stage 0-4 完成: ${OUT_DIR}"
echo "   summary.csv 和 validation_report.json 已生成"
echo ""
echo "下一步:"
echo "  Stage 0-5 (加 LLM 裁判): bash Scripts/run_stage0_5_llama_server.sh"
echo "  训练 W_proj:              bash Scripts/run_wprojection_train.sh"

