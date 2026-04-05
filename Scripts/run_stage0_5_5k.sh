#!/usr/bin/env bash
set -euo pipefail

SPLIT="${1:-all}"  # train / valid / test / all

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANI_DIR="${PROJ_ROOT}/manifests"
ENCODER_CKPT="${PROJ_ROOT}/checkpoints/swinunetr.ckpt"
MODEL_DIR="${PROJ_ROOT}/models/Llama-3.1-8B-Instruct"
CACHE_ROOT="${PROJ_ROOT}/.cache"
HF_HOME_DIR="${PROJ_ROOT}/.hf"

declare -A CASES_CT=( [train]=2000 [valid]=250 [test]=250 )
declare -A CASES_RG=( [train]=2000 [valid]=250 [test]=250 )

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
  echo "模型目录不存在: ${MODEL_DIR}"
  echo "请先下载: huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir models/Llama-3.1-8B-Instruct"
  exit 1
fi

if [ "${SPLIT}" = "all" ]; then
  SPLITS=("train" "valid" "test")
else
  SPLITS=("${SPLIT}")
fi

for S in "${SPLITS[@]}"; do
  CT_CSV="${MANI_DIR}/ctrate_5k_${S}_manifest.csv"
  RG_CSV="${MANI_DIR}/radgenome_5k_${S}_manifest.csv"
  OUT_DIR="${PROJ_ROOT}/outputs/stage0_5_5k/${S}"
  N_CT="${CASES_CT[$S]}"
  N_RG="${CASES_RG[$S]}"

  if [ ! -f "${CT_CSV}" ] || [ ! -f "${RG_CSV}" ]; then
    echo "SKIP ${S}: manifest not found (${CT_CSV} or ${RG_CSV})"
    continue
  fi

  mkdir -p "${OUT_DIR}"

  echo ""
  echo "=========================================="
  echo "Split: ${S}  |  CT-RATE: ${N_CT}  |  RadGenome: ${N_RG}"
  echo "Config: B2'v2 (Evidence Card v2 + Log-smooth Reroute)"
  echo "OUT: ${OUT_DIR}"
  echo "=========================================="

  if [ ! -f "${OUT_DIR}/ckpt_probe_report.json" ]; then
    python "${PROJ_ROOT}/Scripts/ckpt_probe.py" \
      --ckpt_path "${ENCODER_CKPT}" \
      --in_channels 1 \
      --out_channels 2 \
      --feature_size 48 \
      --save_report "${OUT_DIR}/ckpt_probe_report.json"
  fi

  python "${PROJ_ROOT}/run_mini_experiment.py" \
    --ctrate_csv    "${CT_CSV}" \
    --radgenome_csv "${RG_CSV}" \
    --out_dir       "${OUT_DIR}" \
    --max_cases "${N_CT}" \
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
    --reroute_max_retry 1 2>&1 | tee "${OUT_DIR}/run.log"

  python "${PROJ_ROOT}/validate_stage0_4_outputs.py" \
    --out_dir "${OUT_DIR}" \
    --datasets ctrate,radgenome \
    --save_report "${OUT_DIR}/validation_report.json"

  echo ""
  echo "Split ${S} 完成: ${OUT_DIR}"
done

echo ""
echo "=========================================="
echo "全部完成"
echo "=========================================="
echo "输出目录: ${PROJ_ROOT}/outputs/stage0_5_5k/{train,valid,test}/"

