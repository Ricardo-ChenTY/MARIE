#!/usr/bin/env bash
# ============================================================
# 5K Ablation Chain: A0 → A1 → E1 → B2'(v1) → B2'v2 → C2' → D2
# Runs all 7 configurations on 5K test split (250 CT-RATE + 250 RadGenome)
#
# 用法:
#   bash Scripts/run_5k_ablation_chain.sh                    # 跑全部 7 个配置
#   bash Scripts/run_5k_ablation_chain.sh A0                # 只跑 A0
#   bash Scripts/run_5k_ablation_chain.sh B2_prime_v2       # 只跑 B2'v2
#   bash Scripts/run_5k_ablation_chain.sh A0 A1 E1          # 跑 routing-only (快, ~1-2h)
#
# 预计时间:
#   A0, A1, E1: 各 ~1-2h (无 LLM)
#   B2'v1, B2'v2, C2', D2: 各 ~3-5h (有 LLM)
#   总计: ~15-23h
# ============================================================
set -euo pipefail

# ─── 路径 ────────────────────────────────────────
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANI_DIR="${PROJ_ROOT}/manifests"
ENCODER_CKPT="${PROJ_ROOT}/checkpoints/swinunetr.ckpt"
MODEL_DIR="${PROJ_ROOT}/models/Llama-3.1-8B-Instruct"
W_PROJ="${PROJ_ROOT}/outputs/wprojection_5k/w_proj_best.pt"
CACHE_ROOT="${PROJ_ROOT}/.cache"
HF_HOME_DIR="${PROJ_ROOT}/.hf"

CT_CSV="${MANI_DIR}/ctrate_5k_test_manifest.csv"
RG_CSV="${MANI_DIR}/radgenome_5k_test_manifest.csv"
OUT_BASE="${PROJ_ROOT}/outputs/5k_ablation"

N_CASES=250  # per dataset

# ─── 环境 ────────────────────────────────────────
source "${PROJ_ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate provetok

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

# ─── 公共参数 ────────────────────────────────────
COMMON_ARGS=(
  --ctrate_csv    "${CT_CSV}"
  --radgenome_csv "${RG_CSV}"
  --max_cases "${N_CASES}"
  --expected_cases_per_dataset 0
  --cp_strict
  --encoder_ckpt  "${ENCODER_CKPT}"
  --text_encoder  semantic
  --text_encoder_model sentence-transformers/all-MiniLM-L6-v2
  --text_encoder_device cuda
  --device cuda
  --shuffle_seed 42
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

# LLM 相关的公共参数 (B2+)
LLM_ARGS=(
  --stage3c_backend huggingface
  --stage3c_model "${MODEL_DIR}"
  --stage3c_temperature 0.3
  --stage3c_max_tokens 256
)

JUDGE_ARGS=(
  --llm_judge huggingface
  --llm_judge_model "${MODEL_DIR}"
  --llm_judge_hf_torch_dtype bfloat16
  --llm_judge_alpha 0.5
)

REROUTE_ARGS=(
  --reroute_gamma 2.0
  --reroute_max_retry 1
)

# ─── 配置定义 ────────────────────────────────────
run_config() {
  local CONFIG_ID="$1"
  local OUT_DIR="${OUT_BASE}/${CONFIG_ID}"
  shift
  local EXTRA_ARGS=("$@")

  if [ -f "${OUT_DIR}/summary.csv" ]; then
    echo ""
    echo "[SKIP] ${CONFIG_ID}: summary.csv already exists at ${OUT_DIR}"
    echo "       Delete the directory to re-run."
    return 0
  fi

  mkdir -p "${OUT_DIR}"

  echo ""
  echo "=========================================="
  echo "Running: ${CONFIG_ID}"
  echo "OUT: ${OUT_DIR}"
  echo "=========================================="

  python "${PROJ_ROOT}/run_mini_experiment.py" \
    "${COMMON_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    --out_dir "${OUT_DIR}" \
    2>&1 | tee "${OUT_DIR}/run.log"

  echo "[DONE] ${CONFIG_ID} → ${OUT_DIR}"
}

# ─── 7 个配置 ────────────────────────────────────

run_A0() {
  # A0: Identity W_proj + Anatomy-primary spatial routing
  # 无 LLM, 无 W_proj, 纯空间路由
  run_config "A0_identity_spatial" \
    --anatomy_spatial_routing
}

run_A1() {
  # A1: Trained W_proj + Anatomy-primary spatial routing
  # 有 W_proj 但路由仍以 IoU 为主
  run_config "A1_trained_spatial" \
    --anatomy_spatial_routing \
    --w_proj_path "${W_PROJ}"
}

run_E1() {
  # E1: Spatial filter + Semantic rerank (trained W_proj)
  # IoU 过滤 + W_proj cosine 重排
  run_config "E1_spatial_filter_rerank" \
    --spatial_filter_semantic_rerank \
    --w_proj_path "${W_PROJ}"
}

run_B2_prime() {
  # B2'(v1): E1 + LLM gen + Evidence Card v1
  # Evidence card 默认生成, v1 = 无 --strict_laterality
  run_config "B2_evcard_v1" \
    --spatial_filter_semantic_rerank \
    --w_proj_path "${W_PROJ}" \
    "${LLM_ARGS[@]}"
}

run_B2_prime_v2() {
  # B2'v2: E1 + LLM gen + Evidence Card v2 (strict laterality)
  # 这是最终主模型配置
  run_config "B2_evcard_v2" \
    --spatial_filter_semantic_rerank \
    --w_proj_path "${W_PROJ}" \
    "${LLM_ARGS[@]}" \
    --strict_laterality
}

run_C2_prime() {
  # C2': B2'(v1) + LLM Judge (Stage 5)
  # 注意: 基于 v1, 不是 v2 (与 180-case 消融一致)
  run_config "C2_evcard_v1_judge" \
    --spatial_filter_semantic_rerank \
    --w_proj_path "${W_PROJ}" \
    "${LLM_ARGS[@]}" \
    "${JUDGE_ARGS[@]}"
}

run_D2() {
  # D2: C2' + Repair executor (log-smooth reroute)
  run_config "D2_repair" \
    --spatial_filter_semantic_rerank \
    --w_proj_path "${W_PROJ}" \
    "${LLM_ARGS[@]}" \
    "${JUDGE_ARGS[@]}" \
    "${REROUTE_ARGS[@]}"
}

# ─── 执行 ────────────────────────────────────────

ALL_CONFIGS="A0 A1 E1 B2_prime B2_prime_v2 C2_prime D2"

if [ $# -eq 0 ]; then
  TARGETS="${ALL_CONFIGS}"
else
  TARGETS="$*"
fi

for TARGET in ${TARGETS}; do
  case "${TARGET}" in
    A0)       run_A0 ;;
    A1)       run_A1 ;;
    E1)       run_E1 ;;
    B2_prime|B2p|"B2'")       run_B2_prime ;;
    B2_prime_v2|B2pv2|"B2'v2") run_B2_prime_v2 ;;
    C2_prime|C2p|"C2'")       run_C2_prime ;;
    D2)       run_D2 ;;
    *)
      echo "Unknown config: ${TARGET}"
      echo "Valid: A0 A1 E1 B2_prime B2_prime_v2 C2_prime D2"
      exit 1
      ;;
  esac
done

echo ""
echo "=========================================="
echo "5K Ablation Chain 完成"
echo "输出: ${OUT_BASE}/"
echo "=========================================="
echo ""
echo "下一步: 运行分析脚本生成 Table 2 和 Figure 1"
echo "  python Scripts/generate_table2_and_figures.py"
