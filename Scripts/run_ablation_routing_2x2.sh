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
echo "Phase A: Routing-only 2×2 Ablation (Stage 3c OFF, Stage 5 OFF)"
echo "  90 ctrate + 90 radgenome = 180 test cases"
echo "  Measuring Stage 4 violations only (gold text)"
echo "================================================================"

A0_DIR="${BASE_OUT}/A0_identity_spatial"
mkdir -p "${A0_DIR}"
echo ""
echo ">>> A0: identity W_proj + spatial routing"
echo "    OUT: ${A0_DIR}"
python "${PROJ_ROOT}/run_mini_experiment.py" \
  "${COMMON_FLAGS[@]}" \
  --out_dir "${A0_DIR}" \
  --anatomy_spatial_routing \
  2>&1 | tee "${A0_DIR}/run.log"
echo ">>> A0 done"

A1_DIR="${BASE_OUT}/A1_trained_spatial"
mkdir -p "${A1_DIR}"
echo ""
echo ">>> A1: trained W_proj + spatial routing"
echo "    OUT: ${A1_DIR}"
python "${PROJ_ROOT}/run_mini_experiment.py" \
  "${COMMON_FLAGS[@]}" \
  --out_dir "${A1_DIR}" \
  --anatomy_spatial_routing \
  --w_proj_path "${W_PROJ_PATH}" \
  2>&1 | tee "${A1_DIR}/run.log"
echo ">>> A1 done"

A2_DIR="${BASE_OUT}/A2_identity_semantic"
mkdir -p "${A2_DIR}"
echo ""
echo ">>> A2: identity W_proj + semantic routing"
echo "    OUT: ${A2_DIR}"
python "${PROJ_ROOT}/run_mini_experiment.py" \
  "${COMMON_FLAGS[@]}" \
  --out_dir "${A2_DIR}" \
  2>&1 | tee "${A2_DIR}/run.log"
echo ">>> A2 done"

A3_DIR="${BASE_OUT}/A3_trained_semantic"
mkdir -p "${A3_DIR}"
echo ""
echo ">>> A3: trained W_proj + semantic routing"
echo "    OUT: ${A3_DIR}"
python "${PROJ_ROOT}/run_mini_experiment.py" \
  "${COMMON_FLAGS[@]}" \
  --out_dir "${A3_DIR}" \
  --w_proj_path "${W_PROJ_PATH}" \
  2>&1 | tee "${A3_DIR}/run.log"
echo ">>> A3 done"

echo ""
echo "================================================================"
echo "All 4 conditions complete. Comparing results..."
echo "================================================================"

python3 -c "
import pandas as pd

configs = {
    'A0 (identity+spatial)': '${A0_DIR}/summary.csv',
    'A1 (trained+spatial)':  '${A1_DIR}/summary.csv',
    'A2 (identity+semantic)':'${A2_DIR}/summary.csv',
    'A3 (trained+semantic)': '${A3_DIR}/summary.csv',
}

rows = []
for label, path in configs.items():
    df = pd.read_csv(path)
    for ds in ['ctrate', 'radgenome']:
        sub = df[df.dataset == ds]
        ts = sub.n_sentences.sum()
        rows.append({
            'config': label,
            'dataset': ds,
            'cases': len(sub),
            'sentences': ts,
            'violations': sub.n_violations.sum(),
            'viol_rate': f'{sub.n_violations.sum()/ts:.4f}',
            'avg_viol': f'{sub.n_violations.mean():.3f}',
            'zero_viol': len(sub[sub.n_violations==0]),
        })

result = pd.DataFrame(rows)
print(result.to_string(index=False))
print()

for label, path in configs.items():
    df = pd.read_csv(path)
    v = df.n_violations.sum()
    s = df.n_sentences.sum()
    print(f'{label}: total_violations={v}, total_sentences={s}, rate={v/s:.4f}')
"

echo ""
echo "Results saved in: ${BASE_OUT}/"
echo "  A0_identity_spatial/summary.csv"
echo "  A1_trained_spatial/summary.csv"
echo "  A2_identity_semantic/summary.csv"
echo "  A3_trained_semantic/summary.csv"

