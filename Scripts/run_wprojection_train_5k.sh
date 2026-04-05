#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CASES_DIR="${PROJ_ROOT}/outputs/stage0_5_5k/train/cases"
TRAIN_MANIFEST="${PROJ_ROOT}/outputs/stage0_5_5k/train_manifest_wproj.txt"
VAL_MANIFEST="${PROJ_ROOT}/outputs/stage0_5_5k/valid_manifest_wproj.txt"
VAL_CASES_DIR="${PROJ_ROOT}/outputs/stage0_5_5k/valid/cases"
OUT_DIR="${PROJ_ROOT}/outputs/wprojection_5k"

CACHE_ROOT="${PROJ_ROOT}/.cache"
HF_HOME_DIR="${PROJ_ROOT}/.hf"

export XDG_CACHE_HOME="${CACHE_ROOT}"
export HF_HOME="${HF_HOME_DIR}"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/huggingface/transformers"
export SENTENCE_TRANSFORMERS_HOME="${CACHE_ROOT}/sentence_transformers"

mkdir -p "${OUT_DIR}"

N_TRAIN=$(wc -l < "${TRAIN_MANIFEST}")
N_VAL=$(wc -l < "${VAL_MANIFEST}")

echo "=========================================="
echo "W_proj Training (InfoNCE) — 5K dataset"
echo "Train: ${N_TRAIN} cases  |  Val: ${N_VAL} cases"
echo "OUT: ${OUT_DIR}"
echo "=========================================="

MERGED_CASES="${PROJ_ROOT}/outputs/stage0_5_5k/_merged_cases"
mkdir -p "${MERGED_CASES}/ctrate" "${MERGED_CASES}/radgenome"

for ds in ctrate radgenome; do
  for d in "${CASES_DIR}/${ds}"/*/; do
    name=$(basename "$d")
    ln -sfn "$d" "${MERGED_CASES}/${ds}/${name}" 2>/dev/null || true
  done
done

for ds in ctrate radgenome; do
  for d in "${VAL_CASES_DIR}/${ds}"/*/; do
    name=$(basename "$d")
    ln -sfn "$d" "${MERGED_CASES}/${ds}/${name}" 2>/dev/null || true
  done
done

echo "Merged cases dir: ${MERGED_CASES}"
echo "  ctrate: $(ls ${MERGED_CASES}/ctrate/ | wc -l) cases"
echo "  radgenome: $(ls ${MERGED_CASES}/radgenome/ | wc -l) cases"

python "${PROJ_ROOT}/train_wprojection.py" \
  --cases_dir     "${MERGED_CASES}" \
  --train_manifest "${TRAIN_MANIFEST}" \
  --val_manifest   "${VAL_MANIFEST}" \
  --out_dir       "${OUT_DIR}" \
  --text_encoder  semantic \
  --text_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
  --text_encoder_device cuda \
  --epochs        50 \
  --batch_size    64 \
  --lr            1e-3 \
  --tau           0.07 \
  --patience      10 \
  --device        cuda

echo ""
echo "=========================================="
echo "W_proj 训练完成"
echo "  w_proj.pt:      ${OUT_DIR}/w_proj.pt"
echo "  w_proj_best.pt: ${OUT_DIR}/w_proj_best.pt"
echo "  train_log.json: ${OUT_DIR}/train_log.json"
echo "=========================================="
