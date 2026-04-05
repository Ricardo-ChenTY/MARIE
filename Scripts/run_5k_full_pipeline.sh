#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGFILE="${PROJ_ROOT}/outputs/5k_full_pipeline.log"
mkdir -p "$(dirname "${LOGFILE}")"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOGFILE}"
}

log "========== 5K Full Pipeline Start =========="
log "Project root: ${PROJ_ROOT}"

log ""
log "===== STEP 1/3: Running 5K Ablation Chain ====="
log ""

bash "${PROJ_ROOT}/Scripts/run_5k_ablation_chain.sh" 2>&1 | tee -a "${LOGFILE}"

log ""
log "===== STEP 1/3: Ablation Chain Done ====="

log ""
log "===== STEP 2/3: Generating 180-case tables/figures ====="
log ""

source "${PROJ_ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate MARIE

python "${PROJ_ROOT}/Scripts/generate_table2_and_figures.py" \
  2>&1 | tee -a "${LOGFILE}"

log ""
log "===== STEP 2/3: 180-case Done → outputs/paper_figures/ ====="

log ""
log "===== STEP 3/3: Generating 5K tables/figures ====="
log ""

python "${PROJ_ROOT}/Scripts/generate_table2_and_figures.py" \
  --data_dir outputs/5k_ablation \
  2>&1 | tee -a "${LOGFILE}"

log ""
log "===== STEP 3/3: 5K Done → outputs/paper_figures_5k/ ====="

log ""
log "=========================================="
log "5K Full Pipeline 完成!"
log ""
log "输出:"
log "  180-case: outputs/paper_figures/"
log "  5K:       outputs/paper_figures_5k/"
log ""
log "数据文件:"
log "  table2_data.json           (消融表)"
log "  table3_grounding_data.json (grounding 表)"
log "  table2_ablation.tex        (LaTeX)"
log "  fig1_waterfall.pdf"
log "  fig3_counterfactual.pdf"
log ""
log "日志: ${LOGFILE}"
log "=========================================="

