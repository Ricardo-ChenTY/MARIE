# ProveTok_ACM 运行说明

当前主流程：`Stage 0-4`，不调用 LLM 生成。

## 1. 当前锁定配置

这轮准备直接跑 `450/450` 主实验，推荐固定以下开关：

- `--cp_strict`
- `--r2_mode ratio`
- `--r2_min_support_ratio 0.8`
- `--tau_iou 0.05`
- `--anatomy_spatial_routing`
- `--r2_skip_bilateral`
- `--r4_disabled`
- `--r5_fallback_disabled`

对应的 `50/50` smoke 已通过：

- `Validated: 100, Passed: 100, Failed: 0`
- 总违规句率：`42.00% -> 17.375%`
- `R2_ANATOMY: 239 -> 42`
- `R1_LATERALITY: 100 -> 100`

结论：这套配置已经够资格放大到 `450/450`。

## 2. 必需输入

需要两个 manifest CSV：

- `ctrate_manifest.csv`
- `radgenome_manifest.csv`

每个 CSV 至少包含：

- `case_id`
- `volume_path`
- `report_text`

还需要一个可加载的 SwinUNETR checkpoint：

- `SwinUNETR(in_channels=1, out_channels=2, feature_size=48)`

## 3. 环境准备

```bash
nvidia-smi

conda env create -f environment.yaml
conda activate provetok
pip install sentence-transformers
```

## 4. 450/450 主实验命令

### 4.1 Linux / Colab Shell

```bash
CTRATE_CSV="/path/to/ctrate_manifest.csv"
RADGENOME_CSV="/path/to/radgenome_manifest.csv"
ENCODER_CKPT="/path/to/swinunetr.ckpt"
OUT_ROOT="/path/to/outputs"

python run_mini_experiment.py \
  --ctrate_csv "${CTRATE_CSV}" \
  --radgenome_csv "${RADGENOME_CSV}" \
  --out_dir "${OUT_ROOT}/outputs_stage0_4_450_128" \
  --max_cases 450 \
  --expected_cases_per_dataset 450 \
  --cp_strict \
  --encoder_ckpt "${ENCODER_CKPT}" \
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
  --r2_skip_bilateral
```

### 4.2 Windows PowerShell

```powershell
$CTRATE_CSV    = "C:\path\to\ctrate_manifest.csv"
$RADGENOME_CSV = "C:\path\to\radgenome_manifest.csv"
$ENCODER_CKPT  = "C:\path\to\swinunetr.ckpt"
$OUT_ROOT      = "C:\path\to\outputs"

python run_mini_experiment.py `
  --ctrate_csv $CTRATE_CSV `
  --radgenome_csv $RADGENOME_CSV `
  --out_dir "$OUT_ROOT\outputs_stage0_4_450_128" `
  --max_cases 450 `
  --expected_cases_per_dataset 450 `
  --cp_strict `
  --encoder_ckpt $ENCODER_CKPT `
  --text_encoder semantic `
  --text_encoder_model sentence-transformers/all-MiniLM-L6-v2 `
  --text_encoder_device cuda `
  --device cuda `
  --token_budget_b 64 `
  --k_per_sentence 8 `
  --lambda_spatial 0.3 `
  --tau_iou 0.05 `
  --beta 0.1 `
  --r2_mode ratio `
  --r2_min_support_ratio 0.8 `
  --r4_disabled `
  --r5_fallback_disabled `
  --anatomy_spatial_routing `
  --r2_skip_bilateral
```

## 5. 结构验收

```bash
python validate_stage0_4_outputs.py \
  --out_dir "${OUT_ROOT}/outputs_stage0_4_450_128" \
  --datasets ctrate,radgenome \
  --expected_cases_map ctrate=450,radgenome=450 \
  --save_report "${OUT_ROOT}/outputs_stage0_4_450_128/validation_report.json"
```

通过标准：

- `Validated cases: 900`
- `Failed: 0`
- `run_meta.json` 中 `cp_strict = true`
- `run_meta.json` 中 `ctrate.selected_rows = 450`
- `run_meta.json` 中 `radgenome.selected_rows = 450`

## 6. Colab 验收 Notebook

使用：

- [OUTPUT_ANALYSIS_COLAB.ipynb](c:\Users\34228\Desktop\ACM\Smoke_analysis\OUTPUT_ANALYSIS_COLAB.ipynb)

你朋友在 Colab 里只需要：

1. 挂载 Google Drive
2. 打开 notebook
3. 把 `OUT_DIR` 改成 `450/450` 输出目录
4. 运行全部 cells

notebook 现在默认按 `450/450` 口径验收，会检查：

- `expected_cases_map = ctrate=450,radgenome=450`
- `validation_report.json` 是否全过
- `run_meta.json` 的关键开关是否符合主实验配置
- `summary.csv`、`cases/*/*/trace.jsonl` 是否能正常汇总

notebook 会导出：

- `analysis_exports/dataset_aggregate.csv`
- `analysis_exports/sentence_violation_rate.csv`
- `analysis_exports/rule_violation_count.csv`
- `analysis_exports/abnormal_cases_ranked.csv`
- `analysis_exports/sentence_detail.csv`

## 7. 结果怎么看

硬性要求：

- 结构验收通过
- 样本数是 `450/450`
- `R1` 不出现明显反弹
- `R2` 相比旧基线 `2651` 明显下降

软性目标：

- 总体 `violation_sentence_rate` 远低于旧基线 `0.721`
- 如果最终落在 `0.15 ~ 0.25`，说明 `50/50` smoke 的收益基本稳定迁移到了全量
- 如果高于 `0.35`，要回头检查数据分布漂移或参数/规则联动问题

## 8. 输出目录最小交付物

重跑结束后，至少保留：

- `summary.csv`
- `ctrate_case_summary.csv`
- `radgenome_case_summary.csv`
- `run_meta.json`
- `validation_report.json`
- `cases/*/*/trace.jsonl`

如果只做分析，`cache/` 不是必需。

## 9. 常见问题

- `expected_cases_map` 不通过：先检查 manifest 是否真的各有 `450` 行。
- `sentence-transformers` 下载慢：可提前缓存模型，或把 `--text_encoder_model` 指向本地目录。
- 显存不足：可临时改成 `64^3` 验证，但主实验仍建议保留 `128^3`。
- 只要做结果分析：直接用 notebook，不需要重跑主实验。
