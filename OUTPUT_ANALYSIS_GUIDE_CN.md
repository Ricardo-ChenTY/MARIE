# Stage0-4 结果验收说明

更新日期：2026-03-07

本文档现在只服务于一个目标：验收 `450/450` 主实验结果。

## 1. 当前实验背景

旧基线的 `450/450` 结果较差：

- 总体 `violation_sentence_rate ≈ 0.721`
- `R2_ANATOMY = 2651`
- `R1_LATERALITY = 2009`

在加入以下配置后，`50/50` smoke 已明显改善：

- `--r2_mode ratio`
- `--r2_min_support_ratio 0.8`
- `--tau_iou 0.05`
- `--anatomy_spatial_routing`
- `--r2_skip_bilateral`
- `--r4_disabled`
- `--r5_fallback_disabled`

对应 smoke 结果：

- `Validated: 100, Passed: 100, Failed: 0`
- 总违规句率：`42.00% -> 17.375%`
- `R2_ANATOMY: 239 -> 42`
- `R1_LATERALITY: 100 -> 100`

这就是当前放大到 `450/450` 的依据。

## 2. 450/450 硬性验收标准

必须同时满足：

- `validation_report.json` 全通过
- `Validated cases = 900`
- `Failed = 0`
- `run_meta.json` 中 `cp_strict = true`
- `ctrate.selected_rows = 450`
- `radgenome.selected_rows = 450`
- `ctrate.processed_rows = 450`
- `radgenome.processed_rows = 450`
- `r2_skip_bilateral = true`
- `anatomy_spatial_routing = true`
- `r4_disabled = true`
- `r5_fallback_disabled = true`

如果上面有一条不满足，这轮结果就不应直接进报告。

## 3. 450/450 软性目标

这部分不是“一票否决”，但决定结果是否足够好：

- 总体 `violation_sentence_rate` 明显低于旧基线 `0.721`
- `R2_ANATOMY` 明显低于旧基线 `2651`
- `R1_LATERALITY` 不要出现明显反弹

建议用这个区间判断：

- `0.15 ~ 0.25`：很好，说明 smoke 收益基本稳定迁移到全量
- `0.25 ~ 0.35`：能接受，但要复查问题主要集中在哪条规则
- `> 0.35`：不建议直接作为主结果，需要继续诊断

## 4. 建议分析顺序

先看：

- `validation_report.json`
- `run_meta.json`
- `summary.csv`

再看 notebook 导出的聚合文件：

- `dataset_aggregate.csv`
- `sentence_violation_rate.csv`
- `rule_violation_count.csv`
- `abnormal_cases_ranked.csv`

最后再抽查：

- `cases/*/*/trace.jsonl`

## 5. Colab Notebook 用法

使用：

- [OUTPUT_ANALYSIS_COLAB.ipynb](c:\Users\34228\Desktop\ACM\Smoke_analysis\OUTPUT_ANALYSIS_COLAB.ipynb)

现在这个 notebook 默认就是 `450/450` 验收口径：

- 默认 `OUT_DIR` 指向 `outputs_stage0_4_450_128`
- 结构验收强制 `expected_cases_map = ctrate=450,radgenome=450`
- 会打印主实验状态检查
- 会导出 `analysis_exports/*`

## 6. 最小交付物

你朋友跑完后，至少要给你：

- `summary.csv`
- `ctrate_case_summary.csv`
- `radgenome_case_summary.csv`
- `run_meta.json`
- `validation_report.json`
- `cases/*/*/trace.jsonl`

如果只做结果分析，`cache/` 可以不传。

## 7. 当前结论

当前阶段不再需要继续做新的 `50/50` smoke 才能前进。

下一步就是：

1. 用当前锁定配置直接跑 `450/450`
2. 用 notebook 做结构验收和聚合分析
3. 对照旧基线判断这轮是否已经足够进入报告
