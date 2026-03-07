# 50-case 对比分析（2026-03-07）

## 1. 对比对象

- 基线运行（2026-03-06）  
  `outputs_stage0_4_follow_request_20260306_225010/r2_taut005_ratio_0.8_nor4r5`
- 本次运行（2026-03-07）  
  `outputs_stage0_4_follow_request_20260307_1344/r2_taut005_ratio_0.8_nor4r5`

两次均为 50/50（CT-RATE 50 + RadGenome 50），核心参数一致（`tau_iou=0.05`, `r2_mode=ratio`, `r2_min_support_ratio=0.8`, `r4_disabled`, `r5_fallback_disabled`, `cp_strict`），本次仅新增：

- `--anatomy_spatial_routing`

## 2. 结论（一句话）

本次结果相较 2026-03-06 基线**有下降**：总体句级违规率从 `45.75%` 降到 `42.00%`（`-3.75` pct），`R2` 与 `R1` 同步下降。

## 3. 核心指标对比

### 3.1 总体句级违规率

- 基线：`366 / 800 = 45.75%`
- 本次：`336 / 800 = 42.00%`
- 变化：`-30` 违规句（`-3.75` pct）

### 3.2 规则贡献（按总句数 800 归一）

- `R2_ANATOMY`：`269 -> 239`（`33.63% -> 29.88%`，`-3.75` pct）
- `R1_LATERALITY`：`119 -> 100`（`14.88% -> 12.50%`，`-2.38` pct）

说明：本批次违规几乎都来自 `R2` 和 `R1`，其他规则在当前配置下未触发显著贡献。

### 3.3 分数据集句级违规率

- `ctrate`：`48.5% -> 46.5%`（`-2.0` pct）
- `radgenome`：`43.0% -> 37.5%`（`-5.5` pct）

## 4. 错误分布变化（case 维度）

- `violation_ratio == 1.0`：两次均为 `0`
- `violation_ratio >= 0.75`：`9 -> 5`
- `violation_ratio` 中位数：`0.500 -> 0.375`
- `violation_ratio` Q3：`0.625 -> 0.500`

说明：高违规 case 占比进一步收缩，分布整体向低违规侧移动。

## 5. anatomy 维度观察（本次）

`anatomy_r2_breakdown.csv` 显示 R2 仍主要集中在：

- `bilateral`：197
- `mediastinum`：42

`anatomy_all_violation_rate.csv` 显示这两个关键词当前违规率仍为 `1.0`，是下一步最优先清理目标。

## 6. 产物路径

- 汇总：`outputs_stage0_4_follow_request_20260307_1344/r2_taut005_ratio_0.8_nor4r5/summary.csv`
- 验收：`outputs_stage0_4_follow_request_20260307_1344/r2_taut005_ratio_0.8_nor4r5/validation_report.json`
- 分析导出目录：  
  `outputs_stage0_4_follow_request_20260307_1344/r2_taut005_ratio_0.8_nor4r5/analysis_exports/`
  - `dataset_aggregate.csv`
  - `sentence_violation_rate.csv`
  - `rule_violation_count.csv`
  - `case_violation_ranked.csv`
  - `sentence_detail.csv`
  - `anatomy_r2_breakdown.csv`
  - `anatomy_all_violation_rate.csv`

## 7. 下一步建议

在保持 `anatomy_spatial_routing` 开启的前提下，优先针对 `bilateral` / `mediastinum` 做定向规则修正或路由补偿，再进行 50-case 快速回归验证，满足稳定下降后再推进 450/450 全量复核。
