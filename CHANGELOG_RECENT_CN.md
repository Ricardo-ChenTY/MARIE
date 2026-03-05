# Recent Changes (CN)

本文档汇总最近两次核心提交的改动，方便队友快速接手。

## 提交记录

1. `14ddc3e` - Add R2 sweep workflow and analysis notebook
2. `624dbd7` - Refresh Stage0-4 experiment results guide

## 改了什么

### 1) `run_mini_experiment.py`
- 新增 R2 策略参数：
  - `--r2_mode {auto,ratio,max_iou}`
  - `--r2_min_support_ratio`
- 增加参数合法性检查：
  - `r2_min_support_ratio` 必须在 `[0, 1]`
  - `cp_strict` 与 `r2_mode=max_iou` 不允许同时使用（避免口径冲突）

### 2) 新增 R2 sweep 批跑脚本
- 文件：`Scripts/run_r2_sweep_50_cp_strict_colab.sh`
- 用途：在 Colab 上按固定设置快速做 R2 参数扫描（50/50 smoke）

### 3) 新增 R2 sweep 汇总脚本
- 文件：`Scripts/summarize_r2_sweep.py`
- 用途：汇总多个 sweep 输出目录，生成可比较结果，便于选定后续默认配置

### 4) 新增 Colab 分析 notebook
- 文件：`Smoke_analysis/OUTPUT_ANALYSIS_COLAB.ipynb`
- 用途：读取 stage0-4 输出并做结构验收、规则分布、异常 case 分析

### 5) 更新运行说明
- 文件：`RUN_GUIDE_CN.md`
- 主要更新：新增 R2 sweep 快速流程、示例命令与汇总步骤

### 6) 覆盖实验结果说明
- 文件：`OUTPUT_ANALYSIS_GUIDE_CN.md`
- 主要更新：
  - 450/450 输出分析的阅读顺序
  - 指标解释（理想/不理想区间）
  - 下一步实验建议（先做 R2 sweep，再扩到 full run）

## 为什么这样改

- 目标是先把 Stage0-4 的可复现和可诊断能力稳定下来，避免直接被 LLM 生成噪声干扰。
- 先做 R2 sweep，可以快速定位“token-bank / router / verifier”哪一层导致违规率偏高。
- 把运行与分析说明独立成文档，便于 5090 本地跑和 Colab 分析协作分工。

## 你现在该怎么用

1. 生成结果（5090 本地）：按 `RUN_GUIDE_CN.md` 执行 smoke 或 450/450。
2. 结构验收：运行 `validate_stage0_4_outputs.py`，确认输出完整。
3. 分析：用 `Smoke_analysis/OUTPUT_ANALYSIS_COLAB.ipynb` 读取输出目录。
4. 参数扫描：需要调 R2 时，用 `Scripts/run_r2_sweep_50_cp_strict_colab.sh` + `Scripts/summarize_r2_sweep.py`。

## 关键文件清单

- `run_mini_experiment.py`
- `RUN_GUIDE_CN.md`
- `OUTPUT_ANALYSIS_GUIDE_CN.md`
- `Scripts/run_r2_sweep_50_cp_strict_colab.sh`
- `Scripts/summarize_r2_sweep.py`
- `Smoke_analysis/OUTPUT_ANALYSIS_COLAB.ipynb`
