# Recent Changes (CN)

本文档汇总最新一轮诊断与修正，方便协作者快速接手。

---

## 背景：450-case Smoke Run 诊断结论

首次 450/450 全量跑后违规率极高（ctrate 73%、radgenome 71%），
几乎每个 case 的全部句子都触发违规。诊断确认根因如下：

### 根因 1（最主要）：Hash Encoder 无语义

`DeterministicTextEncoder` 用 SHA256 哈希生成随机单位向量，与图像特征做
dot product 等同于随机 routing。Router 随机挑 token，Verifier 必然大量报 R1/R2。
这是 framework 验证阶段的 placeholder，**不能用于真实实验**。

> 修正：sweep 脚本显式传 `--text_encoder semantic`，并加注释说明原因。

### 根因 2：`r2_min_support_ratio=1.0` 过于严苛

要求 100% cited tokens 都满足 IoU ≥ tau。Router cite k=8 个 token，配合随机
routing，几乎必然全部触发 R2（第一大违规来源，2651 次）。

> 修正：sweep 新增 ratio 维度 {1.0, 0.8, 0.6}。

### 根因 3：R5 Negation Fallback 制造假阳性

所有 token 的 `negation_conflict` 硬编码为 0.0，永远走 fallback 分支。
"No effusion"、"No nodule" 等正常放射科表达因含 positive finding 词而触发 R5。

> 当前暂不修改，后续独立跑 R5 开关对比实验。

### 根因 4（结构性）：IoU 阈值与 token 粒度的匹配问题

token bbox 远小于解剖区域 bbox，IoU 天然偏低。**关键结论：此问题与分辨率无关**：

```
64^3：level-2 token (16^3=4096) vs 右上叶 (~58982)  → IoU ≈ 0.069
128^3：level-2 token (32^3=32768) vs 右上叶 (~471859) → IoU ≈ 0.069
```

分子分母同比例缩放，比值不变。因此 64^3 sweep 确定的 tau 区间可以**直接用于
128^3 主实验**，分辨率是特征质量问题，不是参数调优问题。

> 修正：sweep 新增 tau_iou 维度 {0.10, 0.05}。

---

## 本次修改清单

### 1) `Scripts/run_r2_sweep_50_cp_strict_colab.sh`（重写）

- **新增 tau_iou 维度**：{0.10, 0.05} × ratio {1.0, 0.8, 0.6} = 6 组
- **改为 64^3 resize**：sweep 阶段无需 128^3，速度快 4-8x，参数转移性有保证
- **明确 semantic encoder**：显式传 `--text_encoder semantic` 并加注释说明
- **输出目录命名更新**：`r2_tau{t}_ratio_{r}/`（如 `r2_taut010_ratio_0.8`）

### 2) `Scripts/summarize_r2_sweep.py`（小改）

- 默认 glob 从 `r2_ratio_*` 改为 `r2_tau*_ratio_*`
- 汇总表新增 `tau_iou` 列
- 排序键增加 `tau_iou`

### 3) `RUN_GUIDE_CN.md`（新增本地 5090 章节）

- 新增第 9 节：本地 RTX 5090 完整运行指南
- 包含环境准备、sweep 流程、450/450 全量跑命令

---

## 推荐实验顺序

```
第一步：50-case R2 sweep（64^3，6 组）
  bash Scripts/run_r2_sweep_50_cp_strict_colab.sh
  python Scripts/summarize_r2_sweep.py --sweep_root <dir> --save_csv sweep_summary.csv

第二步：确定 (tau, ratio) 最优区间
  观察：violation_sentence_rate 下降幅度、R2_ANATOMY 计数、R1/R5 是否稳定

第三步：450/450 全量（128^3，5090 本地）
  参考 RUN_GUIDE_CN.md 第 9 节

第四步（可选）：R5 fallback 开关对比
  关闭 r5_fallback_lexicon，独立量化 R5 贡献
```

---

## 关键文件清单

- `run_mini_experiment.py`
- `Scripts/run_r2_sweep_50_cp_strict_colab.sh`（本次修改）
- `Scripts/summarize_r2_sweep.py`（本次修改）
- `RUN_GUIDE_CN.md`（本次新增 5090 章节）
- `OUTPUT_ANALYSIS_GUIDE_CN.md`
- `Smoke_analysis/OUTPUT_ANALYSIS_COLAB.ipynb`
