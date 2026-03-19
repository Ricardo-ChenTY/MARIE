# ProveTok 5K-Scale Experiment Report

> **Config**: B2'v2 (Evidence Card v2 + Log-smooth Reroute)
> **Date**: 2026-03-18
> **Dataset**: 5000 cases (2500 CT-RATE + 2500 RadGenome-ChestCT)
> **Hardware**: Azure VM, 1× GPU, Llama-3.1-8B-Instruct (bfloat16)

---

## 1. Dataset

### 1.1 数据来源

| Dataset | Source | Total Downloaded | Excluded (overlap w/ 900) | Used |
|---------|--------|-----------------|--------------------------|------|
| CT-RATE | `ibrahimhamamci/CT-RATE` (HuggingFace) | 2500 | 900 (previous training set) | 2500 |
| RadGenome-ChestCT | `RadGenome/RadGenome-ChestCT` (HuggingFace) | 2500 | 450 (previous training set) | 2500 |

### 1.2 Train / Valid / Test Split

| Split | CT-RATE | RadGenome | Total | 用途 |
|-------|---------|-----------|-------|------|
| train | 2000 | 2000 | 4000 | W_proj 训练 |
| valid | 250 | 250 | 500 | 验证 / Early stopping |
| test | 250 | 250 | 500 | 最终评估 |
| **Total** | **2500** | **2500** | **5000** | |

### 1.3 存储

| Item | Size |
|------|------|
| CT-RATE raw volumes (.nii.gz) | 386 GB |
| RadGenome raw volumes (.nii.gz) | 109 GB |
| Dataset total | 495 GB |
| Pipeline outputs (5k) | 40 GB |
| 存储位置 | `/mnt/dataset/` (1TB Azure 挂载盘) |

---

## 2. Pipeline 配置 (B2'v2 Best Config)

从 900-case 路由消融实验中选出的最优配置：

| Parameter | Value |
|-----------|-------|
| Routing | `anatomy_spatial_routing` (IoU-based) |
| W_proj | Identity (untrained) |
| Text Encoder | `sentence-transformers/all-MiniLM-L6-v2` (semantic) |
| Token Budget | 128 |
| k_per_sentence | 8 |
| tau_iou | 0.04 |
| lambda_spatial | 0.3 |
| Stage 3c | LLM generation (Llama-3.1-8B, temp=0.3) |
| Evidence Card | **v2** (strict laterality gate, depth gate, same-side cleanup) |
| Stage 5 | LLM judge (Llama-3.1-8B, alpha=0.5) |
| Reroute | Log-smooth (gamma=2.0, max_retry=1) |
| R1 | skip_midline, negation_exempt, min_same_side_ratio=0.6 |
| R2 | ratio mode, min_support_ratio=0.8, skip_bilateral |
| R4, R5 | disabled |

---

## 3. Evaluation Methodology

本实验采用**两层评估体系**，分别衡量 3D 空间引用质量和文本生成质量：

### Layer 1: Spatial Consistency (违规率)

ProveTok pipeline 将报告的每句话**路由**到 CT volume 中的 k 个 3D token（octree 节点），形成 sentence→token citation。**Verifier (Stage 4)** 自动检查每条 citation 是否与报告文本在空间上一致：

| Rule | 检查内容 | 示例 |
|------|---------|------|
| R1_LATERALITY | 报告声明的侧别（left/right）与 cited token 的空间位置是否一致 | 报告说 "left pleural effusion"，cited token 是否在左侧？ |
| R2_ANATOMY | Cited token 的 bounding box 与目标解剖区域的 IoU 是否足够 | "liver" 对应的 token 是否覆盖肝脏区域？ |
| R3_DEPTH | Cited token 在 octree 中的深度层级是否与描述的细粒度匹配 | 描述细节特征时，token 应在较深层级 |
| R6a_CROSS_LAT | 跨句引用同一 token 时，侧别声明是否冲突 | 句 A 说 "left"，句 B 引用同一 token 说 "right"？ |
| R6b_CROSS_PRES | 跨句引用同一 token 时，存在性声明是否冲突 | 句 A 说 "有积液"，句 B 引用同一 token 说 "无积液"？ |

**违规率 = 违规句数 / 总句数**。违规率越低，说明 token citation 的空间准确性越高。

### Layer 2: NLG Metrics (文本生成质量)

Stage 3c 使用 LLM (Llama-3.1-8B) 基于 token 的 Evidence Card 生成报告句子。NLG 指标衡量**生成句子与原始报告句子的文本相似度**：

| Metric | 说明 |
|--------|------|
| BLEU-4 | 4-gram 精度，衡量生成文本与参考文本的 n-gram 重叠 |
| ROUGE-L | 最长公共子序列，衡量生成文本与参考文本的召回 |
| METEOR | 综合考虑同义词和词形变化的匹配 |

**注意**：这两层指标衡量不同维度。违规率低说明 3D 引用准确（"指对了地方"），NLG 高说明生成文本忠实（"说对了话"）。理想的 pipeline 应同时优化两者。

---

## 4. Results Overview

### 4.1 Per-Split Summary

| Split | Cases | Sentences | Violations | Viol. Rate | BLEU-4 | ROUGE-L | METEOR |
|-------|-------|-----------|------------|------------|--------|---------|--------|
| train | 4000 | 31,919 | 1,233 | 3.86% | 0.492 | 0.656 | 0.636 |
| valid | 500 | 3,995 | 159 | 3.98% | 0.502 | 0.661 | 0.640 |
| test | 500 | 3,979 | 161 | 4.05% | 0.487 | 0.653 | 0.632 |
| **Total** | **5000** | **39,893** | **1,553** | **3.89%** | **0.492** | **0.656** | **0.636** |

**Key observation**: 三个 split 的违规率非常一致 (3.86% ~ 4.05%)，NLG 指标也高度稳定，说明 pipeline 泛化性良好。

### 4.2 Per-Rule Violation Breakdown

| Rule | Train | Valid | Test | Total | Description |
|------|-------|-------|------|-------|-------------|
| R1_LATERALITY | 39 | 8 | 2 | **49** | 侧别不一致 |
| R2_ANATOMY | 0 | 0 | 0 | **0** | 解剖 IoU 不足 |
| R3_DEPTH | 864 | 103 | 113 | **1,080** | 深度层级错误 |
| R4_SIZE | 0 | 0 | 0 | **0** | (disabled) |
| R5_NEGATION | 0 | 0 | 0 | **0** | (disabled) |
| R6a_CROSS_LAT | 1 | 0 | 0 | **1** | 跨句侧别冲突 |
| R6b_CROSS_PRES | 329 | 48 | 46 | **423** | 跨句存在性冲突 |

**Residual violation 构成**:
- R3_DEPTH 占 69.5% — 主要残余违规源，octree 深度分配与报告描述的细粒度不完全匹配
- R6b_CROSS_PRESENCE 占 27.2% — 跨句 presence 冲突
- R1_LATERALITY 占 3.2% — Evidence Card v2 已大幅压低
- R6a + R2 约 0.1%

### 4.3 Per-Dataset Breakdown (Test Split)

| Dataset | Sentences | Violations | Viol. Rate | R1 | R3 | R6b | BLEU-4 | ROUGE-L | METEOR |
|---------|-----------|------------|------------|-----|-----|------|--------|---------|--------|
| CT-RATE | 1,979 | 88 | 4.45% | 2 | 64 | 22 | 0.467 | 0.626 | 0.603 |
| RadGenome | 2,000 | 73 | 3.65% | 0 | 49 | 24 | 0.506 | 0.680 | 0.660 |

RadGenome 在所有指标上均优于 CT-RATE：违规率更低 (3.65% vs 4.45%)，NLG 更高 (BLEU-4 0.506 vs 0.467)。这与 RadGenome 报告本身更规范、结构化更好一致。

### 4.4 Repair Statistics

| Split | Judge Confirmed | Rerouted | Despecified |
|-------|----------------|----------|-------------|
| train | 904 | 904 | 21 |
| valid | 105 | 105 | 0 |
| test | 122 | 122 | 3 |
| **Total** | **1,131** | **1,131** | **24** |

- LLM Judge 从 1,553 个违规中确认了 1,131 个 (72.8%)，说明 ~27% 的 rule-based 违规被 LLM 判定为 false positive
- 所有确认的违规均触发了 reroute

---

## 5. Comparison: 900-case vs 5000-case

| Metric | 900-case B2'v2 | 5000-case (test) | 5000-case (all) |
|--------|---------------|-------------------|------------------|
| Cases | 450 | 500 | 5000 |
| Sentences | 1,437 | 3,979 | 39,893 |
| Violation Rate | 3.90% | 4.05% | 3.89% |
| R1 | 39 | 2 | 49 |
| R3 | 4 | 113 | 1,080 |
| R6a | 0 | 0 | 1 |
| R6b | 13 | 46 | 423 |
| BLEU-4 | 0.463 | 0.487 | 0.492 |
| ROUGE-L | 0.629 | 0.653 | 0.656 |
| METEOR | 0.608 | 0.632 | 0.636 |

**Analysis**:
- 违规率从 3.90% (900) → 3.89% (5k overall)，完美一致，说明 pipeline 在更大规模数据上无退化
- NLG 指标在 5k 上反而略高 (BLEU-4 0.463→0.492, ROUGE-L 0.629→0.656)，可能因为 5k 采样了更多"规范"报告
- R1 大幅下降 (39→2 in test)，说明 Evidence Card v2 的 strict laterality gate 在新数据上效果更好
- R3 是主要残余违规源，在更大数据上占比更突出

---

## 6. Split 一致性分析

| Metric | Train | Valid | Test | Std |
|--------|-------|-------|------|-----|
| Violation Rate | 3.86% | 3.98% | 4.05% | 0.08% |
| BLEU-4 | 0.492 | 0.502 | 0.487 | 0.006 |
| ROUGE-L | 0.656 | 0.661 | 0.653 | 0.003 |
| METEOR | 0.636 | 0.640 | 0.632 | 0.003 |

所有 split 之间的标准差极小 (Violation Rate σ=0.08%, BLEU-4 σ=0.006)，进一步证实了 pipeline 的稳定性和泛化能力。数据集划分无显著偏差。

---

## 7. 运行统计

| Item | Value |
|------|-------|
| 总运行时间 | ~28 小时 |
| 平均每 case | ~20 秒 |
| GPU 占用 | 1× (Llama-3.1-8B bfloat16) |
| 输出目录 | `outputs/stage0_5_5k/{train,valid,test}/` |
| 输出大小 | 40 GB |

---

## 8. 目录结构

```
outputs/stage0_5_5k/
├── train/
│   ├── cases/
│   │   ├── ctrate/        (2000 cases)
│   │   └── radgenome/     (2000 cases)
│   ├── summary.csv
│   ├── run_meta.json
│   └── run.log
├── valid/
│   ├── cases/
│   │   ├── ctrate/        (250 cases)
│   │   └── radgenome/     (250 cases)
│   ├── summary.csv
│   └── run_meta.json
└── test/
    ├── cases/
    │   ├── ctrate/        (250 cases)
    │   └── radgenome/     (250 cases)
    ├── summary.csv
    └── run_meta.json
```

---

## 9. W_proj Ablation: Trained W_proj + E1 Routing

### 9.1 实验设置

在 train split (4000 cases, 31,919 sentence pairs) 上训练 W_proj 投影矩阵 (InfoNCE loss)，valid split (500 cases) 做 early stopping，然后在 test split (500 cases) 上评估。

| Parameter | Value |
|-----------|-------|
| W_proj 训练数据 | 4000 cases (31,919 pairs) |
| Loss | InfoNCE (tau=0.07) |
| Epochs | 50 (no early stop triggered) |
| Batch size | 64 |
| LR | 1e-3 (CosineAnnealing) |
| Best val loss | 2.496 |
| Routing | `spatial_filter_semantic_rerank` (E1: IoU filter → W_proj cosine rerank) |

### 9.2 Test Split 对比

| Metric | B2'v2 (identity + spatial) | Trained W_proj + E1 | 变化 |
|--------|---------------------------|---------------------|------|
| **Violation Rate** | **4.05%** | 4.32% | +0.27% ↑ |
| R1_LATERALITY | **2** | 105 | +103 ↑ |
| R3_DEPTH | 113 | **23** | **-90 ↓** |
| R6b_CROSS_PRES | 46 | **44** | -2 |
| Total violations | **161** | 172 | +11 ↑ |
| BLEU-4 | **0.487** | 0.477 | -1.0% |
| ROUGE-L | **0.653** | 0.647 | -0.6% |
| METEOR | **0.632** | 0.623 | -0.9% |

### 9.3 Per-Dataset Breakdown

| Config | Dataset | Viol. Rate | R1 | R3 | R6b | BLEU-4 | ROUGE-L |
|--------|---------|------------|-----|-----|------|--------|---------|
| B2'v2 identity | CT-RATE | 4.45% | 2 | 64 | 22 | 0.467 | 0.626 |
| B2'v2 identity | RadGenome | 3.65% | 0 | 49 | 24 | 0.506 | 0.680 |
| Trained W_proj E1 | CT-RATE | 4.70% | 65 | 9 | 19 | 0.454 | 0.621 |
| Trained W_proj E1 | RadGenome | 3.95% | 40 | 14 | 25 | 0.500 | 0.672 |

### 9.4 分析

- **R3_DEPTH 大幅下降** (113→23, -80%): W_proj 语义 rerank 有效帮助选择了深度层级更匹配的 token，证明 learned projection 在 depth matching 上有明确价值
- **R1_LATERALITY 暴增** (2→105): E1 routing 的语义 rerank 步骤破坏了纯空间路由的侧别一致性。W_proj 优化的是语义相关性（InfoNCE），而非空间一致性，导致 rerank 后的 token 可能跨越中线
- **总违规率上升** (4.05%→4.32%): R3 的改善不足以弥补 R1 的恶化
- **NLG 略降**: BLEU-4 -1.0%, ROUGE-L -0.6%，差异不大

### 9.5 结论

**B2'v2 (identity W_proj + spatial routing) 仍为最优配置。** Trained W_proj 在深度匹配 (R3) 上有明确改善，但引入了严重的侧别违规 (R1)。未来优化方向：在 InfoNCE 训练中加入侧别约束项，或在 E1 rerank 阶段增加 laterality-aware filtering。

---

## 10. Conclusion

1. **规模验证通过**: Pipeline 在 5000 cases (39,893 sentences) 上保持了与 900-case 实验一致的违规率 (~3.9%) 和 NLG 质量
2. **泛化性强**: Train/Valid/Test split 之间指标高度一致 (σ < 0.01)，无过拟合或数据偏差迹象
3. **残余违规分析**: R3_DEPTH (69.5%) 和 R6b_CROSS_PRESENCE (27.2%) 是主要残余来源，R1_LATERALITY 已被 Evidence Card v2 压至接近 0
4. **NLG 质量**: BLEU-4 ≈ 0.49, ROUGE-L ≈ 0.65, METEOR ≈ 0.63，在 CT 报告生成任务中属于较高水平
5. **W_proj 消融**: Trained W_proj + E1 routing 大幅降低 R3 (-80%) 但 R1 暴增 (+103)，总违规率略升。B2'v2 仍为最优
6. **下一步**: 探索 laterality-aware InfoNCE 训练，在语义 rerank 中保持侧别一致性
