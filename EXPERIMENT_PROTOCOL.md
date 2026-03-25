# ProveTok Experiment Protocol

> 本文档是论文 Experiment 部分的完整参考，包含数据集、实验设置、消融链定义、评估指标和统计检验的所有细节。

---

## 1. 数据集

### 1.1 数据来源

| Dataset | Source | License | 内容 |
|---------|--------|---------|------|
| CT-RATE | [ibrahimhamamci/CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) | CC BY-NC-SA 4.0 | 3D 胸部 CT volume + 放射学报告 |
| RadGenome-Chest CT | [RadGenome/RadGenome-ChestCT](https://huggingface.co/datasets/ibrahimhamamci/RadGenome-ChestCT) | CC BY-NC-SA 4.0 | 3D 胸部 CT volume + 结构化放射学报告 |

### 1.2 数据划分

从两个数据集各采样 2500 个 case，按 80/10/10 划分（seed=42）：

| Split | CT-RATE | RadGenome | Total | 用途 |
|-------|---------|-----------|-------|------|
| train | 2000 | 2000 | 4000 | W_proj 训练 |
| valid | 250 | 250 | 500 | Early stopping |
| test | 250 | 250 | 500 | **最终评估（所有论文数据来源）** |
| **Total** | **2500** | **2500** | **5000** | |

- Manifest 文件：`manifests/{ctrate,radgenome}_5k_{train,valid,test}_manifest.csv`
- 划分脚本：`Scripts/split_train_val_test.py`（seed=42 确保确定性）
- 因 4 个 case 的 CT volume 损坏，实际 test split 为 496 cases（非 500）

### 1.3 预处理

| 步骤 | 操作 | 参数 |
|------|------|------|
| 加载 | SimpleITK 读取 .nii.gz | — |
| 强度归一化 | CT intensity normalization | clip → [0, 1] |
| 空间重采样 | Trilinear resize to isotropic volume | 128 × 128 × 128 |

所有配置共享相同的预处理流程和缓存（Stage 0-1 encoder 输出在第一个配置运行后缓存复用）。

---

## 2. 模型组件

### 2.1 固定组件（所有配置共享）

| 组件 | 模型 | 参数量 | 状态 |
|------|------|--------|------|
| 3D Encoder | SwinUNETR (MONAI) | 62M | Frozen（自监督预训练权重） |
| Text Encoder | all-MiniLM-L6-v2 | 22M | Frozen |
| Octree Splitter | Adaptive Octree | 0 | 确定性规则 |
| Verifier | Rule-based (R1-R6) | 0 | 确定性规则 |

### 2.2 可训练组件

| 组件 | 参数 | 训练方式 |
|------|------|---------|
| W_proj | 384 × 768 = 295K params | InfoNCE loss, AdamW, cosine schedule |

### 2.3 LLM（Stage 3c 生成 + Stage 5 Judge）

| 设置 | 值 |
|------|-----|
| 模型 | Llama-3.1-8B-Instruct |
| 精度 | bfloat16 |
| Temperature | 0.3 |
| Max tokens | 256 |
| Backend | HuggingFace Transformers (local) |

---

## 3. 七配置消融链

### 3.1 消融设计

每个配置在前一个基础上增加一个组件，隔离各组件的边际贡献：

```
A0 ──→ A1 ──→ E1 ──→ B2' ──→ B2'v2 ──→ C2' ──→ D2
 │       │       │      │        │        │       │
 │       │       │      │        │        │       └─ +Repair executor
 │       │       │      │        │        └─ +LLM Judge (Stage 5)
 │       │       │      │        └─ +Evidence Card v2 (strict)
 │       │       │      └─ +LLM generation + Evidence Card v1
 │       │       └─ Spatial filter + Semantic rerank
 │       └─ +Trained W_proj
 └─ Baseline: Identity W_proj + Spatial routing
```

### 3.2 配置详细定义

| Config | Routing | W_proj | LLM Gen | Evidence Card | Judge | Repair |
|--------|---------|--------|---------|---------------|-------|--------|
| **A0** | anatomy_spatial (IoU-primary) | Identity | — | — | — | — |
| **A1** | anatomy_spatial (IoU-primary) | Trained | — | — | — | — |
| **E1** | spatial_filter + semantic_rerank | Trained | — | — | — | — |
| **B2'** | spatial_filter + semantic_rerank | Trained | Llama-3.1-8B | v1 (SSR ≥ 0.8) | — | — |
| **B2'v2** | spatial_filter + semantic_rerank | Trained | Llama-3.1-8B | v2 (SSR ≥ 0.9, strict) | — | — |
| **C2'** | spatial_filter + semantic_rerank | Trained | Llama-3.1-8B | v1 | Llama-3.1-8B | — |
| **D2** | spatial_filter + semantic_rerank | Trained | Llama-3.1-8B | v1 | Llama-3.1-8B | log-smooth reroute |

### 3.3 公共超参数

所有 7 个配置共享以下参数：

| 参数 | 值 | 含义 |
|------|-----|------|
| `token_budget_b` | 128 | Token bank 最大 token 数 |
| `k_per_sentence` | 8 | 每个句子路由的 top-k token 数 |
| `lambda_spatial` | 0.3 | 空间 IoU prior 权重 |
| `tau_iou` | 0.04 | R2 anatomy IoU 阈值 |
| `beta` | 0.1 | Boundary blending 强度 |
| `r2_min_support_ratio` | 0.8 | R2 ratio mode 支持率阈值 |
| `r4_disabled` | True | R4 SIZE 规则禁用（未校准） |
| `r1_min_same_side_ratio` | 0.6 | R1 同侧比例阈值 |
| `r1_negation_exempt` | True | 否定句跳过 R1 检查 |
| `r1_skip_midline` | True | 中线解剖跳过 R1 检查 |
| `r2_skip_bilateral` | True | bilateral 句跳过 R2 检查 |
| `shuffle_seed` | 42 | Manifest 随机种子 |
| `resize_dhw` | 128 × 128 × 128 | 输入 volume 尺寸 |

### 3.4 Evidence Card v1 vs v2 区别

| 属性 | v1 (B2', C2', D2) | v2 (B2'v2) |
|------|-------------------|------------|
| SSR 阈值 | ≥ 0.8 | ≥ 0.9 |
| 最少非 cross tokens | 1 | 2 |
| 同侧 token cleanup | 无 | 移除对侧 token |
| 深度范围 cleanup | 无 | 移除 out-of-range token |
| `--strict_laterality` | False | True |

---

## 4. W_proj 训练协议

### 4.1 训练设置

| 设置 | 值 |
|------|-----|
| 训练数据 | train split (4000 cases) 的 Stage 0-4 输出 |
| 输入 | (sentence_text, topk_token_ids) pairs from trace.jsonl |
| Loss | Multi-positive InfoNCE (Eq.12) |
| 温度 τ | 0.07 |
| 优化器 | AdamW |
| 学习率 | 1e-3 + cosine annealing |
| Epochs | 50 |
| Early stopping | 基于 validation loss |
| 初始化 | Orthogonal initialization |
| 输出维度 | 384 × 768 (text_dim × token_dim) |

### 4.2 训练结果

| 指标 | 值 |
|------|-----|
| 最终 train loss | 见 `outputs/wprojection_5k/train_log.json` |
| 收敛 epoch | Early stopping |
| 权重文件 | `outputs/wprojection_5k/w_proj_best.pt` |

---

## 5. 评估指标

### 5.1 主指标：Violation Rate（空间一致性）

$$\text{Viol\%} = \frac{\text{违规句数}}{\text{总句数}} \times 100$$

每个句子可触发 0 或多条违规规则，只要 ≥ 1 条违规即计为违规句。

### 5.2 规则定义

| Rule | 检测内容 | 严重度 | 触发条件 |
|------|---------|--------|---------|
| **R1** LATERALITY | 侧别一致性 | 1.0 | 句子声明 left/right，但 cited tokens 的同侧比例 < 0.6 |
| **R2** ANATOMY | 解剖 IoU | 0.8 | Cited tokens 与解剖 bbox 的 IoU 支持率 < 0.8 |
| **R3** DEPTH | 深度层级 | 0.7 | 任一 cited token 的 octree level 在预期范围外 |
| **R4** SIZE | 体积范围 | 0.6 | **禁用**（阈值未校准） |
| **R5** NEGATION | 否定一致性 | 1.0 | 否定句 cite 到阳性 token（**fallback lexicon 禁用**） |
| **R6a** CROSS_LAT | 跨句侧别冲突 | 0.8 | 同一解剖关键词的不同句子声明不同侧别 |
| **R6b** CROSS_PRES | 跨句存在冲突 | 0.7 | 同一解剖关键词一句肯定一句否定 |

活跃规则：R1, R2, R3, R6a, R6b（5 条）。

### 5.3 辅助指标：NLG Metrics

| 指标 | 说明 | 粒度 |
|------|------|------|
| BLEU-4 | n-gram precision | sentence-level, 取均值 |
| ROUGE-L | Longest common subsequence | sentence-level, 取均值 |
| METEOR | Unigram alignment + stemming | sentence-level, 取均值 |

**重要说明**：Planner 使用 reference report 原文作为句子 plan（`topic=reference_sentence`），因此所有配置的 NLG scores 天然较高。NLG metrics 在本实验中是**控制变量**（确保 evidence gating 不破坏文本质量），不是主要贡献指标。

### 5.4 Grounding Metrics（Table 3）

| 指标 | 说明 |
|------|------|
| Violation-free Rate | 1 - Viol% |
| Laterality Accuracy | 侧别匹配句数 / 有侧别声明的总句数 |
| Citation Faithfulness | LLM 生成句的侧别/深度与 evidence card 约束一致的比例 |
| Depth Faithfulness | LLM 生成句的深度描述与 cited tokens 深度一致的比例 |
| Mean Overlap | Cited tokens 与 anatomy bbox 的平均 IoU |
| Hit@k | Top-k tokens 中至少一个落在目标 anatomy bbox 内的比例 |

---

## 6. 统计检验

### 6.1 Bootstrap 置信区间

| 设置 | 值 |
|------|-----|
| 方法 | Patient-level paired bootstrap |
| 重采样次数 R | 5000 |
| 显著性水平 α | 0.05 |
| 置信区间 | Percentile method (2.5th, 97.5th) |
| 采样单位 | Patient（不是 sentence） |

### 6.2 置换检验

| 设置 | 值 |
|------|-----|
| 方法 | Paired permutation test |
| 比较对 | 相邻配置（A0→A1, A1→E1, E1→B2', B2'→B2'v2, B2'v2→C2', C2'→D2） |
| 多重比较校正 | Holm step-down |
| 置换次数 | 10000 |

### 6.3 统计结果

| Comparison | Δ Viol% | p_adjusted | Significant |
|------------|---------|------------|-------------|
| A0 → A1 | −1.63 | 0.008 | **Yes** |
| A1 → E1 | +0.08 | 1.000 | No |
| E1 → B2' | −0.19 | 1.000 | No |
| B2' → B2'v2 | **−1.65** | **0.006** | **Yes** |
| B2'v2 → C2' | +1.80 | 0.006 | Yes (regression) |
| C2' → D2 | −0.21 | 1.000 | No |

两个显著 drop：**A0→A1**（trained W_proj）和 **B2'→B2'v2**（Evidence Card v2 strict gating）。

---

## 7. 主要实验结果

### 7.1 Table 2: 消融链（Violation Rate + Rule 分布）

| Config | Viol% | CI (95%) | R1 | R3 | R6b | LLM calls |
|--------|-------|----------|-----|-----|------|-----------|
| A0 | 7.59 | [6.84, 8.35] | 0 | 237 | 65 | 0 |
| A1 | 6.01 | [5.27, 6.64] | 0 | 174 | 65 | 0 |
| E1 | 6.18 | [5.34, 6.77] | 95 | 86 | 65 | 0 |
| B2' | 6.01 | [5.14, 6.60] | 109 | 86 | 44 | 3,979 |
| **B2'v2** | **4.25** | **[3.60, 4.83]** | **98** | **23** | **48** | **3,979** |
| C2' | 6.18 | [5.25, 6.79] | 113 | 86 | 47 | 4,112 |
| D2 | 5.96 | [5.07, 6.55] | 102 | 86 | 49 | 4,116 |

### 7.2 NLG Metrics（仅 LLM 生成配置有值）

| Config | BLEU-4 | ROUGE-L | METEOR |
|--------|--------|---------|--------|
| B2' | 0.471 | 0.643 | 0.618 |
| B2'v2 | 0.473 | 0.643 | 0.619 |
| C2' | 0.477 | 0.643 | 0.622 |
| D2 | 0.474 | 0.643 | 0.622 |

NLG 指标在 4 个 LLM 配置间差异 < 1%，确认 evidence gating 不影响文本质量。

### 7.3 Bootstrap CI（Viol%）

| Config | Mean | 95% CI |
|--------|------|--------|
| A0 | 7.59 | [6.84, 8.35] |
| A1 | 5.96 | [5.27, 6.64] |
| E1 | 6.04 | [5.34, 6.77] |
| B2' | 5.85 | [5.14, 6.60] |
| **B2'v2** | **4.19** | **[3.60, 4.83]** |
| C2' | 6.00 | [5.25, 6.79] |
| D2 | 5.79 | [5.07, 6.55] |

---

## 8. 补充实验

### 8.1 k-sweep（Sensitivity to k）

在 B2'v2 配置基础上，测试不同 k 值：

| k | 说明 |
|---|------|
| 1 | 单 token citation |
| 4 | 较少 citation |
| **8** | **论文默认值** |
| 12 | 较多 citation |

数据来源：`outputs/paper_figures/fig2_budget_sweep.pdf`

### 8.2 Counterfactual Analysis

测试 evidence token 是否真的被 LLM 使用：
- **Wrong-location counterfactual**: 用随机 token 替换 routed tokens，观察生成结果变化
- **Wrong-content counterfactual**: 保持 token 位置但替换 feature，观察生成结果变化

数据来源：`outputs/paper_figures_5k/fig3_counterfactual_data.json`

---

## 9. 硬件与运行时间

| 资源 | 规格 |
|------|------|
| 平台 | Azure VM |
| GPU | 1× NVIDIA GPU |
| 存储 | 1TB 挂载盘 |
| CT 数据总量 | ~495 GB |

| 配置类型 | 每配置运行时间（500 test cases） |
|---------|-------------------------------|
| A0, A1, E1（无 LLM） | ~1-2 小时 |
| B2', B2'v2, C2', D2（有 LLM） | ~3-5 小时 |
| 7 配置全部 | ~15-23 小时 |
| W_proj 训练 | ~30 分钟 |

---

## 10. 运行命令

### 10.1 完整消融链

```bash
bash Scripts/run_5k_ablation_chain.sh
```

### 10.2 单个配置示例（B2'v2）

```bash
python run_mini_experiment.py \
    --ctrate_csv manifests/ctrate_5k_test_manifest.csv \
    --radgenome_csv manifests/radgenome_5k_test_manifest.csv \
    --max_cases 250 \
    --cp_strict \
    --encoder_ckpt checkpoints/swinunetr.ckpt \
    --text_encoder semantic \
    --text_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
    --device cuda \
    --shuffle_seed 42 \
    --token_budget_b 128 \
    --k_per_sentence 8 \
    --lambda_spatial 0.3 \
    --tau_iou 0.04 \
    --beta 0.1 \
    --r2_mode ratio \
    --r2_min_support_ratio 0.8 \
    --r4_disabled \
    --r1_negation_exempt \
    --r1_skip_midline \
    --r1_min_same_side_ratio 0.6 \
    --r2_skip_bilateral \
    --spatial_filter_semantic_rerank \
    --w_proj_path outputs/wprojection_5k/w_proj_best.pt \
    --stage3c_backend huggingface \
    --stage3c_model models/Llama-3.1-8B-Instruct \
    --stage3c_temperature 0.3 \
    --strict_laterality \
    --out_dir outputs/5k_ablation/B2_evcard_v2
```

### 10.3 生成论文图表

```bash
# Table 2 + Figure 1 (waterfall) + Figure 2 (k-sweep)
python Scripts/generate_table2_and_figures.py

# 统计分析：Bootstrap CI + 置换检验
python Scripts/run_statistical_analysis.py
```

### 10.4 NLG 评估

```bash
python Scripts/evaluate_metrics.py
```

---

## 11. 数据文件索引

| 文件 | 内容 | 论文对应 |
|------|------|---------|
| `outputs/5k_ablation/*/summary.csv` | 7 个配置的 per-case 结果 | 原始数据 |
| `outputs/5k_ablation/*/run_meta.json` | 完整实验参数 | 复现参考 |
| `outputs/paper_figures_5k/table2_data.json` | 消融链汇总数据 | Table 2 |
| `outputs/paper_figures_5k/table3_grounding_data.json` | Grounding 指标 | Table 3 |
| `outputs/evaluation_5k/metrics_all_conditions.*` | NLG 指标 | Table 2 NLG 列 |
| `outputs/paper_figures_5k/statistical_analysis/bootstrap_ci.csv` | Bootstrap CI | Table 2 CI 列 |
| `outputs/paper_figures_5k/statistical_analysis/pairwise_permutation.csv` | 置换检验 p 值 | 正文统计分析 |
| `outputs/paper_figures_5k/statistical_analysis/rule_distribution.csv` | 规则分布 | Table 2 Rule 列 |
| `outputs/paper_figures_5k/fig1_waterfall.*` | 违规率瀑布图 | Figure 1 |
| `outputs/paper_figures_5k/fig2_budget_sweep.*` | k-sweep | Figure 2 |
| `outputs/paper_figures_5k/fig3_counterfactual.*` | Counterfactual | Figure 3 |
| `outputs/paper_figures/fig4*.{pdf,png}` | 定性案例 | Figure 4 |
| `outputs/paper_figures/fig5_comparative.*` | Baseline vs Best | Figure 5 |
| `outputs/wprojection_5k/train_log.json` | W_proj 训练曲线 | Appendix |
