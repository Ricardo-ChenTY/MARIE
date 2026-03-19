# ProveTok 实验全流程记录

> 从 450-case 小规模验证到 5000-case 大规模实验的完整旅程
> **时间跨度**: 2026-03 ~ 2026-03
> **代码仓库**: ProveTok_ACM

---

## 1. Pipeline 概览

ProveTok 是一个 **3D CT 报告生成管线**，核心思想是将 CT volume 切分为 3D token，把报告句子路由到对应的空间区域，从而实现**可溯源**的报告生成。

```
CT Volume → [Stage 0] Artifact Score
          → [Stage 1] SwinUNETR Encoder → token features
          → [Stage 2] Adaptive Octree Splitter → 3D token bank
          → [Stage 3] Router: sentence → top-k tokens (citation)
          → [Stage 3c] LLM Generation (Evidence Card → sentence)
          → [Stage 4] Verifier: check spatial consistency
          → [Stage 5] LLM Judge: confirm/dismiss violations
          → [Reroute] Log-smooth penalty → re-cite → regenerate
```

### 两层评估体系

| Layer | 衡量对象 | 指标 | 目标 |
|-------|---------|------|------|
| **Layer 1: Spatial Consistency** | Token citation 是否空间正确 | 违规率 (R1-R6 rules) | ↓ 越低越好 |
| **Layer 2: NLG Quality** | 生成文本是否忠实于原始报告 | BLEU-4, ROUGE-L, METEOR | ↑ 越高越好 |

**Layer 1 规则说明**:
- **R1_LATERALITY**: 报告说 "left"，cited token 是否在左侧？
- **R2_ANATOMY**: Cited token 的 bbox 与目标解剖区域 IoU 是否足够？
- **R3_DEPTH**: Cited token 的 octree 深度层级是否匹配描述的细粒度？
- **R6a/R6b**: 跨句引用同一 token 时，侧别/存在性声明是否冲突？

---

## 2. Phase 1: 小规模验证 (450/450 cases)

> **报告**: `EXPERIMENT_REPORT_450.md`

### 2.1 目标

验证 ProveTok pipeline 的基本功能：Stage 0-4 能否正确处理 CT volume，生成 token bank，路由句子，并检测违规。

### 2.2 数据

| Dataset | Cases | Source |
|---------|-------|--------|
| CT-RATE | 450 | `ibrahimhamamci/CT-RATE` |
| RadGenome | 450 | `RadGenome/RadGenome-ChestCT` |
| **Total** | **900** | |

### 2.3 配置

- Routing: identity W_proj + spatial routing
- Token budget B = 128, k = 8
- SwinUNETR encoder + semantic text encoder
- **无 Stage 3c/5** (无 LLM 生成/判断)

### 2.4 结果

| Metric | Value |
|--------|-------|
| Total cases | 900 |
| Total violations | 974 |
| Cases with violations | 69.1% |
| Avg violations/case | 1.08 |

**发现**: 基线违规率较高，主要来自 R1 (侧别) 和 R3 (深度)。需要优化路由和添加 LLM 生成。

---

## 3. Phase 2: 路由消融实验 (900 cases)

> **报告**: `EXPERIMENT_REPORT_ROUTING_ABLATION.md`

### 3.1 目标

系统性消融，找到最优的路由 + 生成 + 验证 + 修复配置。

### 3.2 实验矩阵

分 8 个阶段逐步叠加组件：

| Phase | Config | 描述 |
|-------|--------|------|
| **A0** | Identity W_proj + Spatial routing | 基线 |
| **A1** | Trained W_proj + Spatial routing | 学习投影 |
| **A2** | Identity W_proj + Semantic routing | 语义路由 |
| **A3** | Trained W_proj + Semantic routing | 学习 + 语义 |
| **B2** | E1 (spatial filter → semantic rerank) + Stage 3c | +LLM 生成 |
| **C2** | B2 + Stage 5 LLM Judge | +LLM 判断 |
| **D2** | C2 + Log-smooth Reroute | +修复 |
| **B2'** | B2 + Evidence Card v1 | +Evidence Card |
| **B2'v2** | B2 + Evidence Card v2 | +严格侧别/深度门控 |

### 3.3 关键发现

**Evidence Card v2 (B2'v2) 是最优配置**，在所有指标上取得最佳平衡：

| Config | Violation Rate | R1 | R3 | BLEU-4 | ROUGE-L |
|--------|---------------|-----|-----|--------|---------|
| A0 (基线) | ~15% | 132 | 2 | — | — |
| B2 (+LLM gen) | ~12% | 128 | 19 | 0.482 | 0.650 |
| B2'v2 (+EvCard v2) | **3.9%** | **39** | **4** | 0.463 | 0.629 |

**核心洞察**:
1. LLM 生成 (Stage 3c) 提供了文本质量，但引入了空间违规
2. Evidence Card v2 通过 3 个 patch 解决了这个矛盾：
   - P1: Strict laterality gate (SSR ≥ 0.9)
   - P2: Depth gate (≥75% tokens in range)
   - P3: Same-side token cleanup
3. 违规率从 ~15% → 3.9%，NLG 代价仅 ~4%

### 3.4 k-sweep

| k | Violations | R1 | R3 |
|---|-----------|-----|-----|
| 1 | 151 | 132 | 2 |
| 4 | 181 | 157 | 7 |
| **8** | **164** | **128** | **19** |
| 12 | 220 | 168 | 35 |

k=8 是违规率最低点（排除 k=1 的退化情况）。

---

## 4. Phase 3: W_proj 训练实验 (900 cases)

> **报告**: `EXPERIMENT_REPORT_WPROJ.md`

### 4.1 目标

训练 cross-modal projection W_proj (InfoNCE loss)，看是否能提升语义路由质量。

### 4.2 结果

| Metric | Identity W_proj | Trained W_proj |
|--------|----------------|----------------|
| InfoNCE loss | 2.915 | 1.693 (-41.9%) |
| Total violations | 191 | 285 (+49.2%) |

**反直觉发现**: W_proj 训练成功降低了 InfoNCE loss，但下游违规率反而大幅上升。

**根因分析**: 同时改变了两个变量（W_proj + routing mode），trained W_proj 使用 semantic routing 而非 spatial routing，导致侧别信息丢失。

---

## 5. Phase 4: 5K 大规模验证

> **报告**: `EXPERIMENT_REPORT_5K.md`

### 5.1 目标

在 5000 cases 上验证 B2'v2 最优配置的泛化性和规模稳定性。

### 5.2 数据

| Split | CT-RATE | RadGenome | Total | 用途 |
|-------|---------|-----------|-------|------|
| train | 2000 | 2000 | 4000 | W_proj 训练 |
| valid | 250 | 250 | 500 | 验证 |
| test | 250 | 250 | 500 | 最终评估 |
| **Total** | **2500** | **2500** | **5000** | |

存储：495 GB (1TB Azure 挂载盘)

### 5.3 结果

| Split | Cases | Sentences | Viol. Rate | BLEU-4 | ROUGE-L | METEOR |
|-------|-------|-----------|------------|--------|---------|--------|
| train | 4000 | 31,919 | 3.86% | 0.492 | 0.656 | 0.636 |
| valid | 500 | 3,995 | 3.98% | 0.502 | 0.661 | 0.640 |
| test | 500 | 3,979 | 4.05% | 0.487 | 0.653 | 0.632 |
| **All** | **5000** | **39,893** | **3.89%** | **0.492** | **0.656** | **0.636** |

**关键发现**:
1. **规模稳定**: 违规率 900→5000 保持 ~3.9%，完美一致
2. **泛化性强**: Train/Valid/Test σ < 0.01
3. **残余违规**: R3_DEPTH 69.5%, R6b_CROSS_PRES 27.2%, R1 已接近 0

### 5.4 5K W_proj 消融

用 train 4000 cases 训练 W_proj，在 test 500 cases 上评估：

| Config | Viol. Rate | R1 | R3 | BLEU-4 |
|--------|-----------|-----|-----|--------|
| B2'v2 (identity + spatial) | **4.05%** | **2** | 113 | **0.487** |
| Trained W_proj + E1 | 4.32% | 105 | **23** | 0.477 |

Trained W_proj 大幅压低 R3 (-80%)，但 R1 暴增 (+103)。**B2'v2 仍为最优。**

---

## 6. 实验演进总结

```
Phase 1 (450/450)           Phase 2 (900 routing)        Phase 4 (5K)
────────────────           ──────────────────────       ──────────────
基线验证                    系统消融                     大规模验证
违规率 ~15%         →      B2'v2: 3.9%          →      3.89% (稳定)
无 LLM 生成                 +Evidence Card v2            +W_proj 消融
                            +LLM 生成/判断/修复
```

### 关键里程碑

| 阶段 | 违规率 | 关键突破 |
|------|--------|---------|
| Stage 0-4 基线 (A0) | ~15% | Pipeline 功能验证 |
| +LLM 生成 (B2) | ~12% | 文本质量提升，但引入空间违规 |
| +Evidence Card v1 (B2') | ~7.5% | 侧别/深度约束初步生效 |
| **+Evidence Card v2 (B2'v2)** | **3.9%** | **严格门控，违规率降 74%** |
| 5K 验证 | 3.89% | 规模稳定性确认 |

### 最终配置 (B2'v2)

| Component | Choice | 理由 |
|-----------|--------|------|
| Routing | Spatial (IoU-based) | 纯空间路由保持侧别一致性 |
| W_proj | Identity | Trained W_proj 破坏 R1 |
| Generation | Llama-3.1-8B + Evidence Card v2 | 严格门控 + LLM 生成 |
| Verification | R1/R2/R3/R6 rules | 自动空间一致性检查 |
| Repair | LLM Judge + Log-smooth reroute | 确认违规 → 重引用 → 重生成 |

---

## 7. 残余问题与下一步

### 7.1 残余违规构成 (5K test)

| Rule | Count | 占比 | 原因 |
|------|-------|------|------|
| R3_DEPTH | 113 | 70.2% | Octree 深度分配与报告细粒度不完全匹配 |
| R6b_CROSS_PRES | 46 | 28.6% | 跨句 presence 冲突 |
| R1_LATERALITY | 2 | 1.2% | 已被 Evidence Card v2 压至接近 0 |

### 7.2 可能的优化方向

1. **Laterality-aware W_proj**: 在 InfoNCE 训练中加入侧别惩罚项，同时获得 R3 改善和 R1 低位
2. **R3 深度自适应**: 根据句子描述的细粒度动态调整 octree depth 约束
3. **R6b 跨句一致性**: 在 Evidence Card 中加入跨句 presence 检查

---

## 8. 文件索引

| 文件 | 内容 |
|------|------|
| `EXPERIMENT_REPORT_450.md` | Phase 1: 450/450 基线验证 |
| `EXPERIMENT_REPORT_ROUTING_ABLATION.md` | Phase 2: 路由消融 (11 configs + k-sweep + Evidence Card) |
| `EXPERIMENT_REPORT_WPROJ.md` | Phase 3: W_proj 训练实验 (900 cases) |
| `EXPERIMENT_REPORT_5K.md` | Phase 4: 5K 大规模验证 + W_proj 消融 |
| `Scripts/run_stage0_5_5k.sh` | 5K 实验运行脚本 |
| `Scripts/run_wprojection_train_5k.sh` | 5K W_proj 训练脚本 |
| `Scripts/run_stage0_5_5k_trained_wproj.sh` | Trained W_proj 评估脚本 |
| `Scripts/evaluate_metrics.py` | NLG + Spatial 评估工具 |
