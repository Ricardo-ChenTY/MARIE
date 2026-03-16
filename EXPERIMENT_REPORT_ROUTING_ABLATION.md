# ProveTok — Routing Ablation 实验报告

> 实验日期：2026-03-15 ~ 2026-03-16
> 输出目录：`outputs/ablation_routing_2x2/`（Phase A–D2）、`outputs/ablation_k_sweep/`（k sweep）

---

## 1. 背景与动机

在 W_proj 训练实验中，我们发现 **trained W_proj + semantic routing 反而比 identity W_proj + spatial routing 产生更多违规**（+49.2%）。这一违反直觉的结果需要系统性消融来定位原因。

### 1.1 研究问题

1. **W_proj 训练是否有效？** 在控制路由模式下，trained vs identity W_proj 是否降低违规？
2. **语义路由为什么更差？** 违规增长来自哪些具体规则？
3. **能否结合两者优势？** 空间约束的可靠性 + 语义选择的相关性？
4. **k 的最优值是多少？** 每句引用多少个 token 最合适？
5. **Stage 3c / Stage 5 能否进一步降低违规？** LLM 改写和 LLM 验证的增量贡献？
6. **Evidence Card 能否约束 LLM 的 laterality 输出？** 结构化空间摘要对生成质量的影响？
7. **针对性修复（repair executor）是否优于通用 reroute？** action-dispatched repair 的实际效果？

### 1.2 实验设计总览

所有实验统一：180 test cases（90 CT-RATE + 90 RadGenome），`token_budget_b = 128`。

| Phase | 实验 | 路由 | W_proj | Stage 3c | Stage 5 | Evidence Card | Repair | 目的 |
|-------|------|------|--------|:--------:|:-------:|:-------------:|:------:|------|
| A | A0–A3 | spatial / semantic | identity / trained | OFF | OFF | — | — | 隔离 W_proj × 路由模式的交互效应 |
| E1 | E1 | filter+rerank | trained | OFF | OFF | — | — | 结合空间过滤与语义重排 |
| D | k sweep | filter+rerank | trained | OFF | OFF | — | — | 确定最优 k_per_sentence |
| B | B2 | filter+rerank | trained | **ON** | OFF | — | — | Stage 3c 增量效果 |
| C | C2 | filter+rerank | trained | **ON** | **ON** | — | — | Stage 5 增量效果 |
| B2' | B2' | filter+rerank | trained | **ON** | OFF | **ON** | — | Evidence Card 对生成的约束效果 |
| C2' | C2' | filter+rerank | trained | **ON** | **ON** | **ON** | — | Evidence Card + Stage 5 组合效果 |
| D2 | D2 | filter+rerank | trained | **ON** | **ON** | **ON** | **ON** | Action-dispatched repair executor |

---

## 2. Phase A：2×2 Routing Ablation

### 2.1 实验矩阵

|  | Spatial routing | Semantic routing |
|--|:--:|:--:|
| **Identity W_proj** | A0 | A2 |
| **Trained W_proj** | A1 | A3 |

- **Spatial routing**：`anatomy_spatial_routing` — IoU 主导排序，语义仅作 tiebreaker（±ε=0.05）
- **Semantic routing**：`cos(q_s, W_proj · f_i) + λ_spatial · IoU`（λ=0.3）

### 2.2 总体结果

| 条件 | W_proj | Routing | Violations | Rate |
|------|--------|---------|:----------:|:----:|
| A0 | identity | spatial | 208 | 14.4% |
| **A1** | trained | spatial | **189** | **13.1%** |
| A2 | identity | semantic | 344 | 23.9% |
| A3 | trained | semantic | 292 | 20.3% |

> Rate = violations / 1440 sentences（180 cases × 8 sentences/case）

### 2.3 Per-Rule 分解

| 条件 | R1_LATERALITY | R2_ANATOMY | R3_DEPTH | R6b_CROSS |
|------|:---:|:---:|:---:|:---:|
| A0 | 100 | **0** | 91 | 17 |
| A1 | 92 | **0** | 80 | 17 |
| A2 | 114 | **87** | 126 | 17 |
| A3 | 118 | **58** | 99 | 17 |

### 2.4 关键发现

**Finding 1：W_proj 训练有效（在控制路由模式下）。**
- Spatial routing：A1 vs A0 = 189 vs 208（**−9.1%**）
- Semantic routing：A3 vs A2 = 292 vs 344（**−15.1%**）
- 两种路由模式下 W_proj 训练都降低了违规数。

**Finding 2：语义路由的核心问题是 R2（anatomy IoU）。**
- Spatial routing 下 R2 = 0（完美），因为 IoU 排序天然保证空间重叠。
- Semantic routing 下 R2 = 87（identity）/ 58（trained），贡献了语义路由增量违规的主要部分。
- 语义相似度高的 token 不一定在目标解剖区域内。

**Finding 3：R6b 是结构性常量。**
- 所有条件下 R6b ≈ 17，不受路由策略影响。这是数据本身的 cross-presence 问题。

**Finding 4：路由模式的影响远大于 W_proj。**
- A2 >> A0（+65.4%）：切换到语义路由导致大幅退化
- A3 >> A1（+54.5%）：即使 W_proj 训练过，语义路由仍远差于空间路由
- A1 < A0（−9.1%）：W_proj 的改进幅度相对较小

---

## 3. Phase E1：Spatial Filter + Semantic Rerank

### 3.1 动机

Phase A 表明：
- 空间路由保证 R2=0，但不考虑语义相关性
- 语义路由选择语义相关 token，但空间位置不对

**E1 方案**：先用空间约束硬过滤，再用语义相似度重排序。

### 3.2 算法

```
输入：topic 文本 q_s, token 集合 T, anatomy_bbox, expected_level_range
输出：每个 token 的综合得分

1. 对每个 token t_i：
   a. 计算语义分: sem_i = cos(q_s, W_proj · f_i)
   b. 检查空间过滤:
      - IoU(t_i.bbox, anatomy_bbox) ≥ τ_iou  (默认 0.04)
      - t_i.level ∈ [expected_lo, expected_hi]
   c. 分配得分:
      - 通过过滤: score_i = 1.0 + (sem_i + 1) / 2    ∈ [1, 2]
      - 未通过:   score_i = (sem_i + 1) / 2            ∈ [0, 1)

2. 按 score 降序取 top-k
```

**关键设计**：通过过滤的 token 得分 ∈ [1,2]，未通过的 ∈ [0,1)。这保证了任何空间合格 token 都排在不合格 token 前面，同时在每组内部按语义相关性排序。当合格 token 不足 k 个时，自动从不合格中按语义分补充。

### 3.3 结果

| 条件 | Violations | Rate | vs A1 |
|------|:----------:|:----:|:-----:|
| A1 (trained + spatial) | 189 | 13.1% | — |
| **E1** (trained + filter+rerank) | **164** | **11.4%** | **−13.2%** |

Per-Rule 对比：

| 规则 | A1 | E1 | Δ |
|------|:---:|:---:|:---:|
| R1_LATERALITY | 92 | 128 | +36 |
| R2_ANATOMY | 0 | 0 | 0 |
| R3_DEPTH | 80 | 19 | **−61** |
| R6b_CROSS | 17 | 17 | 0 |

### 3.4 分析

- **R2 = 0**：空间硬过滤成功继承了 spatial routing 的 R2 完美表现。
- **R3 大幅下降（80→19，−76%）**：E1 在空间过滤中加入了 level 范围约束，直接过滤掉深度层级不匹配的 token，而纯空间路由只看 IoU 不看 level。
- **R1 上升（92→128，+39%）**：语义 rerank 在空间合格 token 中优先选择了语义相关但位于对侧的 token。纯空间路由按 IoU 排序会倾向于选同侧（因为同侧 token 与 anatomy_bbox 的 IoU 通常更高）。
- **净效果**：R3 的大幅下降（−61）超过了 R1 的上升（+36），总违规下降 25。

---

## 4. Phase D：k_per_sentence Sweep

### 4.1 设置

在 E1 路由基础上，扫描 k ∈ {1, 2, 4, 6, 8, 12, 14}。

### 4.2 结果

| k | Violations | Rate | R1 | R2 | R3 | R6b |
|---|:----------:|:----:|:---:|:---:|:---:|:---:|
| 1 | 151 | 10.5% | 132 | 0 | 2 | 17 |
| 2 | 206 | 14.3% | 187 | 0 | 2 | 17 |
| 4 | 181 | 12.6% | 157 | 0 | 7 | 17 |
| 6 | 175 | 12.2% | 144 | 0 | 14 | 17 |
| **8** | **164** | **11.4%** | 128 | 0 | 19 | 17 |
| 12 | 220 | 15.3% | 168 | 0 | 35 | 17 |
| 14 | 226 | 15.7% | 165 | 0 | 44 | 17 |

### 4.3 分析

- **R2 = 0 across all k**：空间过滤在所有 k 值下都有效。
- **R3 随 k 单调递增**：k 越大，引入越多深度层不匹配的 token。k=1 时 R3 仅 2。
- **R1 呈非单调趋势**：k=1 时 R1=132（每句只 1 个 token 也有 laterality 违规），k=2 时跳到 187（最高），k=8 时 128，k=14 时 165。
- **k=2 异常点**：R1=187 显著高于相邻 k 值。推测当 k=2 时，top-2 token 恰好一左一右触发 laterality 违规的概率最高。
- **R6b = 17 不变**：结构性常量，不受 k 影响。

### 4.4 k 选择

k=1 虽然总违规最低（151），但每句只引用 1 个 token，证据过于稀薄，不足以支撑后续 Stage 3c 的 token-gated 生成。

**选择 k=8** 作为最终配置：
- 违规率 11.4%，在实用 k 值中最优
- 提供充足的证据 token 支撑文本生成
- R3 仍保持在较低水平（19）

---

## 5. Phase B2 & C2：Pipeline 逐层叠加

### 5.1 设置

在 E1（k=8）基础上逐步开启 pipeline 后续阶段：

| 条件 | 路由 | Stage 3c | Stage 5 | 说明 |
|------|------|:--------:|:-------:|------|
| E1 | filter+rerank | OFF | OFF | 仅路由（gold text） |
| B2 | filter+rerank | **ON** | OFF | + LLM token-gated 生成（Llama-3.1-8B） |
| C2 | filter+rerank | **ON** | **ON** | + LLM judge 验证（Llama-3.1-8B，α=0.5） |

### 5.2 结果

| 条件 | Violations | Rate | R1 | R2 | R3 | R6b | vs E1 |
|------|:----------:|:----:|:---:|:---:|:---:|:---:|:-----:|
| E1 | 164 | 11.4% | 128 | 0 | 19 | 17 | — |
| B2 | 159 | 11.0% | 124 | 0 | 19 | 16 | −3.0% |
| **C2** | **159** | **11.0%** | 125 | 0 | 19 | 15 | **−3.0%** |

### 5.3 分析

- **Stage 3c（B2）**：总违规从 164 降到 159（−5），改进微弱。R1 略降（128→124），R6b 减少 1。LLM 改写文本对空间约束类违规帮助有限。
- **Stage 5（C2）**：相比 B2 总违规不变（159），仅 R6b 再减 1（16→15），R1 微增（124→125）。Stage 5 的 LLM judge 重路由在当前设置下几乎无增量效果。
- **结论**：Stage 3c + Stage 5 合计仅减少 5 个违规（164→159，−3.0%），路由层的改进（A0→E1，−21.2%）贡献了绝大部分收益。

---

## 6. Phase B2' & C2'：Evidence Card 消融

### 6.1 动机

Phase B2/C2 表明 Stage 3c 的 LLM 生成对违规降低效果有限（−3.0%）。分析发现，LLM 在生成文本时缺乏对 cited token 空间属性的结构化总结——即使 top-k token 全在右侧，LLM 也可能生成 "bilateral" 或 "left" 等不匹配的 laterality 描述。

**Evidence Card** 是 cited token 的结构化空间摘要，包含：
- `left_count / right_count / cross_count`：laterality 分布
- `dominant_side`：基于 same_side_ratio 的主导侧（left / right / mixed / unknown）
- `laterality_allowed`：基于阈值（0.8）告诉 LLM 允许使用的 laterality 术语（left / right / bilateral / none）
- `level_histogram`：depth 层分布
- `in_range_count / out_of_range_count`：在期望层范围内外的 token 计数

Evidence Card 同时注入 Stage 3c 生成提示和 Stage 5 judge 提示，作为额外的 lexical gate 约束。

### 6.2 设置

| 条件 | 路由 | Stage 3c | Stage 5 | Evidence Card | 说明 |
|------|------|:--------:|:-------:|:-------------:|------|
| B2 | filter+rerank | ON | OFF | — | 基线 Stage 3c |
| C2 | filter+rerank | ON | ON | — | 基线 Stage 3c + Stage 5 |
| B2' | filter+rerank | **ON** | OFF | **ON** | Stage 3c + Evidence Card |
| C2' | filter+rerank | **ON** | **ON** | **ON** | Stage 3c + Stage 5 + Evidence Card |

### 6.3 结果

| 条件 | Violations | Rate | R1 | R2 | R3 | R6b | R6a | vs B2/C2 |
|------|:----------:|:----:|:---:|:---:|:---:|:---:|:---:|:--------:|
| B2 | 159 | 11.0% | 124 | 0 | 19 | 16 | — | — |
| **B2'** | **132** | **9.2%** | **102** | 0 | 19 | 10 | 1 | **−17.0%** |
| C2 | 159 | 11.0% | 125 | 0 | 19 | 15 | — | — |
| **C2'** | **129** | **9.0%** | **97** | 0 | 19 | 12 | 1 | **−18.9%** |

### 6.4 分析

**Finding 1：Evidence Card 是单项改进最大的组件。**
- B2→B2'：违规从 159 降到 132（**−17.0%**），其中 R1 从 124 降到 102（**−17.7%**）。
- C2→C2'：违规从 159 降到 129（**−18.9%**），R1 从 125 降到 97（**−22.4%**）。
- 这是迄今为止单一改动带来的最大降幅。

**Finding 2：`laterality_allowed` 约束直接遏制 R1 违规。**
- Evidence Card 的 `laterality_allowed` 字段告诉 LLM 只允许使用何种 laterality 术语。当 same_side_ratio < 0.8 时，`laterality_allowed = "none"`，LLM 不应输出任何 laterality 词汇。
- R1 从 124→102（B2'）和 125→97（C2'），累计减少 22–28 个违规。

**Finding 3：R6b 附带下降。**
- R6b 从 16→10（B2'）和 15→12（C2'）。Evidence Card 让 LLM 生成更保守的文本，减少了 cross-sentence presence 冲突。

**Finding 4：Stage 5 judge 在 Evidence Card 存在时有更好表现。**
- B2'→C2'：再减 3 个违规（132→129），主要来自 R1 进一步下降（102→97）。Stage 5 judge 在有 Evidence Card 辅助时，更准确地识别和纠正剩余的 laterality 违规。

---

## 7. Phase D2：Repair Executor 消融

### 7.1 动机

C2' 的 Stage 5 使用通用的 log-smooth penalty + reroute + regenerate 流程。我们实现了 **action-dispatched repair executor**，根据 LLM judge 的 `suggested_action` 字段执行针对性修复：

| Action | 修复方式 | 目标规则 |
|--------|---------|---------|
| `drop_laterality` | 正则删除 laterality 词汇 | R1 |
| `drop_depth` | 正则删除 depth 词汇 | R3 |
| `reroute_same_side` | 过滤到 dominant-side token → re-top-k → 重生成 | R1 |
| `merge_conflict` | 保留（未实现） | R6 |
| fallback | 通用 log-smooth penalty → reroute → regenerate → de-specify | 所有 |

### 7.2 结果

| 条件 | Violations | Rate | R1 | R2 | R3 | R6b | R6a | vs C2' |
|------|:----------:|:----:|:---:|:---:|:---:|:---:|:---:|:------:|
| C2' | 129 | 9.0% | 97 | 0 | 19 | 12 | 1 | — |
| **D2** | **132** | **9.2%** | **99** | 0 | 19 | 12 | 2 | **+2.3%** |

### 7.3 分析

**D2 未能改善，略有退化。**

原因分析：Llama-3.1-8B 的 LLM judge 几乎不输出有效的 `suggested_action` 字段。在 180 个 case（约 1440 个 sentence）中，仅触发了 **1 次 `drop_depth`** 操作。其余所有确认的违规都直接 fallback 到通用 log-smooth penalty 路径。

根本原因：8B 模型的结构化 JSON 输出能力不足以可靠地生成 `suggested_action` 字段。judge prompt 要求 JSON 格式输出 `{rule_id, confirmed, severity, suggested_action, offending_span, reasoning}`，但模型倾向于省略或输出无效的 action 值。

**结论**：Repair executor 的设计思路正确，但需要更强的 LLM（如 70B 或 API 模型）才能可靠输出 `suggested_action`。在当前 8B 模型下，C2'（通用 reroute）是更好的选择。

---

## 8. 全局汇总

### 8.1 完整实验结果表

| 条件 | W_proj | Routing | S3c | S5 | EvCard | Repair | Total | R1 | R2 | R3 | R6b | R6a | Rate |
|------|--------|---------|:---:|:--:|:------:|:------:|:-----:|:---:|:---:|:---:|:---:|:---:|:----:|
| A0 | identity | spatial | — | — | — | — | 208 | 100 | 0 | 91 | 17 | — | 14.4% |
| A1 | trained | spatial | — | — | — | — | 189 | 92 | 0 | 80 | 17 | — | 13.1% |
| A2 | identity | semantic | — | — | — | — | 344 | 114 | 87 | 126 | 17 | — | 23.9% |
| A3 | trained | semantic | — | — | — | — | 292 | 118 | 58 | 99 | 17 | — | 20.3% |
| E1 | trained | filter+rerank | — | — | — | — | 164 | 128 | 0 | 19 | 17 | — | 11.4% |
| B2 | trained | filter+rerank | ON | — | — | — | 159 | 124 | 0 | 19 | 16 | — | 11.0% |
| C2 | trained | filter+rerank | ON | ON | — | — | 159 | 125 | 0 | 19 | 15 | — | 11.0% |
| B2' | trained | filter+rerank | ON | — | ON | — | 132 | 102 | 0 | 19 | 10 | 1 | 9.2% |
| **C2'** | trained | filter+rerank | ON | ON | **ON** | — | **129** | **97** | 0 | 19 | 12 | 1 | **9.0%** |
| D2 | trained | filter+rerank | ON | ON | ON | ON | 132 | 99 | 0 | 19 | 12 | 2 | 9.2% |

### 8.2 违规率演进

| 改进步骤 | Violations | Rate | 累计变化 |
|----------|:----------:|:----:|:--------:|
| A0 (baseline: identity + spatial) | 208 | 14.4% | — |
| → A1 (+ trained W_proj) | 189 | 13.1% | −9.1% |
| → E1 (+ filter+rerank routing) | 164 | 11.4% | −21.2% |
| → B2 (+ Stage 3c) | 159 | 11.0% | −23.6% |
| → B2' (+ Evidence Card) | 132 | 9.2% | −36.5% |
| → C2' (+ Stage 5 judge) | 129 | 9.0% | **−38.0%** |

### 8.3 各组件贡献度分析

| 改进组件 | 违规减少量 | 占总减少的比例 |
|---------|:---------:|:-------------:|
| W_proj 训练 (A0→A1) | −19 | 24.1% |
| Filter+rerank routing (A1→E1) | −25 | 31.6% |
| Stage 3c LLM 生成 (E1→B2) | −5 | 6.3% |
| Evidence Card (B2→B2') | −27 | **34.2%** |
| Stage 5 judge (B2'→C2') | −3 | 3.8% |
| **总计 (A0→C2')** | **−79** | **100%** |

### 8.4 关键结论

1. **Evidence Card 是贡献最大的单一组件**：B2→B2' 减少 27 个违规（34.2%），超过 routing 改进的 25 个（31.6%）。Evidence Card 通过 `laterality_allowed` 约束直接遏制了 R1 违规。
2. **E1 routing 仍是基础性改进**：spatial filter + semantic rerank 消除了 R2 违规、大幅降低 R3，是后续所有改进的必要前提。
3. **R1 (laterality) 仍是主要瓶颈**：占最终违规的 97/129 = 75.2%。从 A0 的 100 降到 C2' 的 97，仅净减 3，因为 E1 routing 引入了 +28 个 R1，后被 Evidence Card 大幅抵消。
4. **Stage 3c / Stage 5 的直接增量有限**：不含 Evidence Card 时仅减少 5 个违规（−3.0%）。它们的价值在于为 Evidence Card 提供 LLM 生成/验证的载体。
5. **Repair executor 需更强 LLM 支撑**：D2（action-dispatched repair）在 8B 模型下未能超越 C2'（通用 reroute），因为 judge 几乎不输出有效的 `suggested_action`。
6. **R6b 有所改善**：从结构性常量 17 降到 10–12，Evidence Card 让 LLM 生成更保守的文本。
7. **总体改善**：从 A0 的 208 违规（14.4%）降到 C2' 的 129 违规（9.0%），**累计降低 38.0%**。

### 8.5 剩余违规分布（C2' 最终配置）

```
R1_LATERALITY:   97 / 129  (75.2%)
R3_DEPTH:        19 / 129  (14.7%)
R6b_CROSS:       12 / 129  ( 9.3%)
R6a_CROSS:        1 / 129  ( 0.8%)
R2_ANATOMY:       0 / 129  ( 0.0%)
```

### 8.6 最优配置（C2'）

```
路由:        E1 (spatial filter + semantic rerank, k=8)
W_proj:      trained (contrastive alignment)
Stage 3c:    Llama-3.1-8B-Instruct, temperature=0.3, max_tokens=256
Stage 5:     Llama-3.1-8B-Instruct, α=0.5, log-smooth penalty (γ=2.0)
Evidence Card: laterality_allowed threshold = 0.8
```

---

## 9. 附录：k sweep 详细数据

| k | Total | R1 | R2 | R3 | R6b | Rate |
|---|:-----:|:---:|:---:|:---:|:---:|:----:|
| 1 | 151 | 132 | 0 | 2 | 17 | 10.5% |
| 2 | 206 | 187 | 0 | 2 | 17 | 14.3% |
| 4 | 181 | 157 | 0 | 7 | 17 | 12.6% |
| 6 | 175 | 144 | 0 | 14 | 17 | 12.2% |
| 8 | 164 | 128 | 0 | 19 | 17 | 11.4% |
| 12 | 220 | 168 | 0 | 35 | 17 | 15.3% |
| 14 | 226 | 165 | 0 | 44 | 17 | 15.7% |
