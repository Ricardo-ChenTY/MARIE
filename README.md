# ProveTok: 可验证的 Token 级空间锚定 3D CT 报告生成系统

> **ProveTok** (Provable Evidence Token) 是一个六阶段流水线系统，用于从 3D 胸部 CT 体积数据生成放射学报告。系统的核心创新在于：每一句报告文本都**显式引用**一组来自 CT 体积的空间证据 token，并通过规则验证器检查文本声明与空间证据之间的一致性。与传统端到端报告生成模型（如 R2Gen、CT2Rep）不同，ProveTok 提供了一条完整的**可审计链**——从 3D 子体积到文本声明，放射科医生可以逐句检查每个 token 的空间位置是否支持相应的描述。

---

## 目录

- [1. 研究动机与问题定义](#1-研究动机与问题定义)
- [2. 系统架构总览](#2-系统架构总览)
- [3. 各阶段详细设计](#3-各阶段详细设计)
- [4. W_proj 投影矩阵训练 (InfoNCE)](#4-w_proj-投影矩阵训练-infonce)
- [5. 数据集](#5-数据集)
- [6. 实验设计：7 配置消融链](#6-实验设计7-配置消融链)
- [7. 实验结果](#7-实验结果)
- [8. 项目结构](#8-项目结构)
- [9. 安装与环境配置](#9-安装与环境配置)
- [10. 使用指南](#10-使用指南)
- [11. 配置参数手册](#11-配置参数手册)
- [12. 输出格式](#12-输出格式)

---

## 1. 研究动机与问题定义

### 1.1 现有方法的局限

自动放射学报告生成近年来取得了显著进展（R2Gen, CT2Rep, CT-AGRG, MedVInT），但大多数系统共享一个根本性缺陷：**生成的文本不可验证**。当模型输出"左下叶可见实变影"时，我们无法回答：

1. 模型是否真的"看了"左下叶对应的空间区域？
2. 被引用的空间区域是否在解剖学上与"左下叶"一致？
3. "实变影"的判断是基于图像特征还是语言模型的先验偏差？

这种不透明性在临床场景中尤为危险——放射科医生无法信任也无法高效审核一个黑箱系统的输出。

### 1.2 ProveTok 的设计目标

ProveTok 的核心理念是**空间可审计性 (Spatial Accountability)**：

- **每句话都有引用**：每个报告句子引用 top-k 个 3D 证据 token，每个 token 有明确的 3D 包围盒
- **每个引用都可验证**：6 条规则（R1–R6）检查文本声明与 token 空间属性的一致性
- **违规可修复**：LLM 裁判确认真违规后，修复执行器通过文本改写或 token 重路由消除不一致

### 1.3 形式化定义

给定一个 3D CT 体积 V ∈ ℝ^{D×H×W} 和对应的放射学报告 R = {s₁, s₂, …, sₘ}，ProveTok 的目标是：

1. 构建一个**证据 token 库** T = {t₁, …, t_B}，每个 tᵢ = (bbox_i, f_i, ℓ_i)
2. 为每个句子 sⱼ 选择一组**引用 token** Cⱼ ⊆ T，|Cⱼ| = k
3. 验证 Verify(sⱼ, Cⱼ) = pass 对所有规则 r ∈ {R1, …, R6} 成立
4. 最小化**违规率** ViolRate = Σⱼ |violations(sⱼ)| / M

---

## 2. 系统架构总览

```
输入: 3D CT 体积 (NIfTI, 128³ voxels)
  │
  ├─ Stage 0: 伪影评分 ──────────────────── 每体素伪影风险 g_i
  │
  ├─ Stage 1: SwinUNETR 编码 ────────────── 3D 特征图 F ∈ ℝ^{C×D'×H'×W'}
  │
  ├─ Stage 2: 自适应八叉树分词 ──────────── Token 库 {t_1,...,t_B}, 每个 t_i = (bbox, f, ℓ)
  │
  │  报告文本 ──┐
  │             │
  ├─ Stage 3: 跨模态路由 ───────────────── 每句 → top-k token 引用列表
  │    (可选: W_proj 语义投影)              + 路由分数
  │
  ├─ Stage 3c: 证据卡约束生成 ──────────── LLM 根据证据卡空间约束生成句子
  │    (可选: Evidence Card v1/v2)
  │
  ├─ Stage 4: 规则验证器 ──────────────── R1(侧位) R2(解剖IoU) R3(深度)
  │    (R1–R6 空间一致性检查)              R4(体积) R5(否定) R6(句间一致)
  │
  └─ Stage 5: LLM 裁判 + 修复 ─────────── 确认/驳回违规 → 修复执行
     (可选: Judge + Repair Executor)       (drop_laterality / reroute / de-specify)
  │
输出: 带引用的报告 + trace.jsonl 审计日志
```

---

## 3. 各阶段详细设计

### 3.1 Stage 0: 伪影评分与软门控

**文件**: `ProveTok_Main_experiment/stage0_scorer.py`, `ProveTok_Main_experiment/stage0_2.py`

CT 扫描中常见的金属伪影、运动模糊等噪声区域不应作为证据 token。Stage 0 计算每个空间区域的伪影风险分数，并通过软门控机制抑制高伪影区域的分词优先级。

**伪影风险分数 (Eq.1)**:

```
A_i = clip(w_snr · norm(σ/(|μ|+ε)) + w_streak · norm(streak) + w_outlier · norm(outlier), 0, 1)
```

其中 μ_i, σ_i 是区域内体素的均值/标准差，streak_i 检测线性伪影条纹，outlier_i 检测极端体素值。三个权重默认为 w_snr = 0.5, w_streak = 0.3, w_outlier = 0.2。

**软门控 (Eq.2)**:

```
g_i = σ(-k_gate · (A_i - τ_A))
```

其中 σ 是 sigmoid 函数，k_gate = 15.0 控制门控陡峭度，τ_A = 0.85 是伪影阈值。当 A_i > τ_A 时，g_i ≈ 0，该区域被有效屏蔽。

### 3.2 Stage 1: SwinUNETR 3D 特征编码

**文件**: `ProveTok_Main_experiment/stage1_swinunetr_encoder.py`

使用 MONAI 框架的 SwinUNETR 预训练模型（用于医学图像分割的 Swin Transformer + U-Net 混合架构）作为冻结的 3D 特征提取器。

**架构细节**:
- **输入尺寸**: 128 × 128 × 128 voxels（CT 体积会被重采样到此分辨率）
- **backbone**: Swin Transformer 编码器，层级特征提取
- **特征维度**: feature_size = 48，经编码后输出 F ∈ ℝ^{C×D'×H'×W'}
- **冻结模式**: 所有参数冻结（eval() 模式），不参与梯度计算
- **磁盘缓存**: 编码结果按 case_id 缓存到磁盘，避免重复编码

**关键设计选择**: 我们使用编码器的隐藏状态（而非解码器输出），因为编码器特征保留了更丰富的空间信息。解码器的 skip connection 特征在分割任务中有用，但在 token 构建中我们需要的是**语义密度**而非边界信息。

### 3.3 Stage 2: 自适应八叉树分词

**文件**: `ProveTok_Main_experiment/stage2_octree_splitter.py`

这是 ProveTok 的核心创新之一。传统方法对 CT 体积使用固定网格划分（如 8×8×8），导致细粒度结构被粗糙表示、均匀区域浪费 token 预算。我们采用**自适应八叉树 (Adaptive Octree)** 实现"重要区域精细分割、平坦区域保持粗粒度"。

#### 3.3.1 初始化

从初始深度 init_depth = 2 开始（即 2² = 4 每轴，共 4³ = 64 个初始 cell）。每个 cell 是一个轴对齐的 3D 长方体。

#### 3.3.2 重要性评分 (Eq.3–5)

对每个 cell c_i，计算三个分量：

**不确定性分数 H_i (Eq.4)**:

```
H_i = w_ent · Ĥ(f_i) + w_var · Var(f_i) + w_bnd · |∇f_i|
```

- Ĥ(f_i): 通道能量的归一化熵。将特征图在各通道上的绝对值均值视为"能量分布"，计算其熵并除以 log(C) 归一化。直觉：特征在通道间越均匀分布，不确定性越高，越值得细化
- Var(f_i): 特征图的空间方差。高方差意味着区域内异质性大
- |∇f_i|: 特征图的空间梯度幅值（使用 np.gradient 计算三维梯度）。高梯度表示特征急剧变化的边界区域

权重默认为 w_ent = 0.5, w_var = 0.3, w_bnd = 0.2。

**语义先验分数 P_i (Eq.3)**:

```
P_i = mean(|f_i|)
```

即特征图绝对值的全局均值。高 P_i 表示该区域的 SwinUNETR 特征激活强，可能包含重要的解剖结构或病变。

**同层分位数标准化 (Eq.4)**: H_i 和 P_i 在**同一八叉树层级**内做分位数排名（quantile rank），映射到 [0, 1]。这确保不同层级的 cell 之间可以公平比较。

**综合重要性分数 (Eq.5)**:

```
S_i = g_i · (λ_H · Q(H_i) + (1 - λ_H) · Q(P_i))
```

其中 g_i 是 Stage 0 的软门控，λ_H = 0.6 权衡不确定性和语义先验。

#### 3.3.3 自适应分裂循环

```
while |leaves| < token_budget_B:
    1. 按 S_i 降序排列所有叶节点
    2. 选择得分最高的可分裂节点 c*
       (条件: level < max_depth, voxel_count ≥ min_voxels, S > threshold)
    3. 将 c* 八等分为 8 个子节点
    4. 重新计算所有叶节点的 S_i（因为同层分位数排名会变化）
    5. 重复直到预算耗尽或无可分裂节点
```

- max_depth = 5: 最深分裂到 2⁵ = 32 每轴，但通常远不会到这一步
- min_voxels_to_split = 64: 小于 64 个体素的 cell 不再分裂
- token_budget_B = 128: 默认生成 128 个证据 token

#### 3.3.4 后处理

**非极大值抑制 (NMS)**: 按 S_i 降序，贪心保留 IoU < threshold 的 token，去除空间上高度重叠的冗余 token。默认 nms_iou_threshold = 1.0（即不做 NMS）。

**边界特征融合 (Eq.6–7)**:

```
b_i = mean_{j ∈ N(i)} f_j
f_i^export = (1 - β) · f_i + β · b_i
```

其中 N(i) 是 t_i 的面邻居（face-neighbors，共享一个面的相邻 cell），β = 0.1。这一步平滑了 cell 边界处的特征不连续性。

**稳定排序**: 最终 token 按 (level, z, y, x) 排序分配 token_id，确保相同体积的不同运行产生一致的 token 编号。

#### 3.3.5 输出

每个 token t_i 包含：
- `token_id`: 全局唯一的整数 ID
- `bbox`: BBox3D(x_min, x_max, y_min, y_max, z_min, z_max) — 体素坐标系下的 3D 包围盒
- `level`: 八叉树深度（越深越精细）
- `feature`: 均值池化后的特征向量 f_i ∈ ℝ^{d_v}
- `split_score`: 重要性分数 S_i
- `metadata`: 包含 a_i, h_i, p_i, g_i 等中间计算结果

### 3.4 Stage 3: 跨模态路由

**文件**: `ProveTok_Main_experiment/stage3_router.py`

路由的核心任务：对每个报告句子 s_j，从 token 库 T 中选出最相关的 top-k 个 token 作为引用。

#### 3.4.1 句子规划

**文件**: `ProveTok_Main_experiment/simple_modules.py`

在路由之前，ReportSentencePlanner 对每个句子提取：

- **anatomy_keyword**: 通过关键词匹配提取解剖区域（如 "left lower lobe", "mediastinum", "right lung"）
- **expected_level_range**: 根据句子类型推断预期的八叉树深度范围
  - 弥漫性描述（"diffuse", "bilateral", "edema"）→ 粗层级 [0, 2]
  - 局灶性描述（"nodule", "mass", "lesion"）→ 细层级 [2, 5]
- **is_negated**: 检测否定词（"no", "without", "not", "未见", "无"）
- **expected_volume_range**: 根据大小描述推断体积范围

**解剖区域解析**: RuleBasedAnatomyResolver 维护一个预定义的解剖区域 3D 包围盒字典（归一化坐标）：

| 解剖区域 | 包围盒 (x_min, x_max, y_min, y_max, z_min, z_max) |
|----------|---------------------------------------------------|
| right lung | (0.00, 0.50, 0.00, 1.00, 0.00, 1.00) |
| left lung | (0.50, 1.00, 0.00, 1.00, 0.00, 1.00) |
| right upper lobe | (0.00, 0.50, 0.50, 1.00, 0.00, 1.00) |
| right lower lobe | (0.00, 0.50, 0.00, 0.50, 0.00, 1.00) |
| left upper lobe | (0.50, 1.00, 0.50, 1.00, 0.00, 1.00) |
| left lower lobe | (0.50, 1.00, 0.00, 0.50, 0.00, 1.00) |
| mediastinum | (0.30, 0.70, 0.15, 0.85, 0.00, 1.00) |
| bilateral | (0.00, 1.00, 0.00, 1.00, 0.00, 1.00) |

在使用时，这些归一化坐标乘以实际体积尺寸 (W, H, D) 得到体素坐标。

#### 3.4.2 路由评分

**文本编码**: 使用 sentence-transformers/all-MiniLM-L6-v2（384 维）编码句子得到查询向量 q_s。也支持确定性哈希编码器（256 维，用于无 GPU 环境）。

**投影 (Eq.9)**: Token 特征 f_i 经投影矩阵变换到文本空间：

```
v_i = normalize(W_proj · f_i)
```

其中 W_proj ∈ ℝ^{d_q × d_v}。未训练时 W_proj = I（单位矩阵，前 min(d_q, d_v) 维）。

**路由分数 (Eq.10–11)**:

```
score(s, t_i) = cos(q_s, v_i) + λ_spatial · IoU(bbox_i, bbox_anatomy)
```

λ_spatial = 0.3 权衡语义相似度和空间先验。

#### 3.4.3 两种路由模式

**模式 A: 解剖-空间路由 (A0/A1 配置)**

当 anatomy_spatial_routing = True 时使用 IoU 主导模式：

```
score(s, t_i) = IoU(bbox_i, bbox_anatomy) + ε · cos(q_s, v_i)    (if IoU > 0)
              = ε · cos(q_s, v_i)                                   (otherwise)
```

其中 ε = 0.05 是小的 tiebreaker。**设计动机**：当 W_proj 未训练时（A0）或刚训练时（A1），跨模态 cosine 相似度不可靠，空间 IoU 是唯一可信的信号。

**模式 E: 空间过滤 + 语义重排 (E1+ 配置)**

当 spatial_filter_semantic_rerank = True 时使用两阶段路由：

1. **硬过滤**: 只保留满足以下条件的 token：
   - IoU(t_i, bbox_anatomy) ≥ τ_IoU (默认 0.04)
   - ℓ_i ∈ [ℓ_lo, ℓ_hi]（预期深度范围内）
   - 侧位一致（如果句子声明 "left"，排除中线右侧的 token）
2. **语义重排**: 在通过过滤的候选集内，按 W_proj cosine 相似度排序

分数映射确保**通过过滤的 token 一定排在未通过的前面**：
- 通过: 1.0 + (cos + 1) / 2 ∈ [1, 2]
- 未通过: (cos + 1) / 2 ∈ [0, 1)

**侧位硬过滤**: 当句子声明 "left" 或 "right" 时，根据 token 中心 x 坐标相对于体积中线 x_mid 的位置，排除对侧 token。跨中线 token（在 lateral_tolerance 范围内）始终保留。

### 3.5 Stage 3c: 证据卡约束生成

**文件**: `ProveTok_Main_experiment/stage3c_generator.py`, `ProveTok_Main_experiment/evidence_card.py`

当启用 LLM 生成时，系统不直接使用原始报告句子，而是让 LLM 根据被引用 token 的空间元数据**重新生成**句子。关键约束通过**证据卡 (Evidence Card)** 传递。

#### 3.5.1 证据卡构建

build_evidence_card() 分析 top-k cited token 的空间分布：

**侧位分析**:
- 对每个 token，根据其 bbox.center().x 与中线 x_mid = W/2 的关系分类为 "left"、"right" 或 "cross"
- 计算 same_side_ratio = max(left_count, right_count) / (left_count + right_count)
- 确定 dominant_side：
  - SSR ≥ 0.8 → "left" 或 "right"
  - 双侧各 ≥ 30% → "bilateral"
  - 否则 → "mixed"

**深度分析**:
- 构建 level histogram（各层级 token 数量）
- 如果有预期层级范围，计算 in_range / out_of_range 比例
- ≥ 75% in-range → 允许深度描述

#### 3.5.2 约束门控

证据卡的核心输出是两个约束：

**laterality_allowed**: 告诉 LLM 可以使用什么侧位词
- "left" / "right": 证据明确支持单侧 → LLM 可以说"左侧"/"右侧"
- "bilateral": 证据支持双侧 → LLM 可以说"双侧"
- "none": 证据不足或混合 → **禁止**使用任何侧位词

**depth_allowed**: 告诉 LLM 可否使用深度词
- "in_range" / "unconstrained": → LLM 可以说"上叶"/"下叶"/"基底"等
- "none": → **禁止**使用深度相关词

#### 3.5.3 两版本对比

| 特性 | Evidence Card v1 (B2') | Evidence Card v2 (B2'v2) |
|------|----------------------|------------------------|
| 侧位门控阈值 | SSR ≥ 0.8 | SSR ≥ 0.9 |
| 最少非中线 token | 无要求 | ≥ 2 个 |
| bilateral 条件 | 双侧有 token | 双侧各 ≥ 2 个 |
| 效果 | R1=109, R3=86 | R1=98, R3=23 |
| 配置标志 | strict_laterality=False | strict_laterality=True |

v2 的更严格门控使 R3（深度违规）从 86 降至 23（**-73.3%**），代价仅为 R1 少量增加。

#### 3.5.4 Token 清洗

在 v2 模式下，证据卡构建后还会执行两步清洗（在 stage0_4_runner.py 中）：

1. **侧位清洗**: 如果 dominant_side 明确（"left" 或 "right"），移除对侧 token，只保留同侧 + 跨中线 token（至少保留 2 个）
2. **深度清洗**: 如果有预期层级范围，移除 out-of-range token（至少保留 2 个）
3. 用清洗后的 token 集**重建**证据卡

#### 3.5.5 LLM 调用

**System Prompt**: 角色设定为专家放射科医生，明确约束规则：
- 只有当 laterality_allowed 为 "left"/"right"/"bilateral" 时才能使用侧位词
- 只有当 depth_allowed 为 "in_range" 或 "unconstrained" 时才能使用深度词

**User Prompt** 包含：
1. 已生成的历史句子（保持一致性）
2. 原始 topic 句子
3. Token 上下文（ID, level, center_xyz, volume）
4. 证据卡摘要

**支持的 LLM 后端**:

| 后端 | 模型示例 | 配置方式 |
|------|---------|---------|
| Ollama | qwen2.5:7b | 本地 HTTP |
| HuggingFace | Qwen/Qwen2.5-7B-Instruct | 本地 pipeline |
| OpenAI | gpt-4o-mini | API |
| Anthropic | claude-haiku-4-5-20251001 | API |

### 3.6 Stage 4: 规则验证器 (R1–R6)

**文件**: `ProveTok_Main_experiment/stage4_verifier.py`

验证器对**每一个**已生成（或原始）的句子执行 6 条规则检查。每条规则独立产生 RuleViolation，包含 severity（严重度）、message（说明）和 offending token_ids。

#### R1: 侧位一致性 (R1_LATERALITY)

**检查**: 如果句子文本声明 "left" 或 "right"，被引用的 token 中**同侧 token 的比例**必须 ≥ 阈值。

**算法**:
1. 用正则表达式 `\b(left|lt)\b` 和 `\b(right|rt)\b` 解析文本中的侧位声明
2. 对每个 cited token，根据 bbox.center().x 相对于中线分类为 left/right/cross
3. 计算 same_ratio = same_count / (same_count + opposite_count)（cross 不计入）
4. 如果 same_ratio < r1_min_same_side_ratio（默认 0.6），触发 R1 违规

**例外处理**:
- bilateral/both 声明不触发 R1
- 中线器官（如 "mediastinum"，在 r1_skip_midline_keywords 中）跳过 R1
- 否定句可选跳过（r1_negation_exempt）

**severity**: 1.0（最高严重度，因为侧位错误是临床最危险的）

#### R2: 解剖 IoU 一致性 (R2_ANATOMY)

**检查**: 被引用 token 的包围盒必须与句子声明的解剖区域有足够的空间重叠。

**算法** (默认模式):
1. 从 RuleBasedAnatomyResolver 获取 anatomy_bbox
2. 对每个 cited token 计算 IoU(t_i.bbox, anatomy_bbox)
3. 统计 IoU ≥ τ_IoU（默认 0.04）的 token 比例 good_ratio
4. 如果 good_ratio < r2_min_support_ratio（默认 0.8），触发 R2 违规

**severity**: 根据 IoU gap 和 ratio gap 的最大值动态计算：sev = 0.8 × max(mean_gap, ratio_gap)

**max IoU 模式** (use_max_iou_for_r2 = True): 只要任何一个 token 的 IoU ≥ τ 即通过。更宽松但减少误报。

#### R3: 深度一致性 (R3_DEPTH)

**检查**: 被引用 token 的八叉树层级必须在预期范围内。

**算法**:
1. 从 SentencePlan.expected_level_range 获取 [ℓ_lo, ℓ_hi]
2. 如果有任何 token 的 level 不在此范围内，触发 R3 违规

**severity**: 0.7

**直觉**: 描述弥漫性病变（如 "bilateral pleural effusion"）应该由粗层级 token（level 0–2，覆盖大区域）支持；描述局灶性病变（如 "subcentimeter nodule"）应该由细层级 token（level 2–5，小区域）支持。

#### R4: 体积一致性 (R4_SIZE) — 默认禁用

**检查**: cited token 的 union bbox 体积应在预期范围内。默认禁用 (r4_disabled = True)，因为 CT 体积中的病变大小变异极大。

#### R5: 否定一致性 (R5_NEGATION)

**检查**: 否定句（"no consolidation", "未见结节"）不应引用有阳性发现特征的 token。

**算法**:
1. 检测句子是否含否定词（"no", "without", "not", "未见", "无"）
2. 检查 cited token 的 metadata["negation_conflict"] 分数
3. **词汇回退**: 如果 token 没有否定冲突标记，但句子同时包含否定词和阳性发现词（"nodule", "consolidation", "effusion" 等 13 个词），触发 R5 违规

**severity**: 0.5（回退模式的严重度较低）

#### R6a: 句间侧位冲突 (R6a_CROSS_LATERALITY)

**检查**: 描述同一解剖区域的不同句子不应声明矛盾的侧位。

**算法**: 按 anatomy_keyword 分组所有句子，在每组内检查是否同时有 "left" 和 "right" 声明。severity = 0.7。

#### R6b: 句间存在性冲突 (R6b_CROSS_PRESENCE)

**检查**: 描述同一解剖区域的不同句子不应一句肯定一句否定同一发现。

**算法**: 按 anatomy_keyword 分组，识别阳性句和阴性句，两者同时存在即触发。severity = 0.8。

### 3.7 Stage 5: LLM 裁判与修复执行器

**文件**: `ProveTok_Main_experiment/stage5_llm_judge.py`

Stage 4 的规则验证器可能产生**误报**（false positive violation）。Stage 5 使用 LLM 对每个违规进行二次确认，并对确认的违规执行修复。

#### 3.7.1 LLM 裁判

对每个 Stage-4 违规，构建一个 prompt 包含：
- 报告句子原文
- 违规规则 ID 和详情
- 证据卡摘要（token 空间分布）

LLM 输出结构化 JSON：

```json
{
  "confirmed": true,
  "severity": 0.8,
  "suggested_action": "drop_laterality",
  "offending_span": "left",
  "reasoning": "The sentence mentions left but evidence shows mixed laterality."
}
```

**容错**: 如果 JSON 解析失败，用正则回退匹配 "confirmed": true。如果 LLM 调用本身失败：
- fail_open = True（默认）: 保守策略，保留原始违规
- fail_open = False: 乐观策略，驳回违规

#### 3.7.2 修复执行器

对确认的违规，按优先级执行修复动作：

**1. drop_laterality**: 用正则移除侧位词（"left", "right", "bilateral" 等）

**2. drop_depth**: 用正则移除深度词（"upper", "lower", "apical", "basal" 等）

**3. reroute_same_side**: 过滤 token 到 dominant side → 重新 top-k → LLM 重新生成

**4. 通用回退 (log-smooth penalty)**:

```
r'_i = r_i - γ · ln(1 + sev_i)
```

每个 token 被惩罚的程度与其参与的违规严重度成对数关系。γ = 2.0 控制惩罚强度。重新排序后取 top-k，LLM 重新生成。

如果重新生成后仍然违规 → **de-specify 回退**：移除所有空间定位词后再生成一次。

#### 3.7.3 实际效果

在 5K 测试集上，Stage 5 的效果是**矛盾的**：
- **积极面**: Citation Faithfulness 从 77.6% 提升到 81.1%（+3.5pp）
- **消极面**: Violation Rate 从 4.19% 回升到 6.00%（+1.81pp）

原因：reroute 替换了 B2'v2 中已经最优的 token，引入新的 R1/R3 违规。D2 的修复率仅 11.8%（28/238 修复，20 新引入）。

---

## 4. W_proj 投影矩阵训练 (InfoNCE)

**文件**: `train_wprojection.py`

W_proj ∈ ℝ^{d_q × d_v} 是一个线性投影矩阵，将 token 特征 f_i（SwinUNETR 空间，d_v 维）映射到文本查询空间（sentence-transformers 空间，d_q 维）。训练后，cos(q_s, W_proj · f_i) 能更准确地反映文本-token 的语义相关性。

### 4.1 伪标签提取

**核心思想**: 不需要人工标注，利用 A0 pipeline 的空间 IoU routing 作为伪标签。

**步骤**:
1. 在训练集上跑 A0 pipeline（identity W_proj, anatomy-spatial routing）
2. 每个句子在 trace.jsonl 中记录了 topk_token_ids — 这 8 个 token 是纯空间 IoU 选出的
3. 这些 token ID 成为该句子的**正样本集** P_s

**合理性**: A0 的空间路由虽然不用语义信息，但 IoU 选出的 token 在空间上确实覆盖了正确的解剖区域。InfoNCE 训练的目标是让 W_proj 学会"哪些 token 特征与哪些文本查询应该接近"，这些伪标签提供了足够好的监督信号。

### 4.2 正负样本构造

对于 mini-batch 中的每个句子 s：

**查询向量** q_s: 句子文本经 all-MiniLM-L6-v2 编码得到 384 维向量

**正样本** P_s: 从该句子对应 case 的 tokens.pt 中提取 topk_token_ids 对应的 token 特征向量，取均值得到 f̄_s⁺ = (1/k) Σ_{i∈P_s} f_i

**负样本** N_s: **batch 内负采样** — mini-batch 中其他 N-1 个句子的 f̄⁺ 都是 s 的负样本。

| 参数 | 值 |
|------|-----|
| batch size N | 32 |
| 候选集大小 |C_s| | N = 32（1 正 + 31 负） |
| 正样本 token 数 k | 8 |

### 4.3 多正样本 InfoNCE 损失

实际实现使用均值正样本的标准 InfoNCE（等价于 NT-Xent）：

```
L = -log [ exp(cos(q_s, W_proj · f̄_s⁺) / τ) / Σ_{j=1}^{N} exp(cos(q_s, W_proj · f̄_j⁺) / τ) ]
```

其中 τ = 0.07 是温度参数。τ 越小，分布越尖锐，对硬负样本的惩罚越强。

**数学等价性**: 分母遍历 batch 内所有 N 个样本（包括自身作为正样本），标签是对角线 labels = [0, 1, …, N-1]，直接使用 cross_entropy loss。

### 4.4 训练细节

| 超参数 | 值 | 说明 |
|--------|-----|------|
| 优化器 | AdamW | weight_decay = 1e-4 |
| 学习率调度 | Cosine Annealing | T_max = epochs |
| 初始学习率 | 1e-3 | — |
| 梯度裁剪 | 1.0 | clip_grad_norm_ |
| 初始化 | 正交初始化 | nn.init.orthogonal_ |
| 早停 | patience = 10 | 在 val loss 上 |
| 最大 epoch | 50 | — |
| 随机种子 | 42 | 可复现 |

**输出**:
- w_proj.pt: PyTorch 张量 [d_q, d_v]
- w_proj.npy: NumPy 数组（备份）
- train_log.json: 训练曲线、超参数、最优 val loss

---

## 5. 数据集

| 数据集 | 来源 | 说明 | 报告类型 |
|--------|------|------|---------|
| **CT-RATE** | HuggingFace ibrahimhamamci/CT-RATE | 大规模胸部 CT 数据集 | 放射学报告 |
| **RadGenome-ChestCT** | HuggingFace RadGenome/RadGenome-ChestCT | 基因组标注的胸部 CT | 结构化发现报告 |

### 5K 数据集划分

从两个数据集各采样 2,500 例，共 5,000 例。按 60/20/20 划分：

| 划分 | CT-RATE | RadGenome | 总计 | 用途 |
|------|---------|-----------|------|------|
| train | 1,500 | 1,500 | 3,000 | W_proj 训练 |
| val | 500 | 500 | 1,000 | W_proj 早停 |
| test | 500 | 500 | 1,000 | 消融实验评估 |

**实际测试集**: 过滤损坏/缺失的 NIfTI 后，测试集为 **496 例**（3,979 句子）。

---

## 6. 实验设计：7 配置消融链

我们设计了一条渐进式消融链，每一步只增加一个组件，以隔离每个组件的贡献：

| 配置 | 简称 | 新增组件 | 路由模式 | Evidence Card | LLM Judge | 修复 |
|------|------|---------|---------|---------------|-----------|------|
| **A0** | Identity W + Spatial | 基线 | 解剖-空间 (IoU主导) | — | — | — |
| **A1** | Trained W + Spatial | 训练的 W_proj | 解剖-空间 (IoU主导) | — | — | — |
| **E1** | Spatial filter + Rerank | E1 路由模式 | 硬过滤 + 语义重排 | — | — | — |
| **B2'** | + Evidence Card v1 | v1 证据卡生成 | 同 E1 | v1 (SSR≥0.8) | — | — |
| **B2'v2** | + Evidence Card v2 | v2 严格门控 | 同 E1 | v2 (SSR≥0.9) | — | — |
| **C2'** | + LLM Judge | Stage 5 裁判 | 同 E1 | v1 | Qwen-2.5-7B | — |
| **D2** | + Repair Executor | 完整修复链 | 同 E1 | v1 | Qwen-2.5-7B | 全修复 |

**通用参数** (所有配置共享): B = 128, k = 8, τ_IoU = 0.04, λ_spatial = 0.3

---

## 7. 实验结果

### 7.1 违规率与 Bootstrap 置信区间

使用 patient-level paired bootstrap (R=5,000, α=0.05) 和 Holm step-down 校正：

| 配置 | 违规率 (%) | 95% CI | 总违规数 | Holm p-value | 显著？ |
|------|-----------|--------|---------|-------------|--------|
| A0 | 7.59 | [6.84, 8.35] | 302 | — | — |
| A1 | 5.96 | [5.27, 6.64] | 239 | 0.008 | **是** ↓ |
| E1 | 6.04 | [5.34, 6.77] | 246 | 1.0 | 否 |
| B2' | 5.85 | [5.14, 6.60] | 239 | 1.0 | 否 |
| **B2'v2** | **4.19** | **[3.60, 4.83]** | **169** | **0.006** | **是** ↓ |
| C2' | 6.00 | [5.25, 6.79] | 246 | 0.006 | **是** ↑ 回升 |
| D2 | 5.79 | [5.07, 6.55] | 237 | 1.0 | 否 |

### 7.2 逐规则违规分解

| 配置 | R1 (侧位) | R3 (深度) | R6b (句间) | R2 | R6a | 总计 |
|------|-----------|----------|-----------|-----|-----|------|
| A0 | 0 | 237 | 65 | 0 | 0 | 302 |
| A1 | 0 | 174 | 65 | 0 | 0 | 239 |
| E1 | 95 | 86 | 65 | 0 | 0 | 246 |
| B2' | 109 | 86 | 44 | 0 | 0 | 239 |
| **B2'v2** | **98** | **23** | **48** | **0** | **0** | **169** |
| C2' | 113 | 86 | 47 | 0 | 0 | 246 |
| D2 | 102 | 86 | 49 | 0 | 0 | 237 |

**关键观察**:
- A0→A1: R3 从 237 降至 174（-26.6%），训练的 W_proj 让语义路由更准确
- A1→E1: R3 从 174 降至 86（-50.6%），但 R1 从 0 升至 95。原因：E1 的语义重排可能把对侧 token 排到前面
- B2'→B2'v2: R3 从 86 降至 23（-73.3%），v2 严格门控的核心收益
- R2 和 R6a 在所有配置下都为 0

### 7.3 空间锚定指标

| 配置 | Mean Overlap | Hit@1 | Hit@4 | Hit@8 | Routing Prec. | Lat. Acc. | Cite. Faith. | Depth Faith. |
|------|-------------|-------|-------|-------|---------------|-----------|-------------|-------------|
| A0 | 0.931 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | — | — |
| A1 | 0.931 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | — | — |
| E1 | 0.685 | 76.8 | 96.1 | 99.3 | 75.4 | 97.2 | — | — |
| B2' | 0.685 | 76.8 | 96.1 | 99.3 | 75.4 | 96.9 | 76.9 | 97.3 |
| B2'v2 | 0.689 | 76.8 | 95.9 | 98.2 | 75.8 | 97.1 | 77.6 | 98.4 |
| C2' | 0.685 | 76.8 | 96.1 | 99.3 | 75.4 | 96.7 | 81.1 | 97.3 |
| D2 | 0.685 | 76.8 | 96.1 | 99.3 | 75.4 | 97.0 | 80.6 | 97.3 |

### 7.4 NLG 文本质量

| 配置 | BLEU-4 | ROUGE-L | METEOR |
|------|--------|---------|--------|
| B2' | 0.4707 | 0.6425 | 0.6175 |
| B2'v2 | 0.4725 | 0.6426 | 0.6190 |
| C2' | 0.4771 | 0.6431 | 0.6218 |
| D2 | 0.4738 | 0.6434 | 0.6217 |

NLG 指标在各配置间非常稳定（BLEU-4 差异 < 0.007），说明 Evidence Card 和 Stage 5 修复不影响文本生成质量。

### 7.5 关键发现

1. **最优配置**: B2'v2（Evidence Card v2）以 4.19% 的违规率 [3.60, 4.83] 显著优于所有其他配置

2. **训练 W_proj 有效** (A0→A1): 显著降低 R3 违规（p < 0.01），说明 InfoNCE 训练成功让投影矩阵学到了有意义的跨模态对齐

3. **E1 路由的 trade-off** (A1→E1): 语义重排减少 R3 但引入 R1。这是因为 W_proj cosine 分数可能将语义相关但空间对侧的 token 排高。后续的 Evidence Card v2 通过严格门控缓解了这一问题

4. **Evidence Card v2 是关键创新** (B2'→B2'v2): R3 从 86 降至 23（-73.3%）。SSR ≥ 0.9 + 最少 2 个非中线 token 的双重门控有效阻止了 LLM 生成不支持的深度声明

5. **Stage 5 是一个 trade-off** (B2'v2→C2'): CF 提升 +3.5pp，但 Viol% 回升 +1.81pp。reroute 操作替换了已经最优的 token

6. **D2 修复效果有限**: 仅 11.8% 的修复成功率（28/238 修复，20 新引入），净收益极小

7. **R2 和 R6a 从未违规**: 空间 IoU 路由 + 句间一致性检查在所有配置下都足够可靠

---

## 8. 项目结构

```
ProveTok_ACM/
│
├── ProveTok_Main_experiment/             # 核心流水线模块
│   ├── config.py                         # 所有超参数 dataclass
│   │   ├── SplitConfig                   #   Stage 2 八叉树参数
│   │   ├── RouterConfig                  #   Stage 3 路由参数
│   │   ├── VerifierConfig                #   Stage 4 验证规则参数
│   │   ├── RerouteConfig                 #   Stage 5 重路由参数
│   │   ├── LLMJudgeConfig                #   Stage 5 LLM 裁判参数
│   │   └── ProveTokConfig                #   全局配置容器
│   ├── types.py                          # 核心数据结构
│   │   ├── BBox3D                        #   3D 包围盒 (含 IoU/center/volume)
│   │   ├── EvidenceToken                 #   证据 token (bbox + feature + level)
│   │   ├── SentencePlan                  #   句子规划 (anatomy + level_range)
│   │   ├── SentenceOutput                #   句子输出 (text + citations)
│   │   ├── RuleViolation                 #   规则违规 (rule_id + severity)
│   │   └── SentenceAudit                 #   审计结果 (passed + violations)
│   │
│   ├── stage0_scorer.py                  # Stage 0: 伪影评分器
│   ├── stage0_artifacts.py               # Stage 0: 伪影分量计算
│   ├── stage0_2.py                       # Eq.1-7 数学实现
│   ├── math_utils.py                     # 共享数学函数
│   ├── stage1_swinunetr_encoder.py       # Stage 1: 冻结 SwinUNETR 编码器
│   ├── preprocess.py                     # 体积加载与预处理
│   ├── stage2_octree_splitter.py         # Stage 2: 自适应八叉树分词器
│   ├── text_encoder.py                   # 文本编码器 (hash / sentence-transformers)
│   ├── simple_modules.py                 # 句子规划器 + 解剖区域解析器
│   ├── stage3_router.py                  # Stage 3: 跨模态路由器 (含 W_proj)
│   ├── evidence_card.py                  # 证据卡构建
│   ├── stage3c_generator.py              # Stage 3c: LLM 生成器
│   ├── stage4_verifier.py                # Stage 4: R1-R6 规则验证器
│   ├── stage5_llm_judge.py               # Stage 5: LLM 裁判 + 修复
│   ├── stage0_4_runner.py                # 主编排器 (单 case 全流水线)
│   └── token_bank_io.py                  # Token bank 序列化
│
├── run_mini_experiment.py                # 主 CLI 入口
├── train_wprojection.py                  # W_proj InfoNCE 训练
├── analyze_outputs.py                    # 输出验证 + M5 协议分析
├── validate_stage0_4_outputs.py          # 输出完整性检查
│
├── Scripts/                              # 运行脚本与分析工具
│   ├── run_5k_ablation_chain.sh          #   完整 7 配置消融
│   ├── run_wprojection_train_5k.sh       #   W_proj 5K 训练
│   ├── run_statistical_analysis.py       #   Bootstrap CI + Holm 校正
│   ├── generate_table2_and_figures.py    #   论文表格 + 图
│   ├── evaluate_metrics.py               #   NLG 指标
│   ├── sample_5k_from_hf.py             #   从 HuggingFace 采样
│   ├── split_train_val_test.py           #   数据划分
│   └── ...
│
├── Math_documents/                       # LaTeX 论文源码
├── manifests/                            # 数据集清单 (CSV)
├── outputs/                              # 实验输出
│   ├── 5k_ablation/                      #   7 配置消融结果
│   ├── paper_figures_5k/                 #   论文图表
│   └── wprojection_5k/                   #   W_proj 训练产物
├── checkpoints/                          # 模型权重 (swinunetr.ckpt)
├── environment.yaml                      # Conda 环境
└── requirements.txt                      # Pip 依赖
```

---

## 9. 安装与环境配置

### 方式 A: Conda（推荐）

```bash
conda env create -f environment.yaml
conda activate provetok
```

### 方式 B: Pip

```bash
pip install -r requirements.txt
```

### 核心依赖

| 包 | 版本要求 | 用途 |
|----|---------|------|
| Python | 3.12 | — |
| PyTorch | ≥ 2.0 | 深度学习框架 |
| MONAI | ≥ 1.3 | SwinUNETR 编码器 |
| sentence-transformers | ≥ 3.0 | 文本编码 |
| SimpleITK | — | NIfTI 体积 I/O |
| numpy, pandas, scipy | — | 科学计算 |
| matplotlib, seaborn | — | 可视化 |

### 可选 LLM 后端

```bash
# HuggingFace 本地模型
pip install transformers>=4.40 accelerate

# OpenAI API
pip install openai>=1.0

# Anthropic API
pip install anthropic>=0.20
```

### GPU 要求

| 任务 | 最低 GPU | VRAM |
|------|---------|------|
| SwinUNETR 编码 | 任意 CUDA | ~4 GB |
| W_proj 训练 | CPU 可用 | < 1 GB |
| Stage 3c 生成 (7B) | 任意 CUDA | ~14 GB (bf16) |
| 完整流水线 (含 LLM) | 1× A100 40G | ~18 GB |

---

## 10. 使用指南

### 10.1 数据准备

```bash
# 采样 5K 数据并划分
python Scripts/sample_5k_from_hf.py
python Scripts/split_train_val_test.py

# 下载 NIfTI 体积
bash Scripts/run_5k_download.sh

# 准备 SwinUNETR 权重
mkdir -p checkpoints
# 将 swinunetr.ckpt 放入 checkpoints/
```

### 10.2 运行流水线

```bash
# A0: 基线
python run_mini_experiment.py \
  --ctrate_csv manifests/ctrate_5k_test_manifest.csv \
  --radgenome_csv manifests/radgenome_5k_test_manifest.csv \
  --encoder_ckpt checkpoints/swinunetr.ckpt \
  --out_dir outputs/5k_ablation/A0_identity_spatial \
  --token_budget_b 128 --k_per_sentence 8 \
  --tau_iou 0.04 --anatomy_spatial_routing

# B2'v2: 最优配置
python run_mini_experiment.py \
  --ctrate_csv manifests/ctrate_5k_test_manifest.csv \
  --radgenome_csv manifests/radgenome_5k_test_manifest.csv \
  --encoder_ckpt checkpoints/swinunetr.ckpt \
  --w_proj_path outputs/wprojection_5k/w_proj.pt \
  --out_dir outputs/5k_ablation/B2_evcard_v2 \
  --token_budget_b 128 --k_per_sentence 8 \
  --tau_iou 0.04 --spatial_filter_semantic_rerank \
  --stage3c_backend huggingface --stage3c_model Qwen/Qwen2.5-7B-Instruct \
  --strict_laterality

# D2: 完整流水线
python run_mini_experiment.py \
  --ctrate_csv manifests/ctrate_5k_test_manifest.csv \
  --radgenome_csv manifests/radgenome_5k_test_manifest.csv \
  --encoder_ckpt checkpoints/swinunetr.ckpt \
  --w_proj_path outputs/wprojection_5k/w_proj.pt \
  --out_dir outputs/5k_ablation/D2_repair \
  --token_budget_b 128 --k_per_sentence 8 \
  --tau_iou 0.04 --spatial_filter_semantic_rerank \
  --stage3c_backend huggingface --stage3c_model Qwen/Qwen2.5-7B-Instruct \
  --llm_judge huggingface --llm_judge_model Qwen/Qwen2.5-7B-Instruct
```

### 10.3 训练 W_proj

```bash
# Step 1: 生成伪标签
python run_mini_experiment.py \
  --ctrate_csv manifests/ctrate_5k_train_manifest.csv \
  --radgenome_csv manifests/radgenome_5k_train_manifest.csv \
  --encoder_ckpt checkpoints/swinunetr.ckpt \
  --out_dir outputs/wprojection_5k/traces \
  --token_budget_b 128 --anatomy_spatial_routing

# Step 2: InfoNCE 训练
python train_wprojection.py \
  --cases_dir outputs/wprojection_5k/traces/cases \
  --train_manifest manifests/5k_train_cases.txt \
  --val_manifest manifests/5k_val_cases.txt \
  --out_dir outputs/wprojection_5k \
  --text_encoder semantic \
  --epochs 50 --batch_size 32 --lr 1e-3 --tau 0.07 \
  --patience 10 --device cuda --seed 42
```

### 10.4 分析与评估

```bash
# 验证输出
python analyze_outputs.py \
  --out_dir outputs/5k_ablation/B2_evcard_v2 \
  --expected_cases_map ctrate=248,radgenome=248

# 论文表格
python Scripts/generate_table2_and_figures.py --data_dir outputs/5k_ablation

# 统计显著性分析
python Scripts/run_statistical_analysis.py

# NLG 指标
python Scripts/evaluate_metrics.py \
  --pred_dir outputs/5k_ablation/B2_evcard_v2 \
  --ref_dir outputs/5k_ablation/A0_identity_spatial
```

---

## 11. 配置参数手册

所有超参数定义在 `ProveTok_Main_experiment/config.py`：

### SplitConfig (Stage 2)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| token_budget_b | 64 | 目标 token 数（实验用 128） |
| init_depth | 2 | 初始八叉树深度 |
| max_depth | 5 | 最大递归深度 |
| min_voxels_to_split | 64 | 最小可分裂体素数 |
| nms_iou_threshold | 1.0 | NMS 阈值（1.0=不做） |
| tau_a | 0.85 | 伪影阈值 |
| k_gate | 15.0 | 门控陡峭度 |
| lambda_h | 0.6 | 不确定性权重 |
| beta | 0.1 | 边界融合权重 |

### RouterConfig (Stage 3)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| k_per_sentence | 8 | top-k token 数 |
| lambda_spatial | 0.3 | 空间 IoU 权重 |
| infonce_tau | 0.07 | InfoNCE 温度 |
| anatomy_spatial_routing | False | A0/A1 模式 |
| spatial_filter_semantic_rerank | False | E1+ 模式 |
| anatomy_tiebreak_eps | 0.05 | A0/A1 tiebreaker |

### VerifierConfig (Stage 4)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| tau_anatomy_iou | 0.1 | R2 IoU 阈值（实验用 0.04） |
| r2_min_support_ratio | 1.0 | R2 通过 token 比例 |
| r1_min_same_side_ratio | 1.0 | R1 同侧比例（实验用 0.6） |
| lateral_tolerance | 0.0 | 中线死区宽度 |
| r4_disabled | False | 禁用 R4 |
| r5_fallback_lexicon | True | R5 词汇回退 |

### LLMJudgeConfig (Stage 5)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| backend | "ollama" | LLM 提供者 |
| model | "qwen2.5:7b" | 模型名 |
| alpha | 0.5 | 分数惩罚系数 |
| temperature | 0.0 | 采样温度 |
| fail_open | True | 失败时保守保留违规 |

### RerouteConfig (Stage 5)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| gamma_penalty | 2.0 | Log-smooth 惩罚强度 |
| max_retry | 1 | 最大重生成次数 |

---

## 12. 输出格式

每个 case 生成 trace.jsonl 审计日志：

```
outputs/5k_ablation/B2_evcard_v2/cases/ctrate/case_001/trace.jsonl
```

**第 1 行: Case 元数据**
```json
{
  "type": "case_meta",
  "case_id": "case_001",
  "B": 128, "k": 8,
  "lambda_spatial": 0.3,
  "tau_IoU": 0.04,
  "n_sentences": 7
}
```

**第 2+ 行: 逐句记录**
```json
{
  "type": "sentence",
  "sentence_index": 0,
  "sentence_text": "The heart is normal in size.",
  "original_topic": "The heart is normal in size.",
  "generated": true,
  "anatomy_keyword": "heart",
  "topk_token_ids": [42, 87, 15, 3, 91, 66, 28, 50],
  "topk_scores": [1.92, 1.88, 1.85, 1.83, 1.81, 1.79, 1.77, 1.75],
  "evidence_card": {
    "cited_tokens": 8,
    "dominant_side": "bilateral",
    "laterality_allowed": "bilateral",
    "depth_allowed": "in_range",
    "level_histogram": {"3": 2, "4": 5, "5": 1}
  },
  "violations": [],
  "stage5_judgements": []
}
```

**violations 字段** (有违规时):
```json
"violations": [{
  "sentence_index": 2,
  "rule_id": "R1_LATERALITY",
  "severity": 1.0,
  "message": "Laterality mismatch with claim=left (same_side_ratio=0.375).",
  "token_ids": [14, 22, 35]
}]
```

**stage5_judgements 字段** (Stage 5 启用时):
```json
"stage5_judgements": [{
  "rule_id": "R1_LATERALITY",
  "confirmed": true,
  "adjusted_severity": 0.8,
  "suggested_action": "drop_laterality",
  "offending_span": "left",
  "reasoning": "Evidence shows mixed laterality distribution."
}]
```

所有下游分析脚本（analyze_outputs.py, generate_table2_and_figures.py, run_statistical_analysis.py）都读取 trace.jsonl 作为输入。

---

## 引用

```bibtex
@inproceedings{provetok2026,
  title={ProveTok: Verifiable Token-Grounded 3D CT Report Generation},
  author={[Authors]},
  booktitle={Proceedings of ACM Multimedia},
  year={2026}
}
```
