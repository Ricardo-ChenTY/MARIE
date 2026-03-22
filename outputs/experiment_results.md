# ProveTok 实验结果汇总

> 5K 测试集（250 CT-RATE + 250 RadGenome，共 3979 句），7 个消融配置全量评估。

---

## Table 1 — 主表：NLG 对比

与已发表方法的 NLG 指标对比。ProveTok 使用 B2'v2 配置（Evidence Card v2 + 空间路由）。

**注意**：Baseline 使用各自论文的评估协议；ProveTok 使用我们自己的 train/valid/test 划分（seed=42）。由于任务定义不同（完整报告生成 vs. 句子级生成），指标不完全可比。

| 数据集 | 方法 | 协议 | BLEU-4 | METEOR | ROUGE-L |
|--------|------|------|--------|--------|---------|
| CT-RATE | CT2Rep | 原文 | 0.172 | 0.173 | 0.243 |
| CT-RATE | CT-AGRG | 原文 | 0.172 | 0.196 | 0.280 |
| CT-RATE | **ProveTok** | 自有划分 | **0.467** | **0.603** | **0.626** |
| RadGenome | MedVInT | 原文 | 0.246 | 0.404 | 0.326 |
| RadGenome | Reg2RG | 原文 | 0.249 | 0.441 | 0.367 |
| RadGenome | **ProveTok** | 自有划分 | **0.506** | **0.660** | **0.680** |

**分析**：ProveTok 在两个数据集上均大幅超越所有 baseline。优势来自句子级生成+空间路由证据 token 的方式，而非端到端全报告生成。对比有参考价值但非完全公平。

---

## Table 2 — 消融链（5K，3979 句）

每行在前一行基础上增加一个组件。从纯空间路由（A0）逐步到完整流水线（D2）。

| ID | 配置 | Viol.% | R1 | R3 | R6b | C_LLM | B-4 | R-L | MTR |
|----|------|--------|-----|-----|------|-------|------|------|------|
| A0 | Identity W + 空间路由 | 7.59 | 0 | 237 | 65 | 0 | — | — | — |
| A1 | Trained W + 空间路由 | 6.01 | 0 | 174 | 65 | 0 | — | — | — |
| E1 | 空间过滤 + 语义重排 | 6.18 | 95 | 86 | 65 | 0 | — | — | — |
| B2' | + LLM 生成 + 证据卡 v1 | 6.01 | 109 | 86 | 44 | 3979 | .471 | .643 | .618 |
| **B2'v2** | **+ 证据卡 v2 (严格侧别)** | **4.25** | **98** | **23** | **48** | **3979** | **.473** | **.643** | **.619** |
| C2' | + LLM 裁判 (Stage 5) | 6.18 | 113 | 86 | 47 | 4112 | .477 | .643 | .622 |
| D2 | + 修复执行器 | 5.96 | 102 | 86 | 49 | 4116 | .474 | .643 | .622 |

**关键发现**：

- **A0 → A1**（训练 W_proj）：Viol.% 从 7.59 降至 6.01，R3 从 237 降至 174。训练过的投影矩阵改善了深度一致性。
- **A1 → E1**（语义重排）：R3 大幅下降 174 → 86（空间过滤有效），但引入了 R1 违规（95 个）—— 语义重排用侧别精度换取了更好的 NLG 相关性。
- **E1 → B2'**（+ LLM 生成）：R6b 从 65 降至 44（LLM 减少了跨句矛盾）。开始有 NLG 指标：B-4 = 0.471。
- **B2' → B2'v2**（严格侧别证据卡）：**最大单项改进**。Viol.% 从 6.01 降至 4.25，R3 从 86 降至 23。严格证据卡（SSR ≥ 0.9、至少 2 个非跨中线 token、深度门控）大幅降低深度违规，同时保持 NLG 质量。
- **B2'v2 → C2' → D2**（裁判+修复）：NLG 略有提升（.473 → .477 → .474），但违规率反而上升。裁判/修复循环有助于 NLG，但偶尔通过重路由引入新违规。
- **B2'v2 是最佳权衡点**：违规率最低（4.25%），NLG 有竞争力。

---

## Table 3 — Grounding 与 Citation Faithfulness（5K）

路由级 grounding 指标（token 与解剖区域的空间重叠）和生成级 faithfulness（文本与 token 位置的对齐）。在 564 个 grounding 可评估句子上计算（排除了 "bilateral"，因其整个 volume 天然 100%）。

| ID | 配置 | Overlap | Hit@1 | Hit@8 | 路由精度 | 侧别准确率 | CF | DF | 无违规率 |
|----|------|---------|-------|-------|---------|-----------|------|------|---------|
| A0 | Identity W + 空间 | .931 | 100.0 | 100.0 | 100.0 | 100.0 | — | — | 92.4 |
| A1 | Trained W + 空间 | .931 | 100.0 | 100.0 | 100.0 | 100.0 | — | — | 94.0 |
| E1 | 空间过滤 + 语义重排 | .685 | 76.8 | 99.3 | 75.4 | 97.2 | — | — | 94.0 |
| B2' | + 证据卡 v1 | .685 | 76.8 | 99.3 | 75.4 | 96.9 | 76.9 | 97.3 | 94.2 |
| **B2'v2** | **+ 证据卡 v2** | **.689** | **76.8** | **98.2** | **75.8** | **97.1** | **77.6** | **98.4** | **95.8** |
| C2' | + LLM 裁判 | .685 | 76.8 | 99.3 | 75.4 | 96.7 | **81.1** | 97.3 | 94.0 |
| D2 | + 修复执行器 | .685 | 76.8 | 99.3 | 75.4 | 97.0 | 80.6 | 97.3 | 94.2 |

**列定义**：
- **Overlap**：平均重叠比（交集体积 / token 体积），衡量 cited token 落在解剖区域内的程度
- **Hit@k**：top-k cited token 中是否有至少一个与解剖区域重叠 ≥ 50%
- **路由精度**：cited token 中落在正确解剖区域内的比例
- **侧别准确率**：证据卡中侧别约束句子无 R1 违规的比例
- **CF**（Citation Faithfulness）：生成文本中提到的侧别方向与 cited token 实际位置一致的比例
- **DF**（Depth Faithfulness）：token 深度与预期深度范围匹配的比例
- **无违规率**：所有句子中零违规的比例

**关键发现**：

- **Grounding–NLG 权衡**：A0/A1 纯空间路由的路由精度为 100%，但无法生成文本。E1+ 加入语义重排后精度降至 75.4%，但换来了有意义的 NLG 输出。
- **Citation Faithfulness 在不同 LLM 配置间有差异**：CF 从 76.9%（B2'）→ 77.6%（B2'v2）→ **81.1%（C2'）** → 80.6%（D2）。裁判（C2'）通过过滤侧别不一致的生成，带来了最大的 CF 提升。
- **B2'v2 的深度忠实度最高**（98.4%）和无违规率最高（95.8%）。
- **修复（D2）略微降低了 CF**（81.1 → 80.6）：重路由偶尔引入新的侧别不匹配。

---

## Fig 1 — 瀑布图（消融链违规率）

![Waterfall Plot](./paper_figures_5k/fig1_waterfall.png)

左图：消融链各配置的总违规率（%）。B2'v2 达到最低（4.2%）。

右图：R1（侧别）和 R3（深度）违规计数分解。关键观察：B2'v2 的严格证据卡将 R3 从 86 降至 23，降幅 73%。A0/A1 的 R1 为零（纯空间路由不引入侧别错误），E1+ 通过语义重排引入了 R1。

---

## Fig 2 — 预算扫描（k vs 空间忠实度）

![Budget Sweep](./paper_figures/fig2_budget_sweep.png)

Token 预算 k（每句引用的 token 数）vs. 违规率。k=8 是最佳点：继续增大 k 会引入更多区域外 token（R3 随 k 线性增长），而 k < 8 则降低了上下文覆盖度。k=1 时违规率最低（10.5%），但主要由高 R1（132 个违规）驱动 —— 单 token 路由不够有代表性。

---

## Fig 3 — 反事实敏感性分析

![Counterfactual](./paper_figures/fig3_counterfactual.png)

对 3,979 个句子做扰动分析，验证验证器的敏感性：

| 扰动类型 | 违规率 | R1 | R3 | R6b | 被扰动句数 |
|---------|--------|-----|-----|------|-----------|
| 原始 | 4.55% | 2 | 127 | 52 | — |
| 同义改写（对照组） | 4.55% | 2 | 127 | 52 | 1,710 |
| 侧别翻转 | **15.93%** | **455** | 127 | 52 | 614 |
| 存在性翻转 | 6.11% | 2 | 127 | **114** | 62 |

**分析**：
- **同义改写不变性**：违规率完全相同（4.55%），验证器对表面改写保持稳定，不会误触发。
- **侧别敏感性**：翻转 "left" ↔ "right" 导致违规率 3.5 倍上升（4.55% → 15.93%），R1 从 2 跳至 455。验证器正确检测到侧别-token 位置的不匹配。
- **存在性敏感性**：翻转存在/缺失（"无实变" → "实变"）使 R6b 从 52 增至 114。验证器捕获了存在性翻转引入的跨句矛盾。
- **R3 在所有扰动中保持不变**（127）—— 深度违规只取决于 token 位置，与文本内容无关。

---

## Fig 4 — 定性案例

![Qualitative Case](./paper_figures/fig4_qualitative.png)

一个 CT-RATE 案例的可视化：(a) CT 轴位切片；(b) 路由选中的 8 个 token 的 bbox 叠加；(c) 生成的文本 + 证据卡信息。

该句子描述右肺下叶的实变，证据卡显示：8 个 cited token 全部在右侧（dominant side = right，same-side ratio = 1.00），侧别门控和深度门控均通过（laterality gate = right, depth gate = in_range），零违规。

---

## 方法数学描述 — 各模块 Input / Output 与参数来源

> 本节对流水线的核心模块给出数学定义，标注每个参数的来源和默认值。对应的消融配置见 Table 2。

### M1: 3D 证据 Token Bank（Stage 0-2）

**Stage 0 — 伪影风险评分**

$$A_i = \text{clip}\bigl(w_{\text{snr}}\cdot\text{norm}(\sigma_i/|\mu_i|) + w_{\text{st}}\cdot\text{norm}(\text{streak}_i) + w_{\text{out}}\cdot\text{norm}(\text{outlier}_i),\;0,\;1\bigr)$$

| 符号 | 含义 | 来源 |
|------|------|------|
| $\mu_i, \sigma_i$ | 局部窗口灰度均值 / 标准差 | 从 CT volume 计算 |
| $w_{\text{snr}}, w_{\text{st}}, w_{\text{out}}$ | 权重 (0.5, 0.3, 0.2) | 预设超参 |
| norm(·) | 同深度层的 min-max 归一化 | 在线计算 |

软门控：$g_i = \sigma(-k_{\text{gate}}(A_i - \tau_A))$，其中 $k_{\text{gate}}=15$，$\tau_A=0.85$。效果：$A_i > \tau_A$ 时 $g_i\to 0$（高伪影 token 被压低）。

**Stage 1 — 冻结 3D 编码器**

- 输入：CT volume $V$（resize 到 $128^3$）
- 编码器：SwinUNETR（`feature_size=48`，BTCV 预训练，**权重冻结**）
- 输出：$F = \{f_i \in \mathbb{R}^{768}\}_{i=1}^{N}$

**Stage 2 — 综合重要性评分与 Octree 选择**

$$S_i = g_i \cdot (\lambda_h \cdot q(H_i) + \lambda_p \cdot q(P_i))$$

| 符号 | 含义 | 默认值 |
|------|------|--------|
| $H_i$ | 不确定性分支（entropy + variance + boundary） | 从 $F$ 计算 |
| $P_i$ | 语义先验分数 $\sigma(\text{mean}(\|F\|_{\Omega_i}) - \theta_p)$ | 从 $F$ 计算 |
| $q(\cdot)$ | 分位数归一化 $\text{rank}(x)/n$（同层内） | 在线计算 |
| $\lambda_h, \lambda_p$ | 0.6, 0.4 | 超参 |

自适应 Octree 在初始深度 $d_0=2$（64 cells）上递归分裂，按 $S_i$ 降序选择，直到达到预算 $B$（默认 64 个 token）。每个 token 导出时做邻域融合：

$$f_i^{\text{export}} = (1-\beta)\,f_i + \beta\,\bar{b}_i, \quad \beta=0.1$$

- **最终输出（Token Bank）**：$\{(f_i^{\text{export}}, \text{bbox}_i, \text{level}_i)\}_{i=1}^{B}$，即 $B$ 个带位置和深度信息的视觉 token。

---

### M2: 跨模态路由器（Stage 3b）— 消融的核心

**Step 1 — 视觉投影**

$$v_i = \frac{W_{\text{proj}} \cdot f_i}{\|W_{\text{proj}} \cdot f_i\|_2}$$

| 符号 | 含义 | 维度 / 来源 |
|------|------|-------------|
| $f_i$ | Token Bank 中第 $i$ 个 token 的特征 | $\mathbb{R}^{768}$，Stage 2 输出 |
| $W_{\text{proj}}$ | 投影矩阵（**唯一可学参数**，294,912 个） | $\mathbb{R}^{384 \times 768}$ |
| $v_i$ | 归一化后的视觉嵌入 | $\mathbb{R}^{384}$，$\|v_i\|=1$ |

- **A0** 配置：$W_{\text{proj}} = I_{384 \times 768}$（身份矩阵，不训练）
- **A1+** 配置：$W_{\text{proj}}$ 通过 InfoNCE 训练（见下文）

**Step 2 — 文本查询编码**

$$q_s = \frac{\text{TextEnc}(s)}{\|\text{TextEnc}(s)\|_2}$$

- TextEnc: `all-MiniLM-L6-v2`（冻结），输出 $q_s \in \mathbb{R}^{384}$，$\|q_s\|=1$
- 输入 $s$：句子级 topic（如 "consolidation in right lower lobe"）

**Step 3 — 路由打分**

余弦相似度：$r_i^{(s)} = q_s^\top v_i \in [-1, 1]$

两种模式：

| 模式 | 公式 | 使用场景 |
|------|------|---------|
| 解剖优先（A0, B2'v2 等） | $\hat{r}_i^{(s)} = \text{IoU}(\text{bbox}_i, \text{bbox}_{\text{anat}}^{(s)}) + \varepsilon \cdot r_i^{(s)}$，$\varepsilon=0.05$ | $W_{\text{proj}}$ 未校准时，IoU 为主信号 |
| 语义优先（E1, C2' 等） | $\hat{r}_i^{(s)} = r_i^{(s)} + \lambda_{\text{spatial}} \cdot \text{IoU}(\text{bbox}_i, \text{bbox}_{\text{anat}}^{(s)})$，$\lambda_{\text{spatial}}=0.3$ | $W_{\text{proj}}$ 训练后，语义为主 |

**Step 4 — Top-k 选择**

$$E_s = \text{argtop-}k\{\hat{r}_i^{(s)} \mid i = 1,\ldots,B\}, \quad k=8$$

- **输出**：$E_s$，$k$ 个 cited token 的索引集合。这些 token 直接作为 Stage 3c 的证据输入。

**InfoNCE 训练（仅 A1+ 配置）**

$$\mathcal{L} = -\frac{1}{N}\sum_{n=1}^{N}\log\frac{\exp(q_n^\top p_n / \tau)}{\sum_{m=1}^{N}\exp(q_n^\top p_m / \tau)}, \quad \tau=0.07$$

- $p_n = W_{\text{proj}} \cdot \bar{f}_n^+ / \|W_{\text{proj}} \cdot \bar{f}_n^+\|$，其中 $\bar{f}_n^+$ = 正例 token 特征均值
- 训练数据：通过 bootstrap（先用 $W=I$ 跑一遍流水线，收集 trace 中的 sentence-token 配对）
- 优化器：AdamW（lr=$10^{-3}$），余弦退火，早停 patience=10

**消融配置与路由模式对应关系**（Table 2 参考）：

| 配置 | $W_{\text{proj}}$ | 路由模式 | 空间过滤 |
|------|-------------------|---------|---------|
| A0 | $I$（身份） | 解剖优先 | 无 |
| A1 | 训练后 | 解剖优先 | 无 |
| E1 | 训练后 | 语义优先（$\lambda_{\text{spatial}}=0.3$） | IoU + depth filter |
| B2'~D2 | $I$（身份） | 解剖优先 | 无 |

---

### M3: Token-Gated 生成 & 证据卡（Stage 3c）

**证据卡（Evidence Card）构建**

对 $E_s$ 中每个 token 计算侧别：

$$\text{side}(i) = \begin{cases} \text{left} & x_c > x_{\text{mid}} + \delta \\ \text{right} & x_c < x_{\text{mid}} - \delta \\ \text{cross} & \text{otherwise} \end{cases}$$

其中 $x_c = (x_{\min} + x_{\max})/2$，$x_{\text{mid}} = W/2 = 64$（$128^3$ volume），$\delta=0$。

**主导侧比（Same-Side Ratio, SSR）**：

$$\rho = \frac{\max(n_L, n_R)}{n_L + n_R}$$

| 参数 | 证据卡 v1（B2'） | 证据卡 v2（B2'v2） |
|------|----------------|-------------------|
| SSR 阈值 | 无 | $\rho \geq 0.9$ |
| 非跨中线 token 最少数量 | 无 | $\geq 2$ |
| 深度门控 | 无 | 75% token 在 $[\ell_{\min}, \ell_{\max}]$ 内 |
| 侧别允许字段 | 总是 left/right | 仅通过门控才允许 |

- **输入**：$E_s$（路由选中的 $k$ 个 token）+ 句子 topic
- **输出**：证据卡（laterality_allowed, depth_allowed）+ LLM 生成的报告句子
- LLM：Llama-3.1-8B-Instruct，temperature=0.3

**关键设计**：Citation by Construction — cited token 集合恒等于 routed token 集合（$\hat{E}_t \equiv E_t$），不做后处理抽取，从而保证溯源正确性。

---

### M4: 规则验证器 — 6 条规则的数学定义（Stage 4）

> 坐标系统：LPS 方向（`sitk.DICOMOrient(img, "LPS")`），x 轴 0=右，$x_{\max}$=左。

| 规则 | 检查内容 | 数学条件 | 输入 | 输出 | 严重度 |
|------|---------|---------|------|------|--------|
| **R1** | 侧别一致性 | $\frac{|\text{same}|}{|\text{same}| + |\text{opp}|} \geq \theta_{\text{ratio}}=0.6$ | 句子侧别 + token $x$ 坐标 | pass / fail | 1.0 |
| **R2** | 解剖一致性 | $\frac{|\{i : \text{IoU}(\text{bbox}_i, \text{bbox}_{\text{anat}}) > \tau_{\text{IoU}}\}|}{k} \geq \theta_{\text{support}}=0.8$ | token bbox + 解剖 bbox | pass / fail | 0.8 |
| **R3** | 深度一致性 | $\forall i \in E_s: \text{level}_i \in [\ell_{\min}, \ell_{\max}]$ | token level + 预期范围 | pass / fail | 0.7 |
| **R4** | 尺寸合理性 | union bbox volume ∈ 预期范围 | token bbox | — | 0.6 (禁用) |
| **R5** | 否定处理 | 否定句不应引用阳性 token | 句子极性 + token 语义 | — | 1.0 (禁用) |
| **R6a** | 跨句侧别冲突 | 同一解剖关键词的不同句子不矛盾 | 所有句子 | conflict / ok | 0.7 |
| **R6b** | 跨句存在性冲突 | 同一关键词的正/负性不矛盾 | 所有句子 | conflict / ok | 0.8 |

- R4, R5 在实验中禁用（阈值未校准 / fallback dict 禁用）
- R1 豁免：否定句、中线结构（纵隔、气管、脊柱等）
- R2 豁免：bilateral、left lung、right lung（大结构 IoU 过低）

**违规率计算**：

$$\text{Viol.\%} = \frac{|\{s : \exists r \in \{R1,R2,R3,R6a,R6b\}, \text{rule}_r(s) = \text{fail}\}|}{|\text{all sentences}|} \times 100$$

---

### M4+: LLM 裁判与修复（Stage 5, C2'/D2 配置）

**严重度聚合**：

$$\text{sev}_i^{(s)} = \max_{r \in \text{triggered}} \text{sev}_r(i, s) \in [0,1]$$

**Log-smooth 惩罚**：

$$\hat{r}_i^{(s)\prime} = \hat{r}_i^{(s)} - \gamma \cdot \ln(1 + \text{sev}_i^{(s)}), \quad \gamma=2.0$$

- 最大惩罚：$\gamma \ln 2 \approx 1.386$（当 sev=1 时）
- C2'（LLM 裁判）：LLM 判断违规是否为 false positive，建议修复动作
- D2（修复执行器）：执行 reroute_same_side / drop_laterality / drop_depth 等修复，最多 1 次迭代

---

### Grounding / Faithfulness 指标定义（Table 3 参考）

| 指标 | 公式 | 说明 |
|------|------|------|
| Overlap | $\frac{\text{vol}(\text{bbox}_i \cap \text{bbox}_{\text{anat}})}{\text{vol}(\text{bbox}_i)}$，对 $E_s$ 取均值 | intersection / token 体积（非 IoU） |
| Hit@k | $\mathbb{1}[\exists i \in \text{top-}k : \text{overlap}_i \geq 0.5]$ | 检测是否有 token 落入解剖区域 |
| 路由精度 | $\frac{|\{i \in E_s : \text{overlap}_i \geq 0.5\}|}{|E_s|}$ | cited token 在正确区域内的比例 |
| CF（Citation Faithfulness） | $\frac{|\text{text\_lat matches token\_side}|}{|\text{sentences with lat claim}|}$ | 生成文本的侧别与 token 位置一致 |
| DF（Depth Faithfulness） | $\frac{|\{i : \text{level}_i \in [\ell_{\min}, \ell_{\max}]\}|}{|E_s|}$ | token 深度在预期范围内 |

注：Grounding 评估排除 "bilateral" 关键词（bbox 覆盖整个 volume，所有 token 天然 100%）。评估在 564 个 grounding 可评估句子上计算。

---

## 总结

| 指标 | 最优配置 | 数值 |
|------|---------|------|
| 最低违规率 | B2'v2 | 4.25% |
| 最高 BLEU-4 | C2' | 0.477 |
| 最高 Citation Faithfulness | C2' | 81.1% |
| 最高 Depth Faithfulness | B2'v2 | 98.4% |
| 最高无违规率 | B2'v2 | 95.8% |
| **最佳整体权衡** | **B2'v2** | **Viol 4.25%, B-4 .473, CF 77.6%, DF 98.4%** |

**B2'v2**（严格侧别证据卡 v2）是我们推荐的配置：在空间一致性、NLG 质量和 grounding 忠实度之间达到了最佳平衡。

---

*数据来源：`outputs/paper_figures_5k/table2_data.json`、`outputs/paper_figures_5k/table3_grounding_data.json`*
*图表来源：`outputs/paper_figures_5k/fig1_waterfall.pdf`、`outputs/paper_figures/fig2_budget_sweep.pdf`、`outputs/paper_figures/fig3_counterfactual.pdf`、`outputs/paper_figures/fig4_qualitative.pdf`*
