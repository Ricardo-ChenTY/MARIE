# ProveTok v3.0 实现状态 vs chinese_v3.tex 数学规范

更新日期：2026-03-14（第二轮校对）

本文档逐公式对照 `main.tex`（最新修订版）与 GitHub 代码实现，标注已完成、可接受差异、以及尚未对齐的部分。

---

## 总览

| 类别 | 完全对齐 | 可接受差异 | 未对齐 |
|------|---------|-----------|--------|
| M1 (Token Bank, Eq 1-7) | 9/9 | — | — |
| M2 (Router, Eq 9-12 + anat_primary) | 6/6 | W_proj 未训练 | — |
| M3 (门控生成, Eq 20) | 4/4 | — | — |
| M4 (校验 R1-R6 + 惩罚 + 重路由) | 10/10 | R4 禁用, R5 部分 | — |
| M5 (统计协议) | 3/4 | — | C_attn 代理 |
| **合计** | **32/33** | **2 (可接受)** | **1** |

---

## 1. M1 — Stage 0-2：证据 Token 化与预算约束

| tex 公式 | 描述 | 对应代码 | 状态 |
|---------|------|---------|------|
| Eq(1) | 伪影风险 `A_i = clip(...)` | `stage0_artifacts.py:compute_artifact_components()` | ✅ |
| Eq(2) | 软门控 `g_i = σ(-k(A_i - τ_A))` | `stage0_2.py:soft_gate()` | ✅ |
| Eq(3) | 重要性分数 `S_i = g_i·(λ_h·q(H) + λ_p·q(P))` | `stage0_2.py:importance_score()` | ✅ |
| Eq(4-5) | 边界混合 `f_export = (1-β)f + β·b` | `stage0_2.py:boundary_context_blend()` | ✅ |
| Remark | norm(·) = min-max 归一化 | `stage0_artifacts.py:_minmax_norm()` | ✅ |
| Remark | 邻域为空时 fallback b=f | `boundary_context_blend()` 检查 `not neighbor_features` | ✅ |
| — | 分位数归一化 q(·) 按病例/深度层 | `math_utils.py:quantile_rank()` + per-level 分组 | ✅ |
| — | 自适应八叉树分裂 + NMS + 预算 B | `stage2_octree_splitter.py:AdaptiveOctreeSplitter` | ✅ |
| — | Stage 1: SwinUNETR feature_size=48 冻结 | `stage1_swinunetr_encoder.py:FrozenSwinUNETREncoder` | ✅ |

**无缺口**

---

## 2. M2 — Stage 3a/3b：粗证据规划 + 轻量路由器

| tex 公式 | 描述 | 对应代码 | 状态 |
|---------|------|---------|------|
| Eq(9) | 投影归一化 `v = W·f / ‖W·f‖` | `stage3_router.py:_projected_token()` | ✅ |
| Eq(10) | 余弦路由 `r = cos(q, v)` | `_routing_score()` | ✅ |
| Eq(11) | 空间先验 `r̂ = r + λ·IoU` | `_routing_score()` 第 54 行 | ✅ |
| Eq(anat) | 解剖优先 `r̂ = IoU + ε·r` | `anatomy_spatial_routing` 分支 | ✅ |
| Eq(12) | InfoNCE 多正样本损失 | `infonce_loss()` | ✅ |
| — | 规划预算 `B_plan = min(32, B/4)` | `RouterConfig.planning_budget()` | ✅ |

### 已知差异（可接受）

| 项目 | tex | 实现 | 说明 |
|------|-----|------|------|
| W_proj 训练状态 | 需 InfoNCE 训练 | identity fallback | anatomy_spatial_routing 模式绕过 W_proj 依赖；`train_wprojection.py` 已就绪 |

---

## 3. M3 — Stage 3c：逐句 Token-Gated 生成

| tex 规范 | 描述 | 对应代码 | 状态 |
|---------|------|---------|------|
| Eq(20) | 构造性引用 `Ê_s = E_s` | `stage3c_generator.py` citations ≡ routed IDs | ✅ |
| — | 输入：E_s + 前文历史 | `_build_generation_prompt()` 接收 `history` | ✅ |
| — | 句间历史上下文 | `stage0_4_runner.py:generated_history` 累积 | ✅ |
| — | 日志记录证据 ID | `trace.jsonl:topk_token_ids` | ✅ |

---

## 4. M4 — Stage 4-5：规则校验与重路由

| tex 规范 | 描述 | 对应代码 | 状态 |
|---------|------|---------|------|
| R1 侧别 | center x vs midline | `stage4_verifier.py` R1 block | ✅ |
| R1 增强 | ratio/negation/midline 三项 | `r1_min_same_side_ratio` + `r1_negation_exempt` + `r1_skip_midline_keywords` | ✅ |
| R2 解剖 | IoU ≥ τ | R2 block, support-ratio 模式 | ✅ |
| R2 增强 | 支持率模式 + 大结构豁免 | `r2_min_support_ratio` + `r2_skip_keywords` | ✅ |
| R3 深度 | level range 检查 | R3 block | ✅ |
| R4 大小 | union bbox volume | R4 block（**当前实验禁用**） | ✅ (tex 已标注) |
| R5 否定 | metadata conflict + fallback | R5 block（**fallback 词典禁用**） | ✅ (tex 已标注) |
| R6 跨句 | R6a 侧别冲突 + R6b 存在/缺失 | `cross_sentence_check()` | ✅ |
| 惩罚 | `r' = r - γ·ln(1+sev)` | `reroute_scores_log_smooth()` per-token 模式 | ✅ |
| LLM Judge | 二次裁决 + 四种后端 | `stage5_llm_judge.py:LLMJudge` | ✅ |
| 重路由协议 | reroute → regen → re-verify → de-specify | `stage0_4_runner.py` 第 155-200 行 | ✅ |
| 降格表述 | despecify_text() 移除空间词 | `stage3c_generator.py:despecify_text()` | ✅ |
| 最多一次 | cfg.reroute.max_retry = 1 | `config.py:RerouteConfig` | ✅ |

### severity_by_rule 对照

| 规则 | tex 值 | config.py 值 | 状态 |
|------|--------|-------------|------|
| R1_LATERALITY | 1.0 | 1.0 | ✅ |
| R2_ANATOMY | 0.8 | 0.8 | ✅ |
| R3_DEPTH | 0.7 | 0.7 | ✅ |
| R4_SIZE | 0.6 | 0.6 | ✅ |
| R5_NEGATION | 1.0 | 1.0 | ✅ |
| R6a_CROSS_LATERALITY | 0.7 | 0.7 (hardcoded) | ✅ |
| R6b_CROSS_PRESENCE | 0.8 | 0.8 (hardcoded) | ✅ |

### mediastinum R2 处理方式

| 机制 | tex 描述 | 代码实际 | 状态 |
|------|---------|---------|------|
| bilateral/left lung/right lung | r2_skip_keywords | 同 | ✅ |
| mediastinum | τ_IoU 从 0.05 降至 0.04 | 同（sweep 实验验证） | ✅ |

---

## 5. M5 — 统计协议

| tex 公式 | 描述 | 对应代码 | 状态 |
|---------|------|---------|------|
| Eq(24) | C_LLM = Σ(|p|+|g|) | `analyze_outputs.py` 用调用次数近似 | ⚠️ tex 已标注简化 |
| Eq(28) | C_attn^cache = Σ(p²+gp+g(g-1)/2) | 代码用 k·B 简化代理 | ⚠️ tex 已标注差异 |
| Eq(29) | Bootstrap CI (R=5000, 病人级) | `_bootstrap_ci()` | ✅ |
| — | Holm 步降校正 | `_holm_correction()` | ✅ |

### 未对齐项

| # | 问题 | 严重度 | 修复方案 |
|---|------|--------|---------|
| 1 | C_attn 用 k·B 代理而非 Eq(28) | 低 | 在 generator 中记录 p_j/g_j 后可更新 |

---

## 6. tex vs 代码 完整修改记录 (v2 → v3)

| 修改点 | 位置 | 说明 |
|--------|------|------|
| Stage 1 指定 SwinUNETR | §4.3 | 补充 feature_size=48, spatial_dims=3, resize 128³ |
| R2 mediastinum 处理方式 | §7.1 R2 增强 | 修正：不是 skip，而是 τ_IoU=0.04 阈值解决 |
| severity_by_rule 默认值 | §7.1 末尾 | 新增 R1-R5 严重度具体数值 |
| R4/R5 禁用状态 | §7.1 Remark | 新增：标注 450/450 实验中的启用/禁用状态 |
| LLM Judge 实验数据 | §7.2 Remark | 补充：Llama-3.1-8B、54% 过滤率、四种后端 |
| 重路由协议 | §7.2 | 增加 re-verify 步骤；despecify 词类详细列表 |
| C_LLM 实现简化 | §8.1 Remark | 新增：说明代码用调用次数统计 |
| M5 实现状态 | §8.4 Remark | 新增：Bootstrap CI 和 Holm 已实现 |

---

## 文件清单

| 文件 | 对应 Module | 关键函数/类 |
|------|-----------|-----------|
| `stage0_scorer.py` | M1 Stage 0 | `DeterministicArtifactScorer` |
| `stage0_artifacts.py` | M1 Stage 0 | `compute_artifact_components()`, `_minmax_norm()` |
| `stage0_2.py` | M1 Stage 0-2 | `artifact_risk_score()`, `soft_gate()`, `importance_score()`, `boundary_context_blend()` |
| `math_utils.py` | 通用 | `sigmoid()`, `quantile_rank()`, `normalize_l2()`, `dot()` |
| `stage1_swinunetr_encoder.py` | M1 Stage 1 | `FrozenSwinUNETREncoder` |
| `stage2_octree_splitter.py` | M1 Stage 2 | `AdaptiveOctreeSplitter` |
| `simple_modules.py` | M2 Stage 3a | `ReportSentencePlanner`, `RuleBasedAnatomyResolver` |
| `stage3_router.py` | M2 Stage 3b | `Router`, `infonce_loss()` |
| `stage3c_generator.py` | M3 Stage 3c | `Stage3cGenerator`, `despecify_text()` |
| `stage4_verifier.py` | M4 Stage 4 | `Verifier` (R1-R6) |
| `stage5_llm_judge.py` | M4 Stage 5 | `LLMJudge`, `reroute_scores_log_smooth()` |
| `stage0_4_runner.py` | 全流程 | `run_case_stage0_4()` |
| `config.py` | 全局配置 | `ProveTokConfig` |
| `types.py` | 数据类型 | `BBox3D`, `EvidenceToken`, `SentenceOutput`, `RuleViolation` |
| `analyze_outputs.py` | M5 | `analyze_m5_protocol()`, `_bootstrap_ci()`, `_holm_correction()` |
| `run_mini_experiment.py` | CLI 入口 | 全部参数 |
| `train_wprojection.py` | M2 训练 | InfoNCE W_proj 训练 |
