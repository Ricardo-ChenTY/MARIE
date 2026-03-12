# Mediastinum Tau Sweep 实验指南

## 问题背景

当前配置下，`mediastinum` 解剖结构的所有句子都被标记为违规。需要通过调整 `tau_iou` 阈值来解决这个问题。

## 实验策略

围绕理论最优值 0.0442 做局部密扫，测试 5 个阈值点：
- 0.03
- 0.035
- 0.04
- 0.045
- 0.05

## 重点观察指标

1. **R2_ANATOMY 总数** - 解剖结构违规总数
2. **LLM confirmed 数量** - LLM Judge 确认的违规数
3. **mediastinum confirmed 比例** - mediastinum 被确认违规的比例
4. **总体 violation_sentence_rate** - 整体句子违规率

---

## 使用步骤

### Step 1: 运行 Sweep 实验

```bash
# 在项目根目录下运行
bash Scripts/run_mediastinum_tau_sweep.sh
```

**预计时间**: 约 1-2 小时（5 个配置 × 50+50 病例）

**输出目录**: `outputs/mediastinum_tau_sweep_50/`

生成的子目录：
```
outputs/mediastinum_tau_sweep_50/
├── tau003/      # tau_iou=0.03
├── tau0035/     # tau_iou=0.035
├── tau004/      # tau_iou=0.04
├── tau0045/     # tau_iou=0.045
└── tau005/      # tau_iou=0.05
```

---

### Step 2: 分析 Sweep 结果

#### 2.1 Mediastinum 专项分析（推荐）

```bash
python Scripts/analyze_mediastinum_sweep.py \
  --sweep_root outputs/mediastinum_tau_sweep_50
```

**生成文件**:
```
outputs/mediastinum_tau_sweep_50/mediastinum_analysis/
├── mediastinum_sweep_summary.csv        # Mediastinum 汇总数据
├── mediastinum_tau_curves.png           # Mediastinum 指标曲线图
├── tau*_anatomy_breakdown.csv           # 各 tau 下所有解剖结构详情
└── top_anatomy_tau_curves.png           # Top 10 解剖结构对比曲线
```

**输出示例**:
```
══════════════════════════════════════════════════════════════════
  Mediastinum 专项统计
══════════════════════════════════════════════════════════════════
run_dir   tau_iou  mediastinum_sentences  mediastinum_violations  ...
tau003     0.03           25                     25              ...
tau0035    0.035          25                     20              ...
tau004     0.04           25                     15              ...
tau0045    0.045          25                     10              ...
tau005     0.05           25                      5              ...

推荐配置 (基于 mediastinum_violation_rate 最低):
  tau_iou = 0.05
  mediastinum_violation_rate = 0.2000
  mediastinum_r2_anatomy = 5
  mediastinum_llm_confirmed = 3
```

#### 2.2 通用 Sweep 对比

```bash
python analyze_outputs.py \
  --mode sweep \
  --sweep_root outputs/mediastinum_tau_sweep_50 \
  --sweep_glob 'tau*'
```

**生成文件**:
```
outputs/mediastinum_tau_sweep_50/analysis_exports/
├── sweep_summary.csv           # 所有配置汇总
├── sweep_heatmap.png           # 参数热力图（如果适用）
└── sweep_rule_bars.png         # 规则违规对比柱状图
```

---

### Step 3: 深度分析单个配置

如果某个 tau 值表现最好，可以深入分析该配置：

```bash
# 例如分析 tau=0.04 的结果
python analyze_outputs.py \
  --out_dir outputs/mediastinum_tau_sweep_50/tau004 \
  --expected_cases_map ctrate=50,radgenome=50 \
  --inspect_n 5
```

**生成文件**:
```
outputs/mediastinum_tau_sweep_50/tau004/analysis_exports/
├── summary_bars.png                     # 数据集汇总柱状图
├── dataset_aggregate.csv                # 数据集聚合统计
├── rule_violation_count.png             # 规则违规计数
├── anatomy_r2_breakdown.csv             # R2_ANATOMY 按解剖结构分组
├── anatomy_all_violation_rate.csv       # 所有解剖结构违规率
├── sentence_detail.csv                  # 句子级详情
├── sentence_violation_rate.csv          # 句子违规率统计
├── case_violation_hist.png              # 病例违规率分布
└── abnormal_cases_ranked.csv            # 异常病例排序
```

**查看 mediastinum 详细数据**:
```bash
# 从 sentence_detail.csv 中筛选 mediastinum
cd outputs/mediastinum_tau_sweep_50/tau004/analysis_exports
grep -i mediastinum sentence_detail.csv > mediastinum_sentences.csv
```

---

## 结果解读

### 关键指标

1. **mediastinum_violation_rate**
   - **越低越好**
   - 目标: < 0.5（即少于 50% 的 mediastinum 句子被标记为违规）

2. **mediastinum_r2_anatomy**
   - R2_ANATOMY 规则触发次数
   - **越低越好**

3. **mediastinum_llm_confirmed**
   - LLM Judge 确认的违规数
   - 这是"真正的违规"
   - **越低越好**（但不是越低越好 - 需要平衡）

4. **total_violation_rate**
   - 整体违规率
   - 确保降低 mediastinum 违规率的同时，不会显著提高整体违规率

### 决策标准

选择最优 `tau_iou` 时，考虑以下优先级：

1. **mediastinum_violation_rate** 显著降低
2. **mediastinum_llm_confirmed** 接近 0
3. **total_violation_rate** 没有明显上升
4. 其他解剖结构的违规率没有显著恶化

---

## 后续步骤

找到最优 tau_iou 后：

### 1. 更新主实验配置

编辑 `Scripts/run_smoke50_stage0_5_only.sh`：

```bash
--tau_iou 0.04  # 替换为最优值
```

### 2. 运行完整 450+450 实验

```bash
bash Scripts/run_stage0_5_llama_server.sh  # 使用新的 tau_iou
```

### 3. 验证结果

```bash
python analyze_outputs.py \
  --out_dir outputs/stage0_5_llama_450 \
  --expected_cases_map ctrate=450,radgenome=450
```

---

## 故障排查

### 问题：某个 tau 配置运行失败

**检查日志**:
```bash
tail -100 outputs/mediastinum_tau_sweep_50/tau004/run.log
```

**常见原因**:
- GPU 内存不足 → 减少 batch size 或使用 CPU
- 模型未下载 → 运行 `Scripts/_download_core.py`
- 数据集缺失 → 检查 manifests/ 目录

### 问题：分析脚本报错

**确保依赖安装**:
```bash
pip install pandas matplotlib seaborn
```

### 问题：结果不符合预期

**重新运行单个配置**:
```bash
# 例如重新运行 tau=0.04
rm -rf outputs/mediastinum_tau_sweep_50/tau004
# 然后在 run_mediastinum_tau_sweep.sh 中注释掉其他 tau，只保留 0.04
```

---

## 联系与反馈

如有问题，请查看主项目 [README.md](../README.md) 或提交 issue。
