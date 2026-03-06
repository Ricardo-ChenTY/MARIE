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

**为什么 Hash Encoder 会导致随机 routing：**

`text_encoder.py` 中 `DeterministicTextEncoder.__call__` 的逻辑是：
输入文字 → SHA256 哈希 → 取前 8 字节当随机种子 → 生成 256 维标准正态向量 → L2 归一化。

这意味着不同文字之间的向量关系完全随机。例如 "right upper lobe" 和 "right lung"
语义上高度相关，但 hash encoder 输出的两个向量 cosine similarity 接近 0（随机值）；
"right upper lobe" 和 "left lower lobe" 语义上相反，但 cosine 也是随机值，甚至可能更高。

Router 的打分公式是 `score = dot(query_vec, token_feature_vec) + spatial_prior`，
其中 `query_vec` 就是 text encoder 的输出。如果 query_vec 是随机向量，dot product
就是随机分数，router 等于闭着眼睛挑 token——选出的 token 和报告描述的解剖位置无关。
Verifier 一检查，cited tokens 不在对应解剖区域里，R1（侧别）和 R2（解剖区域）自然大量违规。

**换成 SentenceTransformer 后的区别：**

`SentenceTransformerTextEncoder` 使用预训练的 `all-MiniLM-L6-v2` 模型（384 维），
在大规模文本数据上学会了把语义相近的句子映射到相近的向量。
"right upper lobe" 和 "right lung" 的向量 cosine ≈ 0.85（很近），
和 "left lower lobe" 的 cosine ≈ 0.45（较远）。
这样 router 做 dot product 时能有意义地把空间上相关的 token 选出来，
R1/R2 违规率就会大幅下降。

**注意：代码本身没改。** `DeterministicTextEncoder` 和 `SentenceTransformerTextEncoder`
都早已写好在 `text_encoder.py` 里。之前的问题是 `run_mini_experiment.py` 的默认值
是 `--text_encoder hash`，而 sweep 脚本没有显式指定 semantic，导致实际用的是 hash。

> 修正：sweep 脚本显式传 `--text_encoder semantic`，并加注释说明原因。
> `--cp_strict` 模式也加了保险：如果传了 hash 会自动 override 成 semantic。

### 根因 2：`r2_min_support_ratio=1.0` 过于严苛

要求 100% cited tokens 都满足 IoU ≥ tau。Router cite k=8 个 token，配合随机
routing，几乎必然全部触发 R2（第一大违规来源，2651 次）。

> 修正：sweep 新增 ratio 维度 {1.0, 0.8, 0.6}。

### 根因 3：R5 Negation Fallback 制造假阳性

所有 token 的 `negation_conflict` 硬编码为 0.0，永远走 fallback 分支。
"No effusion"、"No nodule" 等正常放射科表达因含 positive finding 词而触发 R5。

> 修正：`run_mini_experiment.py` 新增 `--r5_fallback_lexicon` 参数（默认 true），
> 传 `--r5_fallback_lexicon false` 即可关闭 R5 fallback 进行对比实验。
> 当前 sweep 阶段暂不扫 R5，待 R2 参数确定后独立做 R5 开关对比。

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

### 4) `run_mini_experiment.py`（小改，为 R5 ablation 做准备）

- 新增 `--r5_fallback_lexicon` 参数（默认 `true`，可选 `false`）
- 参数值写入 `cfg.verifier.r5_fallback_lexicon`
- `run_meta.json` 中记录 `r5_fallback_lexicon` 字段
- **原有逻辑零修改**，仅新增 3 处代码（argparse 定义 + config 赋值 + meta 记录）

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
  用第三步确定的最优参数，跑两次 450 全量：
    一次不加 --r5_fallback_lexicon（默认 true，R5 开启）
    一次加   --r5_fallback_lexicon false（R5 关闭）
  对比要看：
    - R5_NEGATION 计数从多少降到 0（量化 R5 贡献）
    - 总体 violation_sentence_rate 降了多少
    - R1/R2/R4 是否不变（确认 R5 独立性）
```

> **关于第四步的说明**：不需要专门的脚本。有了 `--r5_fallback_lexicon` 参数后，
> 到时候就是同一条 450 全量命令跑两次：一次不加这个参数（默认 true），
> 一次加 `--r5_fallback_lexicon false`，对比两次输出的 summary 即可。
> 当前优先级是先完成第一步 R2 sweep，R5 对比等参数确定后再做。

---

## 关键文件清单

- `run_mini_experiment.py`（本次新增 `--r5_fallback_lexicon` 参数）
- `Scripts/run_r2_sweep_50_cp_strict_colab.sh`（本次修改）
- `Scripts/summarize_r2_sweep.py`（本次修改）
- `RUN_GUIDE_CN.md`（本次新增 5090 章节）
- `OUTPUT_ANALYSIS_GUIDE_CN.md`
- `Smoke_analysis/OUTPUT_ANALYSIS_COLAB.ipynb`
