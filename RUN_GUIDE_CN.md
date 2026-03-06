# ProveTok_ACM 运行说明（给协作者）

这份说明对应当前代码的目标范围：**Stage 0-4**（不调用 LLM 生成）。

## 1. 项目里有什么

当前主流程只做四件事：

1. Stage 0-2：确定性 token bank 构建（SwinUNETR + 八叉树分裂）
2. Stage 3：Router（teacher-forcing，用参考报告句子做查询）
3. Stage 4：Verifier 规则审计（R1-R5）
4. Stage 6：结构化日志落盘（每例可追溯）

关键入口：

- `run_mini_experiment.py`：主运行脚本
- `validate_stage0_4_outputs.py`：结果验收脚本

---

## 2. 数据接口（你要准备什么）

需要两个 manifest CSV（CT-RATE / RadGenome 各一个）。

### 必需列

- `case_id`：病例唯一 ID
- `volume_path`：体数据路径（支持医学影像文件；也支持 `.npy`）
- `report_text`：参考报告文本（用于 teacher-forcing 分句）

### 可选列

- `split`：当你加 `--build_mini` 时用于分层抽样（450/450）

### 体数据说明

- 若是医学影像文件：自动从元数据读取 spacing
- 若是 `.npy`：默认 spacing = `(1.0, 1.0, 1.0)` mm

---

## 3. 怎么运行

## 3.1 一次性跑（含 450/450 mini 构建）

```powershell
python run_mini_experiment.py `
  --ctrate_csv <path_to_ctrate_manifest.csv> `
  --radgenome_csv <path_to_radgenome_manifest.csv> `
  --build_mini `
  --out_dir outputs_stage0_4 `
  --max_cases 450 `
  --expected_cases_per_dataset 450 `
  --cp_strict `
  --encoder_ckpt <path_to_swinunetr_ckpt.pt> `
  --text_encoder semantic `
  --text_encoder_model sentence-transformers/all-MiniLM-L6-v2 `
  --device cuda `
  --token_budget_b 64 `
  --k_per_sentence 8 `
  --lambda_spatial 0.3 `
  --tau_iou 0.1 `
  --beta 0.1
```

如果没有 GPU，可把 `--device cpu`。

## 3.2 只跑已有 manifest（不重建 mini）

去掉 `--build_mini` 即可。

---

## 4. 能得到什么（输出文件）

所有输出在 `--out_dir` 下（默认 `outputs_stage0_4`）。

## 4.1 全局汇总

- `summary.csv`
- `ctrate_case_summary.csv`
- `radgenome_case_summary.csv`
- `run_meta.json`（记录本次运行实际选取/处理样本数、关键超参）
  - 包含 `cp_strict / text_encoder / encoder_ckpt / r2_mode` 等关键信息

## 4.2 每个 case 的核心产物

路径：`outputs_stage0_4/cases/<dataset>/<case_id>/`

- `tokens.npy`：`[B, d]` token 特征
- `tokens.pt`：同上（PyTorch tensor）
- `tokens.json`：每个 token 的结构信息
- `bank_meta.json`：token bank 级元信息
- `trace.jsonl`：逐句路由 + verifier 结果

---

## 5. 每一步的 Input / Output

## Stage 0-2（Deterministic Token Bank）

输入：

- `volume_path` 指向的 3D 体数据
- 配置：`B, depth_max, beta` 等

输出（按 case 落盘）：

- `tokens.npy` / `tokens.pt`
- `tokens.json`（`token_id, level, bbox_3d_voxel, bbox_3d_mm, cached_boundary_flag/params`）
- `bank_meta.json`（`B, depth_max, beta, encoder_name, voxel_spacing, global_bbox`）

## Stage 3（Router, Teacher-Forcing）

输入：

- 参考报告分句得到 `s`
- token bank 的 `f_i` 与 `bbox_i`
- 超参数：`k, lambda_spatial`

计算：

- `v_i = normalize(W_proj f_i)`
- `r_i^(s) = cos(q_s, v_i)`
- 可选空间先验：`+ lambda_spatial * IoU(bbox_i, bbox_anatomy)`
- 取 top-k citations

输出：

- 每句 `topk_token_ids` 和 `topk_scores`（在 `trace.jsonl`）

## Stage 4（Verifier）

输入：

- 原句文本
- 每句 citations（token id）
- token 几何信息（bbox / level）

规则：

- `R1` 侧别一致性
- `R2` 解剖区域一致性
- `R3` 深度层级一致性
- `R4` 大小/范围合理性
- `R5` 否定处理

输出：

- 每句 `violations[]`（在 `trace.jsonl`）

## Stage 6（统一日志）

输入：

- Stage 0-4 全部中间结果

输出：

- `trace.jsonl` 第一行为 case_meta：
  - `B, k, B_plan, lambda_spatial, tau_IoU, ell_coarse, beta`
- 后续每句记录：
  - `sentence_text, q_s, topk_token_ids, topk_scores, violations`

---

## 6. 如何验收“达到预期”

运行验收脚本：

```powershell
python validate_stage0_4_outputs.py `
  --out_dir outputs_stage0_4 `
  --datasets ctrate,radgenome `
  --expected_cases_map ctrate=450,radgenome=450 `
  --save_report outputs_stage0_4\validation_report.json
```

通过标准（脚本自动检查）：

1. 每 case 文件齐全：`tokens.* + bank_meta + trace`
2. `tokens.json / bank_meta.json` 字段完整
3. `trace.jsonl` 有 case_meta + sentence 记录
4. `topk_token_ids` 合法且与 token bank 对齐
5. `topk_scores` 数值格式正确（并做排序检查）

---

## 7. 常见问题

1. 报错 `Missing column ...`  
说明 manifest 缺少必需列，请检查 `case_id, volume_path, report_text`。

2. 跑得慢  
先把 `--max_cases` 调小（如 10）做 smoke test，再放到 450。

3. 显存不足  
改 `--device cpu`，或减小 `--resize_d/h/w`。

4. `.npy` spacing 不准确  
当前默认 `(1,1,1)` mm；如要真实 mm，请用带元数据的医学影像格式。

---

## 8. R2 Sweep Quick Start（50/50，CP strict）

先做 50-case 参数扫描，确认参数区间后再上 450/450 全量，避免浪费算力。

### 扫描维度（2 × 3 = 6 组）

| 参数 | 扫描值 | 说明 |
|------|--------|------|
| `tau_iou` | 0.10, 0.05 | 单个 token 被认为"命中"解剖区域的 IoU 下限 |
| `r2_min_support_ratio` | 1.0, 0.8, 0.6 | cited tokens 中命中比例的最低要求 |

> **为什么用 64^3 做 sweep**：IoU = token_vol / anatomy_vol，分子分母同比例
> 缩放，64^3 和 128^3 下 tau 最优区间相同。速度快 4-8x，结论可直接用于 128^3 主实验。

### 一键跑（Colab 或 5090 本地）

```bash
bash Scripts/run_r2_sweep_50_cp_strict_colab.sh
```

输出目录结构：`r2_tau{t}_ratio_{r}/`，例如：
- `r2_taut010_ratio_1.0/`
- `r2_taut010_ratio_0.8/`
- `r2_taut005_ratio_0.6/`
- ...

### 汇总结果

```bash
python Scripts/summarize_r2_sweep.py \
  --sweep_root /content/drive/MyDrive/Data/outputs_stage0_4_r2sweep_50_cp_strict \
  --glob "r2_tau*_ratio_*" \
  --save_csv /content/drive/MyDrive/Data/outputs_stage0_4_r2sweep_50_cp_strict/sweep_summary.csv
```

汇总表按 `tau_iou` 和 `r2_min_support_ratio` 排序，观察 `violation_sentence_rate`
和 `R2_ANATOMY` 的变化趋势，选出下降明显且 R1/R5 稳定的参数组合，用于第 9 节全量跑。

---

## 9. 本地 RTX 5090 运行指南

### 9.1 环境准备

```bash
# 确认 GPU
nvidia-smi

# 安装依赖（推荐 conda）
conda env create -f environment.yaml
conda activate provetok

# 安装 sentence-transformers（semantic encoder 必需）
pip install sentence-transformers
```

首次运行 semantic encoder 会自动下载 `all-MiniLM-L6-v2`（约 90MB），之后会缓存。

### 9.2 先跑 R2 sweep（64^3，确认参数）

在本地把 `run_r2_sweep_50_cp_strict_colab.sh` 里的路径替换成本地路径后直接跑，
或手动执行：

```bash
CTRATE_CSV="/path/to/ctrate_manifest.csv"
RADGENOME_CSV="/path/to/radgenome_manifest.csv"
ENCODER_CKPT="/path/to/swinunetr.ckpt"
OUT_ROOT="outputs_r2sweep_50"

python run_mini_experiment.py \
  --ctrate_csv "${CTRATE_CSV}" \
  --radgenome_csv "${RADGENOME_CSV}" \
  --out_dir "${OUT_ROOT}/r2_taut010_ratio_0.8" \
  --max_cases 50 \
  --expected_cases_per_dataset 50 \
  --cp_strict \
  --encoder_ckpt "${ENCODER_CKPT}" \
  --text_encoder semantic \
  --text_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
  --text_encoder_device cuda \
  --device cuda \
  --resize_d 64 --resize_h 64 --resize_w 64 \
  --token_budget_b 64 \
  --k_per_sentence 8 \
  --lambda_spatial 0.3 \
  --tau_iou 0.10 \
  --beta 0.1 \
  --r2_mode ratio \
  --r2_min_support_ratio 0.8
```

### 9.3 全量跑（128^3，450/450）

sweep 确定参数后，把 tau 和 ratio 换成最优值，去掉 resize 参数（默认 128^3）：

```bash
python run_mini_experiment.py \
  --ctrate_csv "/path/to/ctrate_manifest.csv" \
  --radgenome_csv "/path/to/radgenome_manifest.csv" \
  --out_dir outputs_stage0_4_450_128 \
  --max_cases 450 \
  --expected_cases_per_dataset 450 \
  --cp_strict \
  --encoder_ckpt "/path/to/swinunetr.ckpt" \
  --text_encoder semantic \
  --text_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
  --text_encoder_device cuda \
  --device cuda \
  --token_budget_b 64 \
  --k_per_sentence 8 \
  --lambda_spatial 0.3 \
  --tau_iou <sweep 确定的最优值> \
  --beta 0.1 \
  --r2_mode ratio \
  --r2_min_support_ratio <sweep 确定的最优值>
```

### 9.3.1 可选：R5 Fallback 开关对比

全量跑完成后，如需量化 R5 Negation Fallback 的贡献，可以用同样的参数再跑一次，
只加 `--r5_fallback_lexicon false`：

```bash
python run_mini_experiment.py \
  --ctrate_csv "/path/to/ctrate_manifest.csv" \
  --radgenome_csv "/path/to/radgenome_manifest.csv" \
  --out_dir outputs_stage0_4_450_128_r5off \
  --max_cases 450 \
  --expected_cases_per_dataset 450 \
  --cp_strict \
  --encoder_ckpt "/path/to/swinunetr.ckpt" \
  --text_encoder semantic \
  --text_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
  --text_encoder_device cuda \
  --device cuda \
  --token_budget_b 64 \
  --k_per_sentence 8 \
  --lambda_spatial 0.3 \
  --tau_iou <同上> \
  --beta 0.1 \
  --r2_mode ratio \
  --r2_min_support_ratio <同上> \
  --r5_fallback_lexicon false
```

对比两次输出的 summary，重点看：

- `R5_NEGATION` 计数：从多少降到 0（量化 R5 贡献）
- `violation_sentence_rate`：总体下降幅度
- `R1/R2/R4`：应基本不变（确认 R5 与其他规则独立）

> 汇总脚本 `summarize_r2_sweep.py` 当前不读 `r5_fallback_lexicon` 字段。
> 如需在同一张表里对比 R5 开/关结果，后续可在汇总表中新增该列。

### 9.4 验收

```bash
python validate_stage0_4_outputs.py \
  --out_dir outputs_stage0_4_450_128 \
  --datasets ctrate,radgenome \
  --expected_cases_map ctrate=450,radgenome=450 \
  --save_report outputs_stage0_4_450_128/validation_report.json
```

### 9.5 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `sentence-transformers` 下载慢 | 首次下载模型 | 挂 VPN 或提前下载放到本地后用 `--text_encoder_model /path/to/model` |
| 显存不足（128^3） | 5090 有 32GB 不太可能，但批量跑时注意 | 加 `--device cpu` 降级排查 |
| `expected_cases` 不匹配 | manifest 行数不足 | 去掉 `--expected_cases_per_dataset` 或检查 manifest |
| sweep 结果目录找不到 | glob 旧命名 `r2_ratio_*` | 改为 `--glob "r2_tau*_ratio_*"` |
