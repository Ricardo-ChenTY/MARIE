# ProveTok_ACM

中文运行说明：[`RUN_GUIDE_CN.md`](./RUN_GUIDE_CN.md)
数据集下载说明：[`Scripts/DOWNLOAD_GUIDE.md`](./Scripts/DOWNLOAD_GUIDE.md)

## 快速开始（服务器）

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据和模型权重

当前项目内固定使用以下路径：

| 文件 | 说明 |
|------|------|
| `manifests/ctrate_manifest.csv` | CT-RATE 数据集清单（含 case_id / volume_path / report_text 列） |
| `manifests/radgenome_manifest.csv` | RadGenome 数据集清单（同上） |
| `checkpoints/swinunetr.ckpt` | SwinUNETR CT 图像编码器权重 |
| `dataset/CT-RATE/` | CT-RATE `.nii.gz` 数据目录 |
| `dataset/RadGenome-ChestCT/` | RadGenome `.nii.gz` 数据目录 |

可选（Stage 5 LLM 裁判，需先在 HuggingFace 申请 Llama-3 访问权限）：

```bash
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir models/Llama-3.1-8B-Instruct
```

如果需要重新生成随机 `450/450` 数据与 manifest：

```bash
source /data/ProveTok_ACM/miniconda3/etc/profile.d/conda.sh
conda activate /data/ProveTok_ACM/miniconda3/envs/provetok
bash Scripts/run_random_450_download.sh
```

### 3. 直接运行

运行脚本已经固定使用项目内路径，不需要再手改 `CTRATE_CSV / RADGENOME_CSV / ENCODER_CKPT`：

```bash
source /data/ProveTok_ACM/miniconda3/etc/profile.d/conda.sh
conda activate /data/ProveTok_ACM/miniconda3/envs/provetok

# 推荐：使用优化配置运行 200/200 验证
bash Scripts/run_stage0_5_llama_200.sh

# 或者：完整 450/450 实验（需更新配置，见下方）
bash Scripts/run_stage0_5_llama_server.sh

# 分析结果
python analyze_outputs.py \
  --out_dir outputs/stage0_5_llama_200 \
  --expected_cases_map ctrate=200,radgenome=200
```

### 4. 优化配置（推荐）

基于 mediastinum 阈值 sweep 实验结果，推荐使用以下配置：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--tau_iou` | **0.04** | 解决 mediastinum 100% 违规问题的临界值 |
| `--token_budget_b` | **128** | 标准分辨率（128³），更高精度 |
| `--r2_min_support_ratio` | 0.8 | R2_ANATOMY 最小支持率 |
| `--llm_judge` | huggingface | 使用 Llama-3.1-8B 过滤误报 |
| `--llm_judge_alpha` | 0.5 | LLM 确认阈值 |

**关键改进**：
- ✅ `tau_iou=0.04` 完全消除 R2_ANATOMY 违规（相比 0.05 减少 100%）
- ✅ mediastinum 违规率从 100% 降至 0%
- ✅ 整体违规率稳定在 ~10%（健康范围）
- ✅ LLM Judge 过滤 ~50% 误报

详细实验结果见 [`EXPERIMENT_RESULTS.md`](./EXPERIMENT_RESULTS.md)

## 系统架构

| 阶段 | 说明 |
|------|------|
| Stage 0-2 | SwinUNETR 编码 CT 图像，构建空间 token bank |
| Stage 3 | Router：用报告句子查询 top-k tokens（teacher-forcing） |
| Stage 4 | Verifier：R1 侧位 / R2 解剖 IoU / R3 深度 / R5 否定一致性检查 |
| Stage 5 | LLM 裁判（可选）：对 Stage 4 违规做二次确认，过滤误报 |
| W_proj | InfoNCE 训练的线性投影（Stage 3c 前置） |

## 输出目录

```
outputs/
  stage0_4_450/          # Stage 0-4 baseline 结果
  stage0_5_llama_450/    # Stage 0-5 + LLM 裁判结果
  wprojection/           # W_proj 训练产物 (w_proj.pt)
```

## 文档

- [RUN_GUIDE_CN.md](./RUN_GUIDE_CN.md) — 完整运行说明（中文）
- [CHANGELOG_RECENT_CN.md](./CHANGELOG_RECENT_CN.md) — 近期改动记录
