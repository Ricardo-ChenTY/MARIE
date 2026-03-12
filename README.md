# ProveTok_ACM

中文运行说明：[`RUN_GUIDE_CN.md`](./RUN_GUIDE_CN.md)
数据集下载说明：[`Scripts/DOWNLOAD_GUIDE.md`](./Scripts/DOWNLOAD_GUIDE.md)

## 快速开始（服务器）

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据和模型权重

需要在本地准备以下文件（不在仓库中）：

| 文件 | 说明 |
|------|------|
| `ctrate_manifest.csv` | CT-RATE 数据集清单（含 case_id / volume_path / report_text 列） |
| `radgenome_manifest.csv` | RadGenome 数据集清单（同上） |
| `swinunetr.ckpt` | SwinUNETR CT 图像编码器权重 |

可选（Stage 5 LLM 裁判，需先在 HuggingFace 申请 Llama-3 访问权限）：

```bash
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir models/Llama-3.1-8B-Instruct
```

### 3. 填写路径并运行

```bash
# 编辑路径（改 CTRATE_CSV / RADGENOME_CSV / ENCODER_CKPT 三行）
vim Scripts/run_stage0_4_server.sh

# Step 1: Stage 0-4 baseline（纯规则，450/450）
bash Scripts/run_stage0_4_server.sh

# 验收结果
python analyze_outputs.py --out_dir outputs/stage0_4_450

# Step 2（可选）: Stage 0-5，加 LLM 裁判
vim Scripts/run_stage0_5_llama_server.sh  # 同样填路径
bash Scripts/run_stage0_5_llama_server.sh

# Step 3（可选）: 训练 W_proj
bash Scripts/run_wprojection_train.sh
```

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
