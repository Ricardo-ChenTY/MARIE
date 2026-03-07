# Colab 450/450 重跑指南

这份文档给协作者在 Google Colab 上直接重跑当前锁定的 `450/450` 主实验。

## 1. 当前锁定配置

主实验使用：

- `--cp_strict`
- `--r2_mode ratio`
- `--r2_min_support_ratio 0.8`
- `--tau_iou 0.05`
- `--anatomy_spatial_routing`
- `--r2_skip_bilateral`
- `--r4_disabled`
- `--r5_fallback_disabled`

这套配置已经在 `50/50` smoke 上验证过：

- `Validated: 100, Passed: 100, Failed: 0`
- 总违规句率：`42.00% -> 17.375%`
- `R2_ANATOMY: 239 -> 42`
- `R1_LATERALITY`: 不变

## 2. Colab 需要准备的文件

- `/content/drive/MyDrive/Data/manifests/ctrate_manifest.csv`
- `/content/drive/MyDrive/Data/manifests/radgenome_manifest.csv`
- `/content/drive/MyDrive/Data/checkpoints/swinunetr.ckpt`

输出目录建议：

- `/content/drive/MyDrive/Data/outputs_stage0_4_450_128`

## 3. 先做 ckpt 探测

```bash
!python Scripts/ckpt_probe.py \
  --ckpt_path /content/drive/MyDrive/Data/checkpoints/swinunetr.ckpt \
  --in_channels 1 \
  --out_channels 2 \
  --feature_size 48 \
  --save_report /content/drive/MyDrive/Data/ckpt_probe_report.json
```

只有 `compatible=true` 才继续主实验。

## 4. Colab 运行步骤

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content
!test -d ProveTok_ACM || git clone https://github.com/Ricardo-ChenTY/ProveTok_ACM.git
%cd /content/ProveTok_ACM
```

```bash
!pip install -U pip
!pip install numpy pandas scipy simpleitk monai "sentence-transformers>=3.0"
```

```bash
!python run_mini_experiment.py \
  --ctrate_csv /content/drive/MyDrive/Data/manifests/ctrate_manifest.csv \
  --radgenome_csv /content/drive/MyDrive/Data/manifests/radgenome_manifest.csv \
  --out_dir /content/drive/MyDrive/Data/outputs_stage0_4_450_128 \
  --max_cases 450 \
  --expected_cases_per_dataset 450 \
  --cp_strict \
  --encoder_ckpt /content/drive/MyDrive/Data/checkpoints/swinunetr.ckpt \
  --text_encoder semantic \
  --text_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
  --text_encoder_device cuda \
  --device cuda \
  --token_budget_b 64 \
  --k_per_sentence 8 \
  --lambda_spatial 0.3 \
  --tau_iou 0.05 \
  --beta 0.1 \
  --r2_mode ratio \
  --r2_min_support_ratio 0.8 \
  --r4_disabled \
  --r5_fallback_disabled \
  --anatomy_spatial_routing \
  --r2_skip_bilateral
```

## 5. 结构验收

```bash
!python validate_stage0_4_outputs.py \
  --out_dir /content/drive/MyDrive/Data/outputs_stage0_4_450_128 \
  --datasets ctrate,radgenome \
  --expected_cases_map ctrate=450,radgenome=450 \
  --save_report /content/drive/MyDrive/Data/outputs_stage0_4_450_128/validation_report.json
```

## 6. Colab 分析 notebook

分析使用：

- [OUTPUT_ANALYSIS_COLAB.ipynb](c:\Users\34228\Desktop\ACM\Smoke_analysis\OUTPUT_ANALYSIS_COLAB.ipynb)

操作：

1. 打开 notebook
2. 确认 `OUT_DIR = /content/drive/MyDrive/Data/outputs_stage0_4_450_128`
3. 运行全部 cells

notebook 会自动：

- 重跑结构验收
- 检查 `run_meta.json` 是否符合当前主实验配置
- 导出 `analysis_exports/*`

## 7. 结果达标标准

硬性要求：

- `Validated cases = 900`
- `Failed = 0`
- `cp_strict = true`
- `ctrate.selected_rows = 450`
- `radgenome.selected_rows = 450`

软性目标：

- 总体 `violation_sentence_rate` 明显低于旧基线 `0.721`
- `R2_ANATOMY` 明显低于旧基线 `2651`
- `R1_LATERALITY` 不明显反弹

## 8. 需要交付给分析同学的最小文件

- `summary.csv`
- `ctrate_case_summary.csv`
- `radgenome_case_summary.csv`
- `run_meta.json`
- `validation_report.json`
- `cases/*/*/trace.jsonl`
