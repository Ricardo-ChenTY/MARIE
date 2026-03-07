# Recent Changes (CN)

更新日期：2026-03-07

## 本次更新做了什么

这次没有继续改主实验逻辑，重点是把文档和分析入口统一到 `450/450` 主实验口径。

### 1. 锁定 450/450 主实验配置

当前推荐配置固定为：

- `--cp_strict`
- `--r2_mode ratio`
- `--r2_min_support_ratio 0.8`
- `--tau_iou 0.05`
- `--anatomy_spatial_routing`
- `--r2_skip_bilateral`
- `--r4_disabled`
- `--r5_fallback_disabled`

锁定依据是最新 `50/50` smoke：

- `Validated: 100, Passed: 100, Failed: 0`
- 总违规句率：`42.00% -> 17.375%`
- `R2_ANATOMY: 239 -> 42`
- `R1_LATERALITY: 100 -> 100`

### 2. 更新运行说明

文件：

- [RUN_GUIDE_CN.md](c:\Users\34228\Desktop\ACM\RUN_GUIDE_CN.md)

更新内容：

- 改成以 `450/450` 主实验为中心
- 保留 Linux/Colab Shell 和 Windows PowerShell 两套命令
- 明确写出结构验收命令
- 明确写出 Colab notebook 怎么做验收和分析

### 3. 更新结果验收说明

文件：

- [OUTPUT_ANALYSIS_GUIDE_CN.md](c:\Users\34228\Desktop\ACM\OUTPUT_ANALYSIS_GUIDE_CN.md)

更新内容：

- 不再停留在旧 baseline 的描述
- 新增 `50/50` smoke 改善结果
- 新增 `450/450` 硬性验收标准
- 新增 `450/450` 软性目标区间

### 4. 更新 Colab 分析 notebook

文件：

- [OUTPUT_ANALYSIS_COLAB.ipynb](c:\Users\34228\Desktop\ACM\Smoke_analysis\OUTPUT_ANALYSIS_COLAB.ipynb)

更新目标：

- 默认切到 `450/450` 单次运行分析
- 结构验收强制 `expected_cases_map = ctrate=450,radgenome=450`
- 直接显示主实验状态检查
- 导出文件名与文档统一

## 现在你朋友该怎么做

5090 跑实验的人：

1. 按 [RUN_GUIDE_CN.md](c:\Users\34228\Desktop\ACM\RUN_GUIDE_CN.md) 的 `450/450` 主实验命令运行
2. 再运行同一份 guide 里的结构验收命令
3. 把结果目录交给分析同学

Colab 做分析的人：

1. 打开 [OUTPUT_ANALYSIS_COLAB.ipynb](c:\Users\34228\Desktop\ACM\Smoke_analysis\OUTPUT_ANALYSIS_COLAB.ipynb)
2. 把 `OUT_DIR` 改成这次 `450/450` 结果目录
3. 运行全部 cells
4. 根据 [OUTPUT_ANALYSIS_GUIDE_CN.md](c:\Users\34228\Desktop\ACM\OUTPUT_ANALYSIS_GUIDE_CN.md) 判断是否达标

## 当前判断

现在最重要的不是再改 bilateral 逻辑，而是先把 `450/450` 主实验跑出来并完成标准化验收。
