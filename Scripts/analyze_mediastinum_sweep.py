#!/usr/bin/env python3
"""
分析 Mediastinum Tau Sweep 的结果
专注于 mediastinum 解剖结构在不同 tau_iou 阈值下的表现

用法:
  python Scripts/analyze_mediastinum_sweep.py \
    --sweep_root outputs/mediastinum_tau_sweep_50
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.0)


def parse_run_mediastinum_stats(run_dir: Path) -> dict:
    """解析单个运行目录中 mediastinum 的统计数据"""

    # 读取 run_meta 获取参数
    run_meta_path = run_dir / "run_meta.json"
    tau_iou = None
    if run_meta_path.exists():
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        tau_iou = run_meta.get("tau_iou")

    # 统计变量
    mediastinum_sentences = 0
    mediastinum_violations = 0
    mediastinum_r2_anatomy = 0
    mediastinum_llm_confirmed = 0

    total_sentences = 0
    total_violations = 0
    total_r2_anatomy = 0
    total_llm_confirmed = 0

    # 遍历所有 trace.jsonl 文件
    cases_dir = run_dir / "cases"
    if not cases_dir.exists():
        return {
            "run_dir": run_dir.name,
            "tau_iou": tau_iou,
            "mediastinum_sentences": 0,
            "mediastinum_violations": 0,
            "mediastinum_r2_anatomy": 0,
            "mediastinum_llm_confirmed": 0,
            "mediastinum_violation_rate": 0.0,
            "total_sentences": 0,
            "total_violation_rate": 0.0,
        }

    for trace_file in cases_dir.glob("*/*/trace.jsonl"):
        with trace_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)
                if obj.get("type") != "sentence":
                    continue

                total_sentences += 1

                anatomy = (obj.get("anatomy_keyword") or "").lower()
                vios = obj.get("violations") or []
                j5 = obj.get("stage5_judgements") or []

                # 统计总体违规
                if vios:
                    total_violations += 1

                # 统计 R2_ANATOMY
                r2_count = sum(
                    1 for v in vios
                    if isinstance(v, dict) and v.get("rule_id") == "R2_ANATOMY"
                )
                total_r2_anatomy += r2_count

                # 统计 LLM confirmed
                llm_confirmed = sum(
                    1 for j in j5
                    if isinstance(j, dict) and j.get("confirmed")
                )
                total_llm_confirmed += llm_confirmed

                # 如果是 mediastinum
                if "mediastinum" in anatomy:
                    mediastinum_sentences += 1

                    if vios:
                        mediastinum_violations += 1

                    mediastinum_r2_anatomy += r2_count
                    mediastinum_llm_confirmed += llm_confirmed

    return {
        "run_dir": run_dir.name,
        "tau_iou": tau_iou,
        "mediastinum_sentences": mediastinum_sentences,
        "mediastinum_violations": mediastinum_violations,
        "mediastinum_r2_anatomy": mediastinum_r2_anatomy,
        "mediastinum_llm_confirmed": mediastinum_llm_confirmed,
        "mediastinum_violation_rate": round(
            mediastinum_violations / max(1, mediastinum_sentences), 4
        ),
        "total_sentences": total_sentences,
        "total_violations": total_violations,
        "total_r2_anatomy": total_r2_anatomy,
        "total_llm_confirmed": total_llm_confirmed,
        "total_violation_rate": round(
            total_violations / max(1, total_sentences), 4
        ),
    }


def analyze_anatomy_breakdown(sweep_root: Path) -> dict[str, pd.DataFrame]:
    """分析每个 tau 下所有解剖结构的违规情况"""

    anatomy_data = defaultdict(lambda: {
        "tau_iou": None,
        "anatomy_stats": Counter(),
        "anatomy_violations": Counter(),
    })

    for run_dir in sorted(sweep_root.glob("tau*")):
        if not run_dir.is_dir():
            continue

        # 读取 tau_iou
        run_meta_path = run_dir / "run_meta.json"
        tau_iou = None
        if run_meta_path.exists():
            run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
            tau_iou = run_meta.get("tau_iou")

        anatomy_data[run_dir.name]["tau_iou"] = tau_iou

        # 统计每个解剖结构
        cases_dir = run_dir / "cases"
        if not cases_dir.exists():
            continue

        for trace_file in cases_dir.glob("*/*/trace.jsonl"):
            with trace_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    obj = json.loads(line)
                    if obj.get("type") != "sentence":
                        continue

                    anatomy = (obj.get("anatomy_keyword") or "").lower()
                    if not anatomy:
                        continue

                    anatomy_data[run_dir.name]["anatomy_stats"][anatomy] += 1

                    vios = obj.get("violations") or []
                    if vios:
                        anatomy_data[run_dir.name]["anatomy_violations"][anatomy] += 1

    # 转换为 DataFrame
    result = {}
    for run_name, data in anatomy_data.items():
        rows = []
        for anatomy in data["anatomy_stats"]:
            total = data["anatomy_stats"][anatomy]
            violations = data["anatomy_violations"][anatomy]
            rows.append({
                "anatomy": anatomy,
                "total_sentences": total,
                "violation_sentences": violations,
                "violation_rate": round(violations / max(1, total), 4),
            })

        df = pd.DataFrame(rows).sort_values("violation_rate", ascending=False)
        result[run_name] = df

    return result


def main():
    parser = argparse.ArgumentParser(
        description="分析 Mediastinum Tau Sweep 结果"
    )
    parser.add_argument(
        "--sweep_root", type=str, required=True,
        help="Sweep 输出根目录，如 outputs/mediastinum_tau_sweep_50"
    )
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root)
    if not sweep_root.exists():
        print(f"ERROR: sweep_root not found: {sweep_root}")
        return

    export_dir = sweep_root / "mediastinum_analysis"
    export_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    # Part 1: Mediastinum 专项统计
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  Mediastinum 专项统计")
    print("="*70)

    run_dirs = sorted([d for d in sweep_root.glob("tau*") if d.is_dir()])
    if not run_dirs:
        print("ERROR: No tau* directories found")
        return

    sweep_stats = []
    for run_dir in run_dirs:
        stats = parse_run_mediastinum_stats(run_dir)
        sweep_stats.append(stats)

    sweep_df = pd.DataFrame(sweep_stats).sort_values("tau_iou")
    print(sweep_df.to_string(index=False))

    # 保存 CSV
    csv_path = export_dir / "mediastinum_sweep_summary.csv"
    sweep_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved: {csv_path}")

    # ═══════════════════════════════════════════════════════════
    # Part 2: 可视化 - Mediastinum 违规率曲线
    # ═══════════════════════════════════════════════════════════
    if len(sweep_df) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Mediastinum violation rate
        ax = axes[0, 0]
        ax.plot(sweep_df["tau_iou"], sweep_df["mediastinum_violation_rate"],
                marker="o", linewidth=2, markersize=8, color="#E74C3C")
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="100%")
        ax.set_xlabel("tau_iou")
        ax.set_ylabel("Violation Rate")
        ax.set_title("Mediastinum Violation Rate")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 添加数值标签
        for _, row in sweep_df.iterrows():
            ax.text(row["tau_iou"], row["mediastinum_violation_rate"] + 0.02,
                   f"{row['mediastinum_violation_rate']:.3f}",
                   ha="center", fontsize=9)

        # 2. Mediastinum R2_ANATOMY count
        ax = axes[0, 1]
        ax.plot(sweep_df["tau_iou"], sweep_df["mediastinum_r2_anatomy"],
                marker="s", linewidth=2, markersize=8, color="#3498DB")
        ax.set_xlabel("tau_iou")
        ax.set_ylabel("R2_ANATOMY Count")
        ax.set_title("Mediastinum R2_ANATOMY Violations")
        ax.grid(True, alpha=0.3)

        for _, row in sweep_df.iterrows():
            ax.text(row["tau_iou"], row["mediastinum_r2_anatomy"] + 0.5,
                   f"{row['mediastinum_r2_anatomy']}",
                   ha="center", fontsize=9)

        # 3. Mediastinum LLM confirmed
        ax = axes[1, 0]
        ax.plot(sweep_df["tau_iou"], sweep_df["mediastinum_llm_confirmed"],
                marker="^", linewidth=2, markersize=8, color="#F39C12")
        ax.set_xlabel("tau_iou")
        ax.set_ylabel("LLM Confirmed Count")
        ax.set_title("Mediastinum LLM Judge Confirmed")
        ax.grid(True, alpha=0.3)

        for _, row in sweep_df.iterrows():
            ax.text(row["tau_iou"], row["mediastinum_llm_confirmed"] + 0.5,
                   f"{row['mediastinum_llm_confirmed']}",
                   ha="center", fontsize=9)

        # 4. 对比：Mediastinum vs Total violation rate
        ax = axes[1, 1]
        ax.plot(sweep_df["tau_iou"], sweep_df["mediastinum_violation_rate"],
                marker="o", linewidth=2, markersize=8, color="#E74C3C",
                label="Mediastinum")
        ax.plot(sweep_df["tau_iou"], sweep_df["total_violation_rate"],
                marker="o", linewidth=2, markersize=8, color="#27AE60",
                label="Overall")
        ax.set_xlabel("tau_iou")
        ax.set_ylabel("Violation Rate")
        ax.set_title("Mediastinum vs Overall Violation Rate")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.suptitle(f"Mediastinum Tau Sweep Analysis\n{sweep_root.name}",
                    fontsize=14, y=0.995)
        plt.tight_layout()

        fig_path = export_dir / "mediastinum_tau_curves.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Saved: {fig_path}")

    # ═══════════════════════════════════════════════════════════
    # Part 3: 解剖结构违规率分析
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  所有解剖结构违规率分析")
    print("="*70)

    anatomy_breakdown = analyze_anatomy_breakdown(sweep_root)

    # 找出 top 违规的解剖结构
    all_anatomy_violations = Counter()
    for run_name, df in anatomy_breakdown.items():
        for _, row in df.iterrows():
            all_anatomy_violations[row["anatomy"]] += row["violation_sentences"]

    top_anatomy = [k for k, v in all_anatomy_violations.most_common(10)]

    # 为每个 tau 生成报告
    for run_name in sorted(anatomy_breakdown.keys()):
        df = anatomy_breakdown[run_name]
        print(f"\n{run_name}:")
        print(f"  Top 10 违规解剖结构:")
        print(df.head(10).to_string(index=False, max_colwidth=30))

        # 保存完整 CSV
        csv_path = export_dir / f"{run_name}_anatomy_breakdown.csv"
        df.to_csv(csv_path, index=False)

    # 绘制 top anatomy 在不同 tau 下的变化
    if top_anatomy and len(anatomy_breakdown) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))

        # 准备数据
        tau_values = []
        anatomy_rates = {anat: [] for anat in top_anatomy}

        for run_name in sorted(anatomy_breakdown.keys()):
            df = anatomy_breakdown[run_name]

            # 获取 tau_iou
            tau = None
            run_meta_path = sweep_root / run_name / "run_meta.json"
            if run_meta_path.exists():
                run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
                tau = run_meta.get("tau_iou")

            if tau is not None:
                tau_values.append(tau)

                for anat in top_anatomy:
                    row = df[df["anatomy"] == anat]
                    if not row.empty:
                        anatomy_rates[anat].append(row.iloc[0]["violation_rate"])
                    else:
                        anatomy_rates[anat].append(0.0)

        # 绘制
        colors = sns.color_palette("husl", len(top_anatomy))
        for anat, color in zip(top_anatomy, colors):
            if len(anatomy_rates[anat]) == len(tau_values):
                ax.plot(tau_values, anatomy_rates[anat],
                       marker="o", label=anat, linewidth=1.5, color=color)

        ax.set_xlabel("tau_iou")
        ax.set_ylabel("Violation Rate")
        ax.set_title("Top 10 Anatomy Violation Rates Across Tau")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_path = export_dir / "top_anatomy_tau_curves.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n✓ Saved: {fig_path}")

    # ═══════════════════════════════════════════════════════════
    # 总结
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  分析完成")
    print("="*70)
    print(f"所有结果保存在: {export_dir}")
    print("\n关键文件:")
    print(f"  - mediastinum_sweep_summary.csv     (Mediastinum 汇总)")
    print(f"  - mediastinum_tau_curves.png        (Mediastinum 可视化)")
    print(f"  - tau*_anatomy_breakdown.csv        (各 tau 解剖结构详情)")
    print(f"  - top_anatomy_tau_curves.png        (Top 10 解剖结构对比)")

    # 找出最优 tau
    best_idx = sweep_df["mediastinum_violation_rate"].idxmin()
    best_row = sweep_df.loc[best_idx]
    print("\n推荐配置 (基于 mediastinum_violation_rate 最低):")
    print(f"  tau_iou = {best_row['tau_iou']}")
    print(f"  mediastinum_violation_rate = {best_row['mediastinum_violation_rate']:.4f}")
    print(f"  mediastinum_r2_anatomy = {best_row['mediastinum_r2_anatomy']}")
    print(f"  mediastinum_llm_confirmed = {best_row['mediastinum_llm_confirmed']}")


if __name__ == "__main__":
    main()
