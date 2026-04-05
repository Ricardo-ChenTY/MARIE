#!/usr/bin/env python3
"""
MARIE Stage0-4 Output Analysis  (server / local, no Colab)

用法:
  python analyze_outputs.py --out_dir /path/to/outputs_stage0_4_450

  python analyze_outputs.py --out_dir /path/to/outputs_stage0_5_llama_450

  python analyze_outputs.py --mode sweep --sweep_root /path/to/sweeps

图片保存到 {out_dir}/analysis_exports/  (matplotlib Agg backend, 无需 display)
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120

LOCKED_CONFIG = {
    "cp_strict":               True,
    "r2_mode":                 "ratio",
    "r2_min_support_ratio":    0.8,
    "tau_iou":                 0.05,
    "r2_skip_bilateral":       True,
    "r1_negation_exempt":      True,
    "r1_skip_midline":         True,
    "r1_min_same_side_ratio":  0.6,
    "anatomy_spatial_routing": True,
    "r4_disabled":             True,
    "r5_fallback_disabled":    True,
}



def _banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path.name}")


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"  [saved] {path.name}")


def _parse_expected_cases_map(s: str) -> dict[str, int]:
    """'ctrate=450,radgenome=450' -> {'ctrate': 450, 'radgenome': 450}"""
    result = {}
    for part in s.split(","):
        k, v = part.strip().split("=")
        result[k.strip()] = int(v.strip())
    return result



def run_validation(out_path: Path, expected_cases_map: str) -> None:
    _banner("1. 结构验收（validate_stage0_4_outputs.py）")
    script = Path(__file__).parent / "validate_stage0_4_outputs.py"
    if not script.exists():
        print("  [SKIP] validate_stage0_4_outputs.py not found")
        return

    datasets = ",".join(_parse_expected_cases_map(expected_cases_map).keys())
    result = subprocess.run(
        [
            sys.executable, str(script),
            "--out_dir", str(out_path),
            "--datasets", datasets,
            "--expected_cases_map", expected_cases_map,
            "--save_report", str(out_path / "validation_report.json"),
        ],
        capture_output=True, text=True,
    )
    out = result.stdout + result.stderr
    print(out[-4000:] if len(out) > 4000 else out)


def check_run_meta(out_path: Path, expected_cases_map: str) -> dict:
    _banner("2. run_meta.json 验收（锁定配置核对）")
    rmp = out_path / "run_meta.json"
    vrp = out_path / "validation_report.json"

    run_meta: dict = {}
    if rmp.exists():
        run_meta = json.loads(rmp.read_text(encoding="utf-8"))
        print("run_meta.json found:")
        for k, v in run_meta.items():
            if not isinstance(v, dict):
                print(f"  {k}: {v}")
    else:
        print("  [WARN] run_meta.json not found")

    validation_rows: list = []
    if vrp.exists():
        validation_rows = json.loads(vrp.read_text(encoding="utf-8"))
        failed = sum(1 for row in validation_rows if not row.get("passed", False))
        print(f"\nvalidation_report.json: {len(validation_rows)} rows, {failed} failed")
    else:
        print("  [WARN] validation_report.json not found")

    checks = []
    for key, expected in LOCKED_CONFIG.items():
        val = run_meta.get(key)
        if isinstance(expected, float):
            passed = val is not None and abs(float(val) - expected) < 1e-9
        elif isinstance(expected, bool):
            passed = val is True if expected else val is False
        else:
            passed = val == expected
        checks.append({"check": f"{key} == {expected}", "passed": passed})

    ecm = _parse_expected_cases_map(expected_cases_map)
    for ds, n in ecm.items():
        meta = run_meta.get(ds, {})
        if not isinstance(meta, dict):
            meta = {}
        for field in ("selected_rows", "processed_rows"):
            checks.append({
                "check": f"{ds}.{field} == {n}",
                "passed": meta.get(field) == n,
            })

    if validation_rows:
        checks.append({
            "check": "validation_report all passed",
            "passed": all(r.get("passed", False) for r in validation_rows),
        })

    status_df = pd.DataFrame(checks)
    pass_count = int(status_df["passed"].sum())
    total = len(status_df)
    print(f"\n{'PASS' if pass_count == total else 'FAIL'}  ({pass_count}/{total} checks)")
    failed_checks = status_df[~status_df["passed"]]
    if not failed_checks.empty:
        print("Failed checks:")
        for _, row in failed_checks.iterrows():
            print(f"  ✗ {row['check']}")
    else:
        print("All checks passed ✓")

    return run_meta



def analyze_summary(out_path: Path, export_dir: Path, expected_cases_map: str) -> pd.DataFrame:
    _banner("3. summary.csv 分析")
    summary_csv = out_path / "summary.csv"
    if not summary_csv.exists():
        print(f"  [SKIP] summary.csv not found: {summary_csv}")
        return pd.DataFrame()

    summary = pd.read_csv(summary_csv)
    print(f"总行数: {len(summary)}")
    expected_total = sum(_parse_expected_cases_map(expected_cases_map).values())
    if len(summary) != expected_total:
        print(f"  [WARN] expected {expected_total} rows, got {len(summary)}")

    agg_cols = {
        "case_id":     ("cases",            "count"),
        "n_tokens":    ("avg_tokens",        "mean"),
        "n_sentences": ("avg_sentences",     "mean"),
        "n_violations":("avg_violations",    "mean"),
    }
    extra = {}
    if "n_judge_confirmed" in summary.columns:
        extra["n_judge_confirmed"] = ("total_judge_confirmed", "sum")

    agg_dict: dict = {v[0]: (k, v[1]) for k, v in agg_cols.items()}
    agg_dict["total_violations"] = ("n_violations", "sum")
    if extra:
        agg_dict.update({v[0]: (k, v[1]) for k, v in extra.items()})

    agg = summary.groupby("dataset", as_index=False).agg(**agg_dict)
    agg["avg_vio_per_sent"] = (
        agg["avg_violations"] / agg["avg_sentences"].clip(lower=0.01)
    ).round(3)
    agg = agg.round({"avg_tokens": 1, "avg_sentences": 2, "avg_violations": 3})
    print(agg.to_string(index=False))

    metrics = [
        ("avg_tokens",     "Avg Tokens / Case"),
        ("avg_sentences",  "Avg Sentences / Case"),
        ("avg_violations", "Avg Violations / Case"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = ["#4C72B0", "#DD8452"]
    for ax, (col, title) in zip(axes, metrics):
        bars = ax.bar(agg["dataset"], agg[col], color=colors[: len(agg)])
        ax.set_title(title)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=10,
            )
    plt.suptitle(f"Run: {out_path.name}", fontsize=11)
    plt.tight_layout()
    _save_fig(fig, export_dir / "summary_bars.png")

    _save_csv(agg, export_dir / "dataset_aggregate.csv")
    return agg



def parse_traces(out_path: Path, export_dir: Path):
    _banner("4. Trace 解析（句子级）")
    trace_files = sorted((out_path / "cases").glob("*/*/trace.jsonl"))
    print(f"找到 {len(trace_files)} 个 trace.jsonl")

    rows = []
    rule_counter: Counter = Counter()

    for tf in trace_files:
        dataset = tf.parts[-3]
        case_id = tf.parts[-2]
        with tf.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("type") != "sentence":
                    continue
                vios = obj.get("violations") or []
                rule_ids = [
                    v["rule_id"]
                    for v in vios
                    if isinstance(v, dict) and "rule_id" in v
                ]
                for rid in rule_ids:
                    rule_counter[rid] += 1

                j5 = obj.get("stage5_judgements") or []
                n_confirmed = sum(
                    1 for j in j5 if isinstance(j, dict) and j.get("confirmed")
                )

                rows.append({
                    "dataset":          dataset,
                    "case_id":          case_id,
                    "sentence_index":   obj.get("sentence_index"),
                    "sentence_text":    obj.get("sentence_text", ""),
                    "anatomy_keyword":  obj.get("anatomy_keyword", ""),
                    "n_topk":           len(obj.get("topk_token_ids") or []),
                    "n_violations":     len(vios),
                    "has_violation":    int(len(vios) > 0),
                    "rule_ids":         "|".join(rule_ids),
                    "n_judge_confirmed": n_confirmed,
                })

    sent_df = pd.DataFrame(rows)
    if sent_df.empty:
        print("  [WARN] No sentence rows found")
        return sent_df, pd.DataFrame()

    print(f"解析了 {len(sent_df)} 句")
    vio_count = int(sent_df["has_violation"].sum())
    vio_rate  = sent_df["has_violation"].mean()
    print(f"  有违规的句子: {vio_count} ({vio_rate:.1%})")
    if sent_df["n_judge_confirmed"].sum() > 0:
        confirmed = int(sent_df["n_judge_confirmed"].sum())
        print(f"  LLM 确认违规句: {confirmed}")

    rule_df = pd.DataFrame([
        {"rule_id": k, "count": v}
        for k, v in sorted(rule_counter.items(), key=lambda x: -x[1])
    ])
    if not rule_df.empty:
        print("\n规则违规计数:")
        print(rule_df.to_string(index=False))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(rule_df["rule_id"], rule_df["count"],
               color=sns.color_palette("husl", len(rule_df)))
        ax.set_title(f"Violation Count by Rule  ({out_path.name})")
        ax.set_ylabel("count")
        for bar in ax.patches:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(int(bar.get_height())),
                ha="center", va="bottom",
            )
        plt.tight_layout()
        _save_fig(fig, export_dir / "rule_violation_count.png")

    sent_agg = sent_df.groupby("dataset", as_index=False).agg(
        sentences=("sentence_text", "count"),
        violation_sentences=("has_violation", "sum"),
        total_violations=("n_violations", "sum"),
    )
    sent_agg["violation_sentence_rate"] = (
        sent_agg["violation_sentences"] / sent_agg["sentences"].clip(lower=1)
    ).round(4)
    print("\n句子违规率:")
    print(sent_agg.to_string(index=False))

    anat_r2 = (
        sent_df[sent_df["rule_ids"].str.contains("R2_ANATOMY", na=False)]
        .groupby("anatomy_keyword")["case_id"]
        .count()
        .rename("R2_count")
        .reset_index()
        .sort_values("R2_count", ascending=False)
        .head(20)
    )
    if not anat_r2.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(anat_r2["anatomy_keyword"][::-1], anat_r2["R2_count"][::-1],
                color="#4C72B0")
        ax.set_title("Top Anatomy Keywords Triggering R2_ANATOMY")
        ax.set_xlabel("violation count")
        plt.tight_layout()
        _save_fig(fig, export_dir / "anatomy_r2.png")

    anat_all = sent_df.groupby("anatomy_keyword", as_index=False).agg(
        n_sentences=("sentence_text", "count"),
        n_violation_sentences=("has_violation", "sum"),
    )
    anat_all["violation_rate"] = (
        anat_all["n_violation_sentences"] / anat_all["n_sentences"].clip(lower=1)
    ).round(3)
    anat_all = anat_all.sort_values("violation_rate", ascending=False)

    _save_csv(sent_agg, export_dir / "sentence_violation_rate.csv")
    _save_csv(rule_df,  export_dir / "rule_violation_count.csv")
    _save_csv(sent_df,  export_dir / "sentence_detail.csv")
    if not anat_r2.empty:
        _save_csv(anat_r2,  export_dir / "anatomy_r2_breakdown.csv")
    _save_csv(anat_all, export_dir / "anatomy_all_violation_rate.csv")

    return sent_df, rule_df



def analyze_cases(sent_df: pd.DataFrame, out_path: Path, export_dir: Path) -> None:
    _banner("5. 病例级分析")
    if sent_df.empty:
        print("  [SKIP] No sentence data")
        return

    case_df = sent_df.groupby(["dataset", "case_id"], as_index=False).agg(
        n_sentences=("sentence_text", "count"),
        n_violation_sentences=("has_violation", "sum"),
        n_violations_total=("n_violations", "sum"),
    )
    case_df["violation_ratio"] = (
        case_df["n_violation_sentences"] / case_df["n_sentences"].clip(lower=1)
    ).round(4)
    case_df = case_df.sort_values(
        ["violation_ratio", "n_violations_total"], ascending=[False, False]
    )

    print(f"总病例数: {len(case_df)}")
    print(f"violation_ratio=1.0: {(case_df['violation_ratio'] == 1.0).sum()} 例")
    print(f"violation_ratio=0.0: {(case_df['violation_ratio'] == 0).sum()} 例")
    print("\n违规率最高的 20 例:")
    print(case_df.head(20).to_string(index=False))

    datasets_present = case_df["dataset"].unique().tolist()
    fig, axes = plt.subplots(1, len(datasets_present),
                             figsize=(6 * len(datasets_present), 4))
    if len(datasets_present) == 1:
        axes = [axes]
    pal = ["#4C72B0", "#DD8452"]
    for ax, ds, color in zip(axes, datasets_present, pal):
        sub = case_df[case_df["dataset"] == ds]["violation_ratio"]
        ax.hist(sub, bins=20, edgecolor="white", color=color, alpha=0.85)
        ax.axvline(sub.mean(), color="red", linestyle="--",
                   label=f"mean={sub.mean():.3f}")
        ax.set_title(f"{ds} — Case Violation Ratio")
        ax.set_xlabel("violation_ratio")
        ax.set_ylabel("# cases")
        ax.legend()
    plt.tight_layout()
    _save_fig(fig, export_dir / "case_violation_hist.png")

    _save_csv(case_df, export_dir / "abnormal_cases_ranked.csv")



def _compute_cost_metrics(out_path: Path) -> pd.DataFrame:
    """
    Parse traces to compute per-case cost metrics:
      - n_llm_gen_calls:  Stage 3c generation calls
      - n_llm_judge_calls: Stage 5 judge calls
      - C_LLM: total LLM calls (gen + judge + regen)
      - C_attn: attention cost proxy = sum(k * B) per sentence
                where k = top-k count, B = token bank size
    """
    trace_files = sorted((out_path / "cases").glob("*/*/trace.jsonl"))
    rows = []
    for tf in trace_files:
        dataset = tf.parts[-3]
        case_id = tf.parts[-2]
        B = 128  # default
        n_gen = 0
        n_judge = 0
        n_rerouted = 0
        n_despecified = 0
        n_sentences = 0
        total_k = 0

        with tf.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("type") == "case_meta":
                    B = obj.get("B", 128)
                    continue
                if obj.get("type") != "sentence":
                    continue
                n_sentences += 1
                topk = obj.get("topk_token_ids") or []
                total_k += len(topk)

                if obj.get("generated", False):
                    n_gen += 1
                j5 = obj.get("stage5_judgements") or []
                n_judge += len(j5)
                if obj.get("rerouted_citations"):
                    n_rerouted += 1
                    if obj.get("generated", False):
                        n_gen += 1
                stop = obj.get("stop_reason", "")
                if stop == "de_specified":
                    n_despecified += 1
                    n_gen += 1  # de-specify = one more gen call

        C_LLM = n_gen + n_judge
        C_attn = total_k * B  # k·B attention proxy

        rows.append({
            "dataset": dataset,
            "case_id": case_id,
            "n_sentences": n_sentences,
            "n_llm_gen_calls": n_gen,
            "n_llm_judge_calls": n_judge,
            "n_rerouted": n_rerouted,
            "n_despecified": n_despecified,
            "C_LLM": C_LLM,
            "C_attn": C_attn,
            "B": B,
        })
    return pd.DataFrame(rows)


def _bootstrap_ci(
    values: np.ndarray,
    R: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple:
    """Patient-level bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    n = len(values)
    boot_means = np.empty(R)
    for i in range(R):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = values[idx].mean()
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(values.mean()), float(lo), float(hi)


def _holm_correction(p_values: List[float], alpha: float = 0.05) -> List[dict]:
    """
    Holm step-down multiple comparison correction.
    Returns list of {p, p_adjusted, rejected} sorted by original order.
    """
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [None] * m
    max_adj = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (m - rank)
        adj = min(adj, 1.0)
        adj = max(adj, max_adj)  # enforce monotonicity
        max_adj = adj
        results[orig_idx] = {
            "p": p,
            "p_adjusted": round(adj, 6),
            "rejected": adj < alpha,
        }
    return results


def analyze_m5_protocol(
    out_path: Path,
    export_dir: Path,
    sent_df: pd.DataFrame,
) -> None:
    """M5 Statistical Protocol: cost accounting, bootstrap CI, Holm correction."""
    _banner("M5 Statistical Protocol")

    cost_df = _compute_cost_metrics(out_path)
    if cost_df.empty:
        print("  [SKIP] No trace data for cost computation")
        return

    print("=== LLM Cost Accounting ===")
    cost_agg = cost_df.groupby("dataset", as_index=False).agg(
        cases=("case_id", "count"),
        total_C_LLM=("C_LLM", "sum"),
        mean_C_LLM=("C_LLM", "mean"),
        total_C_attn=("C_attn", "sum"),
        mean_C_attn=("C_attn", "mean"),
        total_rerouted=("n_rerouted", "sum"),
        total_despecified=("n_despecified", "sum"),
    ).round(2)
    print(cost_agg.to_string(index=False))
    _save_csv(cost_agg, export_dir / "m5_cost_aggregate.csv")
    _save_csv(cost_df, export_dir / "m5_cost_per_case.csv")

    print("\n=== Bootstrap CI (R=5000, α=0.05) ===")
    boot_results = []
    for ds in cost_df["dataset"].unique():
        ds_sent = sent_df[sent_df["dataset"] == ds]
        if ds_sent.empty:
            continue
        case_vio_rate = ds_sent.groupby("case_id")["has_violation"].mean().values
        mean_val, lo, hi = _bootstrap_ci(case_vio_rate, R=5000)
        boot_results.append({
            "dataset": ds,
            "metric": "violation_rate",
            "mean": round(mean_val, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "n_patients": len(case_vio_rate),
        })
        print(f"  {ds}: violation_rate = {mean_val:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  (n={len(case_vio_rate)})")

        ds_cost = cost_df[cost_df["dataset"] == ds]["C_LLM"].values
        if len(ds_cost) > 0:
            mean_c, lo_c, hi_c = _bootstrap_ci(ds_cost, R=5000)
            boot_results.append({
                "dataset": ds,
                "metric": "C_LLM",
                "mean": round(mean_c, 2),
                "ci_lo": round(lo_c, 2),
                "ci_hi": round(hi_c, 2),
                "n_patients": len(ds_cost),
            })
            print(f"  {ds}: C_LLM = {mean_c:.2f}  95% CI [{lo_c:.2f}, {hi_c:.2f}]")

    boot_df = pd.DataFrame(boot_results)
    if not boot_df.empty:
        _save_csv(boot_df, export_dir / "m5_bootstrap_ci.csv")

    print("\n=== Holm Step-Down Correction ===")
    datasets = sorted(cost_df["dataset"].unique())
    if len(datasets) >= 2:
        from itertools import combinations
        p_values = []
        comparisons = []
        for ds_a, ds_b in combinations(datasets, 2):
            vals_a = sent_df[sent_df["dataset"] == ds_a].groupby("case_id")["has_violation"].mean().values
            vals_b = sent_df[sent_df["dataset"] == ds_b].groupby("case_id")["has_violation"].mean().values
            obs_diff = abs(vals_a.mean() - vals_b.mean())
            pooled = np.concatenate([vals_a, vals_b])
            rng = np.random.RandomState(42)
            n_a = len(vals_a)
            count_extreme = 0
            R_perm = 5000
            for _ in range(R_perm):
                rng.shuffle(pooled)
                perm_diff = abs(pooled[:n_a].mean() - pooled[n_a:].mean())
                if perm_diff >= obs_diff:
                    count_extreme += 1
            p_val = (count_extreme + 1) / (R_perm + 1)
            p_values.append(p_val)
            comparisons.append(f"{ds_a} vs {ds_b}")

        if p_values:
            holm_results = _holm_correction(p_values)
            holm_rows = []
            for comp, hr in zip(comparisons, holm_results):
                hr["comparison"] = comp
                holm_rows.append(hr)
                status = "REJECTED" if hr["rejected"] else "not rejected"
                print(f"  {comp}: p={hr['p']:.4f}, p_adj={hr['p_adjusted']:.4f} ({status})")
            holm_df = pd.DataFrame(holm_rows)
            _save_csv(holm_df, export_dir / "m5_holm_correction.csv")
    else:
        print("  [SKIP] Need ≥2 datasets for Holm correction")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868"][:len(datasets)]

    for i, ds in enumerate(datasets):
        vals = cost_df[cost_df["dataset"] == ds]["C_LLM"]
        axes[0].hist(vals, bins=15, alpha=0.6, label=ds, color=colors[i], edgecolor="white")
    axes[0].set_title("C_LLM per Case")
    axes[0].set_xlabel("LLM calls")
    axes[0].set_ylabel("# cases")
    axes[0].legend()

    for i, ds in enumerate(datasets):
        vals = cost_df[cost_df["dataset"] == ds]["C_attn"]
        axes[1].hist(vals, bins=15, alpha=0.6, label=ds, color=colors[i], edgecolor="white")
    axes[1].set_title("C_attn (Attention Cost Proxy) per Case")
    axes[1].set_xlabel("k × B")
    axes[1].set_ylabel("# cases")
    axes[1].legend()

    plt.tight_layout()
    _save_fig(fig, export_dir / "m5_cost_distributions.png")



def inspect_case_trace(dataset_name: str, case_id: str, root: Path) -> None:
    trace_path = root / "cases" / dataset_name / case_id / "trace.jsonl"
    if not trace_path.exists():
        print(f"Not found: {trace_path}")
        return

    with trace_path.open("r", encoding="utf-8") as f:
        data = [json.loads(ln) for ln in f if ln.strip()]

    meta = next((d for d in data if d.get("type") == "case_meta"), {})
    sentences = [d for d in data if d.get("type") == "sentence"]

    print(f"\n{'='*60}")
    print(f"Case: {case_id}  ({dataset_name})")
    print(
        f"B={meta.get('B')}  k={meta.get('k')}  "
        f"lambda={meta.get('lambda_spatial')}  tau_IoU={meta.get('tau_IoU')}"
    )
    print(f"{'='*60}")

    for s in sentences:
        vios = s.get("violations") or []
        status = "X" if vios else "OK"
        anat = s.get("anatomy_keyword", "-")
        text = s.get("sentence_text", "")
        print(f"\n  [{status}] #{s.get('sentence_index')} [{anat}]")
        print(f"       {text}")
        print(f"       cited: {s.get('topk_token_ids')}")
        for v in vios:
            if isinstance(v, dict):
                print(
                    f"       -> [{v.get('rule_id')}] "
                    f"severity={v.get('severity', 0):.3f}  "
                    f"{v.get('message', '')}"
                )
        j5 = s.get("stage5_judgements") or []
        for j in j5:
            if isinstance(j, dict):
                flag = "✓" if j.get("confirmed") else "✗"
                print(
                    f"       Stage5 {flag} [{j.get('rule_id')}] "
                    f"sev={j.get('adjusted_severity', 0):.2f}  "
                    f"{j.get('reasoning', '')[:80]}"
                )


def random_sample_inspect(dataset_name: str, n: int, root: Path) -> None:
    ds_dir = root / "cases" / dataset_name
    if not ds_dir.exists():
        print(f"Dataset dir not found: {ds_dir}")
        return
    all_cases = [d.name for d in ds_dir.iterdir() if d.is_dir()]
    if not all_cases:
        print(f"No cases in {ds_dir}")
        return
    sampled = random.sample(all_cases, min(n, len(all_cases)))
    print(f"\n### {dataset_name} — random {len(sampled)} cases ###")
    for case_id in sampled:
        inspect_case_trace(dataset_name, case_id, root)



def _parse_sweep_run(run_dir: Path) -> dict:
    rm: dict = {}
    rmp = run_dir / "run_meta.json"
    if rmp.exists():
        rm = json.loads(rmp.read_text(encoding="utf-8"))

    sentence_total = 0
    sentence_with_vio = 0
    rc: Counter = Counter()
    ds_sent: Counter = Counter()
    ds_vio: Counter = Counter()

    cases_dir = run_dir / "cases"
    if cases_dir.exists():
        for tf in cases_dir.glob("*/*/trace.jsonl"):
            ds = tf.parts[-3]
            with tf.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("type") != "sentence":
                        continue
                    sentence_total += 1
                    ds_sent[ds] += 1
                    vios = obj.get("violations") or []
                    if vios:
                        sentence_with_vio += 1
                        ds_vio[ds] += 1
                    for v in vios:
                        if isinstance(v, dict):
                            rc[str(v.get("rule_id", "UNKNOWN"))] += 1

    return {
        "run_dir":                  run_dir.name,
        "tau_iou":                  rm.get("tau_iou"),
        "r2_mode":                  rm.get("r2_mode", "ratio"),
        "r2_min_support_ratio":     rm.get("r2_min_support_ratio"),
        "r4_disabled":              rm.get("r4_disabled", False),
        "r5_fallback_disabled":     rm.get("r5_fallback_disabled", False),
        "sentence_total":           sentence_total,
        "violation_sentence_rate":  round(sentence_with_vio / max(1, sentence_total), 4),
        "ctrate_vio_rate":          round(ds_vio["ctrate"] / max(1, ds_sent["ctrate"]), 4),
        "radgenome_vio_rate":       round(ds_vio["radgenome"] / max(1, ds_sent["radgenome"]), 4),
        "R1_LATERALITY":            rc.get("R1_LATERALITY", 0),
        "R2_ANATOMY":               rc.get("R2_ANATOMY", 0),
        "R3_DEPTH":                 rc.get("R3_DEPTH", 0),
        "R4_SIZE":                  rc.get("R4_SIZE", 0),
        "R5_NEGATION":              rc.get("R5_NEGATION", 0),
        "violations_total":         sum(rc.values()),
    }


def analyze_sweep(sweep_root: Path, sweep_glob: str, export_dir: Path) -> None:
    _banner("Sweep 参数对比")
    run_dirs = sorted([p for p in sweep_root.glob(sweep_glob) if p.is_dir()])
    if not run_dirs:
        print(f"  [WARN] No run dirs found in {sweep_root} with glob={sweep_glob!r}")
        return
    print(f"Found {len(run_dirs)} runs:")
    for d in run_dirs:
        print(f"  {d.name}")

    sweep_rows = [_parse_sweep_run(d) for d in run_dirs]
    sweep_df = pd.DataFrame(sweep_rows).sort_values(
        "violation_sentence_rate", ascending=True
    )
    print(sweep_df.to_string(index=False))

    try:
        piv_rate = sweep_df.pivot_table(
            values="violation_sentence_rate",
            index="tau_iou", columns="r2_min_support_ratio", aggfunc="mean",
        )
        piv_r2 = sweep_df.pivot_table(
            values="R2_ANATOMY",
            index="tau_iou", columns="r2_min_support_ratio", aggfunc="sum",
        )
        if not piv_rate.empty and not piv_r2.empty:
            piv_rate.index   = [f"tau={v}" for v in piv_rate.index]
            piv_rate.columns = [f"ratio={v}" for v in piv_rate.columns]
            piv_r2.index     = [f"tau={v}" for v in piv_r2.index]
            piv_r2.columns   = [f"ratio={v}" for v in piv_r2.columns]

            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            sns.heatmap(piv_rate, annot=True, fmt=".3f", cmap="RdYlGn_r",
                        vmin=0.0, vmax=1.0, ax=axes[0], linewidths=0.5)
            axes[0].set_title("violation_sentence_rate\n(越低越好)")
            sns.heatmap(piv_r2, annot=True, fmt=".0f", cmap="YlOrRd",
                        ax=axes[1], linewidths=0.5)
            axes[1].set_title("R2_ANATOMY count\n(越低越好)")
            plt.suptitle("R2 Sweep: tau_iou × r2_min_support_ratio",
                         fontsize=13, y=1.03)
            plt.tight_layout()
            _save_fig(fig, export_dir / "sweep_heatmap.png")
    except Exception as e:
        print(f"  [WARN] heatmap failed: {e}")

    rule_cols = ["R1_LATERALITY", "R2_ANATOMY", "R4_SIZE", "R5_NEGATION"]
    labels = [
        f"tau={row.tau_iou}\nratio={row.r2_min_support_ratio}"
        for _, row in sweep_df.iterrows()
    ]
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2), 5))
    colors = sns.color_palette("husl", len(rule_cols))
    for i, (col, color) in enumerate(zip(rule_cols, colors)):
        ax.bar(x + i * width, sweep_df[col].values, width, label=col, color=color)
    ax.set_xticks(x + width * (len(rule_cols) - 1) / 2)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Rule Violation Count per Sweep Combination")
    ax.set_ylabel("count")
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, export_dir / "sweep_rule_bars.png")

    best = sweep_df.sort_values("violation_sentence_rate").iloc[0]
    print(f"\n最低 violation_sentence_rate: {best['violation_sentence_rate']:.4f}")
    print(f"  -> tau_iou={best['tau_iou']},  r2_min_support_ratio={best['r2_min_support_ratio']}")

    _save_csv(sweep_df, export_dir / "sweep_summary.csv")



def main() -> None:
    parser = argparse.ArgumentParser(
        description="MARIE Stage0-4/5 Output Analysis (server mode)"
    )
    parser.add_argument(
        "--out_dir", type=str,
        default=None,
        help="单次运行输出目录（single 模式）",
    )
    parser.add_argument(
        "--mode", choices=["single", "sweep"], default="single",
        help="single: 分析单次结果; sweep: 对比多组参数结果",
    )
    parser.add_argument(
        "--expected_cases_map", type=str, default="ctrate=450,radgenome=450",
        help="验收口径，如 ctrate=450,radgenome=450",
    )
    parser.add_argument(
        "--sweep_root", type=str, default=None,
        help="sweep 模式: 包含多个运行目录的父目录",
    )
    parser.add_argument(
        "--sweep_glob", type=str, default="*",
        help="sweep 模式: 用于匹配运行子目录的 glob 模式",
    )
    parser.add_argument(
        "--inspect_n", type=int, default=0,
        help="随机抽查 N 个病例（0=不抽查）",
    )
    parser.add_argument(
        "--inspect_case", type=str, default=None,
        help="查特定病例，格式: dataset/case_id",
    )
    args = parser.parse_args()

    if args.mode == "single":
        if args.out_dir is None:
            parser.error("--out_dir is required for single mode")
        out_path = Path(args.out_dir)
        if not out_path.exists():
            sys.exit(f"ERROR: out_dir not found: {out_path}")

        export_dir = out_path / "analysis_exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        run_validation(out_path, args.expected_cases_map)
        run_meta = check_run_meta(out_path, args.expected_cases_map)
        analyze_summary(out_path, export_dir, args.expected_cases_map)
        sent_df, _ = parse_traces(out_path, export_dir)
        analyze_cases(sent_df, out_path, export_dir)
        analyze_m5_protocol(out_path, export_dir, sent_df)

        if args.inspect_n > 0:
            _banner("随机病例抽查")
            for ds in ["ctrate", "radgenome"]:
                random_sample_inspect(ds, args.inspect_n, out_path)

        if args.inspect_case:
            parts = args.inspect_case.split("/", 1)
            if len(parts) == 2:
                inspect_case_trace(parts[0], parts[1], out_path)
            else:
                print(f"[WARN] --inspect_case should be dataset/case_id, got: {args.inspect_case}")

        _banner("完成")
        print(f"所有导出文件: {export_dir}")

    elif args.mode == "sweep":
        if args.sweep_root is None:
            parser.error("--sweep_root is required for sweep mode")
        sweep_root = Path(args.sweep_root)
        if not sweep_root.exists():
            sys.exit(f"ERROR: sweep_root not found: {sweep_root}")

        export_dir = sweep_root / "analysis_exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        analyze_sweep(sweep_root, args.sweep_glob, export_dir)
        _banner("完成")
        print(f"所有导出文件: {export_dir}")


if __name__ == "__main__":
    main()

