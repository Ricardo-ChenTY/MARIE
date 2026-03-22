#!/usr/bin/env python3
"""
ProveTok 5K Ablation — Statistical Significance & Error Analysis

Generates:
  1. Patient-level paired bootstrap CI (R=5000) for adjacent ablation configs
  2. Holm step-down correction across all pairwise comparisons
  3. Error analysis: violation breakdown by anatomy, rule distribution, repair rate
  4. 180-case vs 5K consistency comparison

Output: outputs/paper_figures_5k/statistical_analysis/
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.05)

ROOT = Path(__file__).resolve().parent.parent
ABLATION_DIR = ROOT / "outputs" / "5k_ablation"
OUT_DIR = ROOT / "outputs" / "paper_figures_5k" / "statistical_analysis"

# Ablation chain order (canonical)
CONFIGS = [
    ("A0", "A0_identity_spatial"),
    ("A1", "A1_trained_spatial"),
    ("E1", "E1_spatial_filter_rerank"),
    ("B2'", "B2_evcard_v1"),
    ("B2'v2", "B2_evcard_v2"),
    ("C2'", "C2_evcard_v1_judge"),
    ("D2", "D2_repair"),
]

ADJACENT_PAIRS = [
    ("A0", "A1"),
    ("A1", "E1"),
    ("E1", "B2'"),
    ("B2'", "B2'v2"),
    ("B2'v2", "C2'"),
    ("C2'", "D2"),
]

RULE_IDS = ["R1_LATERALITY", "R2_ANATOMY", "R3_DEPTH", "R6a_PRESENCE", "R6b_PRESENCE"]


# ═══════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════

def load_all_traces() -> Dict[str, pd.DataFrame]:
    """Load sentence-level trace data for all configs. Returns {label: DataFrame}."""
    all_data = {}
    for label, dirname in CONFIGS:
        rows = []
        cases_dir = ABLATION_DIR / dirname / "cases"
        if not cases_dir.exists():
            print(f"  [WARN] Missing: {cases_dir}")
            continue
        for ds_dir in sorted(cases_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            dataset = ds_dir.name
            for case_dir in sorted(ds_dir.iterdir()):
                trace_path = case_dir / "trace.jsonl"
                if not trace_path.exists():
                    continue
                case_id = case_dir.name
                with trace_path.open(encoding="utf-8") as f:
                    for line in f:
                        d = json.loads(line.strip())
                        if d.get("type") != "sentence":
                            continue
                        vios = d.get("violations") or []
                        rule_counts = Counter(
                            v.get("rule_id", "UNKNOWN") for v in vios if isinstance(v, dict)
                        )
                        rows.append({
                            "dataset": dataset,
                            "case_id": case_id,
                            "sentence_index": d.get("sentence_index"),
                            "anatomy_keyword": d.get("anatomy_keyword"),
                            "generated": d.get("generated", False),
                            "has_violation": len(vios) > 0,
                            "n_violations": len(vios),
                            "R1": rule_counts.get("R1_LATERALITY", 0),
                            "R2": rule_counts.get("R2_ANATOMY", 0),
                            "R3": rule_counts.get("R3_DEPTH", 0),
                            "R6a": rule_counts.get("R6a_PRESENCE", 0),
                            "R6b": rule_counts.get("R6b_CROSS_PRESENCE", 0),
                            "evidence_card": d.get("evidence_card"),
                            "stage5_judgements": d.get("stage5_judgements"),
                        })
        df = pd.DataFrame(rows)
        print(f"  {label:8s}: {len(df):5d} sentences, {df['case_id'].nunique():3d} cases")
        all_data[label] = df
    return all_data


# ═══════════════════════════════════════════════════════════
# Bootstrap CI & Permutation Test
# ═══════════════════════════════════════════════════════════

def bootstrap_ci(
    values: np.ndarray, R: int = 5000, alpha: float = 0.05, seed: int = 42
) -> Tuple[float, float, float]:
    """Patient-level bootstrap confidence interval. Returns (mean, ci_lo, ci_hi)."""
    rng = np.random.RandomState(seed)
    n = len(values)
    boot_means = np.array([values[rng.randint(0, n, size=n)].mean() for _ in range(R)])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(values.mean()), float(lo), float(hi)


def paired_permutation_test(
    vals_a: np.ndarray, vals_b: np.ndarray, R: int = 5000, seed: int = 42
) -> float:
    """Two-sample permutation test for difference in means. Returns p-value."""
    obs_diff = abs(vals_a.mean() - vals_b.mean())
    pooled = np.concatenate([vals_a, vals_b])
    n_a = len(vals_a)
    rng = np.random.RandomState(seed)
    count_extreme = 0
    for _ in range(R):
        rng.shuffle(pooled)
        perm_diff = abs(pooled[:n_a].mean() - pooled[n_a:].mean())
        if perm_diff >= obs_diff:
            count_extreme += 1
    return (count_extreme + 1) / (R + 1)


def holm_correction(p_values: List[float], alpha: float = 0.05) -> List[dict]:
    """Holm step-down correction. Returns list of {p, p_adjusted, rejected}."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [None] * m
    max_adj = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(p * (m - rank), 1.0)
        adj = max(adj, max_adj)
        max_adj = adj
        results[orig_idx] = {
            "p": round(p, 6),
            "p_adjusted": round(adj, 6),
            "rejected": adj < alpha,
        }
    return results


# ═══════════════════════════════════════════════════════════
# 1. Statistical Significance Tests
# ═══════════════════════════════════════════════════════════

def run_significance_tests(all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run paired bootstrap CI + permutation tests for adjacent ablation pairs."""
    print("\n" + "=" * 60)
    print("  1. Statistical Significance Tests")
    print("=" * 60)

    # Compute patient-level violation rate for each config
    def case_viol_rate(df: pd.DataFrame) -> pd.Series:
        return df.groupby("case_id")["has_violation"].mean()

    # Bootstrap CI for each config
    ci_rows = []
    for label, _ in CONFIGS:
        if label not in all_data:
            continue
        df = all_data[label]
        cvr = case_viol_rate(df).values
        mean_val, lo, hi = bootstrap_ci(cvr)
        ci_rows.append({
            "config": label,
            "metric": "viol_rate",
            "mean": round(mean_val * 100, 2),
            "ci_lo": round(lo * 100, 2),
            "ci_hi": round(hi * 100, 2),
            "n_patients": len(cvr),
        })
        print(f"  {label:8s}: Viol% = {mean_val*100:.2f}%  95% CI [{lo*100:.2f}%, {hi*100:.2f}%]  (n={len(cvr)})")

    ci_df = pd.DataFrame(ci_rows)

    # Pairwise permutation tests for adjacent configs
    print("\n--- Adjacent Pairwise Comparisons (Viol%) ---")
    perm_rows = []
    p_values = []
    for cfg_a, cfg_b in ADJACENT_PAIRS:
        if cfg_a not in all_data or cfg_b not in all_data:
            continue
        vals_a = case_viol_rate(all_data[cfg_a]).values
        vals_b = case_viol_rate(all_data[cfg_b]).values
        p_val = paired_permutation_test(vals_a, vals_b)
        diff = vals_b.mean() - vals_a.mean()
        p_values.append(p_val)
        perm_rows.append({
            "comparison": f"{cfg_a} → {cfg_b}",
            "mean_A": round(vals_a.mean() * 100, 2),
            "mean_B": round(vals_b.mean() * 100, 2),
            "diff": round(diff * 100, 2),
            "p_value": round(p_val, 4),
        })

    # Holm correction
    if p_values:
        holm_results = holm_correction(p_values)
        for row, hr in zip(perm_rows, holm_results):
            row["p_adjusted"] = hr["p_adjusted"]
            row["significant"] = hr["rejected"]
            sig = "*" if hr["rejected"] else ""
            print(f"  {row['comparison']:16s}: Δ={row['diff']:+.2f}pp  p={row['p_value']:.4f}  p_adj={hr['p_adjusted']:.4f} {sig}")

    perm_df = pd.DataFrame(perm_rows)
    return ci_df, perm_df


# ═══════════════════════════════════════════════════════════
# 2. Error Analysis
# ═══════════════════════════════════════════════════════════

def run_error_analysis(all_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Violation breakdown by anatomy, rule distribution, per-config."""
    print("\n" + "=" * 60)
    print("  2. Error Analysis")
    print("=" * 60)

    # --- 2a. Rule distribution per config ---
    print("\n--- Rule Distribution per Config ---")
    rule_rows = []
    for label, _ in CONFIGS:
        if label not in all_data:
            continue
        df = all_data[label]
        n = len(df)
        rule_rows.append({
            "config": label,
            "n_sentences": n,
            "R1": int(df["R1"].sum()),
            "R3": int(df["R3"].sum()),
            "R6b": int(df["R6b"].sum()),
            "R2": int(df["R2"].sum()),
            "R6a": int(df["R6a"].sum()),
            "total_violations": int(df["n_violations"].sum()),
            "viol_rate_%": round(df["has_violation"].mean() * 100, 2),
        })
    rule_df = pd.DataFrame(rule_rows)
    print(rule_df.to_string(index=False))

    # --- 2b. Violation breakdown by anatomy keyword (best config: B2'v2) ---
    print("\n--- Violation Breakdown by Anatomy (B2'v2) ---")
    target_label = "B2'v2"
    if target_label in all_data:
        df = all_data[target_label]
        anat_rows = []
        for anat, grp in df.groupby("anatomy_keyword", dropna=False):
            anat_name = anat if anat else "(no anatomy)"
            n_sent = len(grp)
            n_viol = int(grp["has_violation"].sum())
            anat_rows.append({
                "anatomy": anat_name,
                "n_sentences": n_sent,
                "n_violated": n_viol,
                "viol_rate_%": round(n_viol / n_sent * 100, 1) if n_sent > 0 else 0,
                "R1": int(grp["R1"].sum()),
                "R3": int(grp["R3"].sum()),
                "R6b": int(grp["R6b"].sum()),
            })
        anat_df = pd.DataFrame(anat_rows).sort_values("n_violated", ascending=False)
        print(anat_df.head(20).to_string(index=False))
    else:
        anat_df = pd.DataFrame()

    # --- 2c. Per-dataset breakdown ---
    print("\n--- Per-Dataset Violation Rates ---")
    ds_rows = []
    for label, _ in CONFIGS:
        if label not in all_data:
            continue
        df = all_data[label]
        for ds, grp in df.groupby("dataset"):
            n = len(grp)
            n_viol = int(grp["has_violation"].sum())
            ds_rows.append({
                "config": label,
                "dataset": ds,
                "n_sentences": n,
                "n_violated": n_viol,
                "viol_rate_%": round(n_viol / n * 100, 2) if n > 0 else 0,
            })
    ds_df = pd.DataFrame(ds_rows)
    # Pivot for readability
    if not ds_df.empty:
        pivot = ds_df.pivot_table(
            values="viol_rate_%", index="config", columns="dataset"
        )
        print(pivot.to_string())

    return rule_df, anat_df, ds_df


# ═══════════════════════════════════════════════════════════
# 3. D2 Repair Analysis
# ═══════════════════════════════════════════════════════════

def run_repair_analysis(all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Analyze D2 repair effectiveness by comparing C2' vs D2 at sentence level."""
    print("\n" + "=" * 60)
    print("  3. D2 Repair Effectiveness")
    print("=" * 60)

    if "C2'" not in all_data or "D2" not in all_data:
        print("  [SKIP] Need both C2' and D2 data")
        return pd.DataFrame()

    c2 = all_data["C2'"].set_index(["dataset", "case_id", "sentence_index"])
    d2 = all_data["D2"].set_index(["dataset", "case_id", "sentence_index"])

    # Align on same sentences
    common = c2.index.intersection(d2.index)
    c2c = c2.loc[common]
    d2c = d2.loc[common]

    # Sentences that had violations in C2'
    had_viol = c2c["has_violation"]
    n_had_viol = int(had_viol.sum())

    # Of those, how many were fixed in D2?
    fixed = had_viol & ~d2c["has_violation"]
    n_fixed = int(fixed.sum())

    # New violations introduced by D2
    new_viol = ~c2c["has_violation"] & d2c["has_violation"]
    n_new = int(new_viol.sum())

    # Per-rule repair
    repair_rows = []
    for rule in ["R1", "R3", "R6b"]:
        c2_count = int(c2c[rule].sum())
        d2_count = int(d2c[rule].sum())
        delta = d2_count - c2_count
        repair_rows.append({
            "rule": rule,
            "C2_count": c2_count,
            "D2_count": d2_count,
            "delta": delta,
            "reduction_%": round((1 - d2_count / max(c2_count, 1)) * 100, 1),
        })

    repair_df = pd.DataFrame(repair_rows)

    print(f"  Sentences with violations in C2': {n_had_viol}")
    print(f"  Fixed by D2 repair:               {n_fixed} ({n_fixed/max(n_had_viol,1)*100:.1f}%)")
    print(f"  New violations introduced:         {n_new}")
    print(f"\n  Per-rule changes (C2' → D2):")
    print(repair_df.to_string(index=False))

    return repair_df


# ═══════════════════════════════════════════════════════════
# 4. 180-case vs 5K Consistency
# ═══════════════════════════════════════════════════════════

def run_consistency_check() -> Optional[pd.DataFrame]:
    """Compare 180-case and 5K ablation results if both are available."""
    print("\n" + "=" * 60)
    print("  4. 180-case vs 5K Consistency")
    print("=" * 60)

    # Load 5K table2 data
    t2_5k_path = ROOT / "outputs" / "paper_figures_5k" / "table2_data.json"
    if not t2_5k_path.exists():
        print("  [SKIP] 5K table2_data.json not found")
        return None
    with open(t2_5k_path) as f:
        t2_5k = json.load(f)

    # Load 180 table2 data
    t2_180_path = ROOT / "outputs" / "paper_figures" / "table2_data.json"
    if not t2_180_path.exists():
        print("  [SKIP] 180-case table2_data.json not found")
        return None
    with open(t2_180_path) as f:
        t2_180 = json.load(f)

    # Build comparison
    map_5k = {r["label"]: r for r in t2_5k}
    map_180 = {r["label"]: r for r in t2_180}

    rows = []
    for label in ["A0", "A1", "E1", "B2'", "B2'v2", "C2'", "D2"]:
        r5k = map_5k.get(label, {})
        r180 = map_180.get(label, {})
        if not r5k or not r180:
            continue
        rows.append({
            "config": label,
            "viol_180": r180.get("viol_rate"),
            "viol_5k": r5k.get("viol_rate"),
            "R1_180": r180.get("R1"),
            "R1_5k": r5k.get("R1"),
            "R3_180": r180.get("R3"),
            "R3_5k": r5k.get("R3"),
            "R6b_180": r180.get("R6b"),
            "R6b_5k": r5k.get("R6b"),
        })

    if not rows:
        print("  [SKIP] No matching configs found")
        return None

    comp_df = pd.DataFrame(rows)
    print(comp_df.to_string(index=False))

    # Rank correlation
    if len(comp_df) >= 3:
        from scipy import stats
        tau, p_tau = stats.kendalltau(comp_df["viol_180"], comp_df["viol_5k"])
        rho, p_rho = stats.spearmanr(comp_df["viol_180"], comp_df["viol_5k"])
        print(f"\n  Kendall τ = {tau:.3f} (p={p_tau:.4f})")
        print(f"  Spearman ρ = {rho:.3f} (p={p_rho:.4f})")
        comp_df.attrs["kendall_tau"] = round(tau, 3)
        comp_df.attrs["spearman_rho"] = round(rho, 3)

    return comp_df


# ═══════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════

def plot_ci_forest(ci_df: pd.DataFrame, out_dir: Path) -> None:
    """Forest plot of bootstrap CIs for each config's violation rate."""
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ci_df["config"].values
    means = ci_df["mean"].values
    ci_lo = ci_df["ci_lo"].values
    ci_hi = ci_df["ci_hi"].values

    y = np.arange(len(labels))
    ax.barh(y, means, xerr=[means - ci_lo, ci_hi - means],
            capsize=4, color=sns.color_palette("Blues_d", len(labels)),
            edgecolor="black", linewidth=0.5, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Violation Rate (%)")
    ax.set_title("Bootstrap 95% CI — Violation Rate per Config (5K test)")
    ax.invert_yaxis()
    for i, (m, lo, hi) in enumerate(zip(means, ci_lo, ci_hi)):
        ax.text(hi + 0.15, i, f"{m:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "bootstrap_ci_forest.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(out_dir / "bootstrap_ci_forest.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [saved] bootstrap_ci_forest.pdf/.png")


def plot_rule_stacked_bar(rule_df: pd.DataFrame, out_dir: Path) -> None:
    """Stacked bar chart of rule violations per config."""
    fig, ax = plt.subplots(figsize=(10, 5))
    configs = rule_df["config"].values
    x = np.arange(len(configs))
    width = 0.6
    rules = ["R1", "R3", "R6b"]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    bottoms = np.zeros(len(configs))
    for rule, color in zip(rules, colors):
        vals = rule_df[rule].values.astype(float)
        ax.bar(x, vals, width, bottom=bottoms, label=rule, color=color, edgecolor="white")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right")
    ax.set_ylabel("Violation Count")
    ax.set_title("Rule Violation Distribution Across Ablation Chain (5K)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "rule_stacked_bar.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(out_dir / "rule_stacked_bar.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [saved] rule_stacked_bar.pdf/.png")


def plot_anatomy_violations(anat_df: pd.DataFrame, out_dir: Path) -> None:
    """Top-15 anatomy keywords by violation count."""
    if anat_df.empty:
        return
    top = anat_df.head(15).copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(top))
    width = 0.25
    for i, (rule, color) in enumerate(zip(["R1", "R3", "R6b"], ["#e74c3c", "#3498db", "#2ecc71"])):
        ax.bar(x + i * width, top[rule].values, width, label=rule, color=color)
    ax.set_xticks(x + width)
    ax.set_xticklabels(top["anatomy"].values, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Violation Count")
    ax.set_title("Violation Breakdown by Anatomy (B2'v2, top 15)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "anatomy_violations.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(out_dir / "anatomy_violations.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [saved] anatomy_violations.pdf/.png")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    print("ProveTok 5K — Statistical Analysis & Error Breakdown")
    print(f"Data: {ABLATION_DIR}")
    print(f"Output: {OUT_DIR}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all trace data
    print("Loading traces...")
    all_data = load_all_traces()
    if not all_data:
        sys.exit("ERROR: No data loaded")

    # 1. Statistical significance
    ci_df, perm_df = run_significance_tests(all_data)
    ci_df.to_csv(OUT_DIR / "bootstrap_ci.csv", index=False)
    perm_df.to_csv(OUT_DIR / "pairwise_permutation.csv", index=False)
    print(f"  [saved] bootstrap_ci.csv, pairwise_permutation.csv")

    # 2. Error analysis
    rule_df, anat_df, ds_df = run_error_analysis(all_data)
    rule_df.to_csv(OUT_DIR / "rule_distribution.csv", index=False)
    if not anat_df.empty:
        anat_df.to_csv(OUT_DIR / "anatomy_violations.csv", index=False)
    ds_df.to_csv(OUT_DIR / "per_dataset_violations.csv", index=False)
    print(f"  [saved] rule_distribution.csv, anatomy_violations.csv, per_dataset_violations.csv")

    # 3. D2 Repair analysis
    repair_df = run_repair_analysis(all_data)
    if not repair_df.empty:
        repair_df.to_csv(OUT_DIR / "d2_repair_analysis.csv", index=False)
        print(f"  [saved] d2_repair_analysis.csv")

    # 4. 180 vs 5K consistency
    comp_df = run_consistency_check()
    if comp_df is not None:
        comp_df.to_csv(OUT_DIR / "consistency_180_vs_5k.csv", index=False)
        print(f"  [saved] consistency_180_vs_5k.csv")

    # Plots
    print("\nGenerating plots...")
    plot_ci_forest(ci_df, OUT_DIR)
    plot_rule_stacked_bar(rule_df, OUT_DIR)
    plot_anatomy_violations(anat_df, OUT_DIR)

    # Summary JSON
    summary = {
        "n_configs": len(all_data),
        "n_cases_per_config": {label: int(df["case_id"].nunique()) for label, df in all_data.items()},
        "bootstrap_R": 5000,
        "alpha": 0.05,
        "adjacent_pairs_tested": len(perm_df),
        "significant_pairs": int(perm_df["significant"].sum()) if "significant" in perm_df.columns else 0,
        "best_config_viol": ci_df.loc[ci_df["mean"].idxmin(), "config"] if not ci_df.empty else None,
    }
    with open(OUT_DIR / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [saved] analysis_summary.json")

    print("\n" + "=" * 60)
    print("  DONE — All results in: " + str(OUT_DIR))
    print("=" * 60)


if __name__ == "__main__":
    main()
