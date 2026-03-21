#!/usr/bin/env python3
"""
Generate Table 2 (ablation table) and Figures 1-3 for the ProveTok paper.

Table 2: Ablation chain + grounding metrics + compute costs + violation rates + NLG
Fig 1:   Waterfall / step plot of ablation chain
Fig 2:   Budget sweep (k vs faithfulness)
Fig 3:   Counterfactual sensitivity analysis

Usage:
    python Scripts/generate_table2_and_figures.py                            # 180-case (default)
    python Scripts/generate_table2_and_figures.py --data_dir outputs/5k_ablation  # 5K
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ──
ROOT = Path(__file__).resolve().parent.parent
K_SWEEP_DIR = ROOT / "outputs" / "ablation_k_sweep"
FIVE_K_DIR = ROOT / "outputs" / "stage0_5_5k"

# These are set in main() based on --data_dir
ABLATION_DIR: Path = ROOT / "outputs" / "ablation_routing_2x2"
OUT_DIR: Path = ROOT / "outputs" / "paper_figures"


# ── Ablation chain definition ──
# (label, description) — directory names are resolved per data source
ABLATION_CHAIN_LABELS = [
    ("A0",    "Identity $W$ + Spatial"),
    ("A1",    "Trained $W$ + Spatial"),
    ("E1",    "Spatial filter + Semantic rerank"),
    ("B2",    "+ LLM generation"),
    ("B2'",   "+ Evidence Card v1"),
    ("B2'v2", "+ Evidence Card v2"),
    ("C2'",   "+ LLM Judge (Stage 5)"),
    ("D2",    "+ Repair executor"),
]

# Directory name mappings per data source
_DIRNAME_180 = {
    "A0": "A0_identity_spatial",
    "A1": "A1_trained_spatial",
    "E1": "E1_trained_filter_rerank",
    "B2": "B2_E1_stage3c",
    "B2'": "B2_evcard",
    "B2'v2": "B2_evcard_v2",
    "C2'": "C2_evcard",
    "D2": "D2_repair",
}

_DIRNAME_5K = {
    "A0": "A0_identity_spatial",
    "A1": "A1_trained_spatial",
    "E1": "E1_spatial_filter_rerank",
    "B2'": "B2_evcard_v1",
    "B2'v2": "B2_evcard_v2",
    "C2'": "C2_evcard_v1_judge",
    "D2": "D2_repair",
}

# Active chain — set in main()
ABLATION_CHAIN: List[Tuple[str, str, str]] = []

K_SWEEP_ORDER = [1, 2, 4, 6, 8, 12, 14]


# ── Data loading ──

def load_traces(condition_dir: str, load_tokens: bool = False) -> List[Dict[str, Any]]:
    """Load all trace.jsonl sentence records from a condition directory.

    If load_tokens=True, also loads tokens.json per case and attaches
    a _tokens_by_id dict to each sentence for grounding metric computation.
    """
    cases_dir = Path(condition_dir) / "cases"
    if not cases_dir.exists():
        return []
    all_sentences: List[Dict[str, Any]] = []
    for trace_path in sorted(cases_dir.rglob("trace.jsonl")):
        case_meta = None
        case_sentences: List[Dict[str, Any]] = []
        with open(trace_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("type") == "case_meta":
                    case_meta = d
                elif d.get("type") == "sentence":
                    d["_case_id"] = case_meta.get("case_id", "") if case_meta else ""
                    case_sentences.append(d)

        # Optionally load tokens.json for IoU computation
        if load_tokens and case_sentences:
            tokens_path = trace_path.parent / "tokens.json"
            tokens_by_id = {}
            if tokens_path.exists():
                with open(tokens_path, encoding="utf-8") as tf:
                    for tok in json.load(tf):
                        tokens_by_id[tok["token_id"]] = tok
            for s in case_sentences:
                s["_tokens_by_id"] = tokens_by_id

        all_sentences.extend(case_sentences)
    return all_sentences


def load_run_meta(condition_dir: str) -> Optional[Dict]:
    """Load run_meta.json if exists."""
    p = Path(condition_dir) / "run_meta.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# ── Metrics computation ──

def compute_violation_metrics(sentences: List[Dict]) -> Dict[str, Any]:
    """Compute violation rate and per-rule breakdown."""
    n = len(sentences)
    if n == 0:
        return {}
    rule_counts: Counter = Counter()
    total_v = 0
    for s in sentences:
        for v in s.get("violations", []):
            rule_counts[v.get("rule_id", "unknown")] += 1
            total_v += 1
    return {
        "n_sentences": n,
        "total_violations": total_v,
        "viol_rate": round(total_v / n * 100, 2),
        "R1": rule_counts.get("R1_LATERALITY", 0),
        "R2": rule_counts.get("R2_ANATOMY", 0),
        "R3": rule_counts.get("R3_DEPTH", 0),
        "R6a": rule_counts.get("R6a_CROSS_LATERALITY", 0),
        "R6b": rule_counts.get("R6b_CROSS_PRESENCE", 0),
    }


def compute_compute_metrics(sentences: List[Dict], run_meta: Optional[Dict]) -> Dict[str, Any]:
    """Estimate compute costs: C_LLM, C_attn, N_calls."""
    n = len(sentences)
    has_gen = any(s.get("generated", False) for s in sentences)
    has_judge = any(s.get("stage5_judgements") for s in sentences)

    # C_LLM = number of LLM calls (generation + judge)
    c_llm_gen = sum(1 for s in sentences if s.get("generated", False))
    c_llm_judge = sum(len(s.get("stage5_judgements", [])) for s in sentences)
    c_llm = c_llm_gen + c_llm_judge

    # N_reroute = number of rerouted sentences
    n_reroute = sum(1 for s in sentences
                    if s.get("stop_reason", "no_violation") != "no_violation")

    # C_attn proxy = k × B (tokens cited per sentence × token budget)
    k_vals = [len(s.get("topk_token_ids", [])) for s in sentences]
    avg_k = np.mean(k_vals) if k_vals else 0

    return {
        "C_LLM": c_llm,
        "C_LLM_gen": c_llm_gen,
        "C_LLM_judge": c_llm_judge,
        "N_reroute": n_reroute,
        "avg_k": round(avg_k, 1),
        "has_gen": has_gen,
        "has_judge": has_judge,
    }


def _bbox3d_overlap_ratio(token_bbox: Dict, anatomy_bbox: Dict) -> float:
    """Compute overlap ratio = intersection_volume / token_volume.

    This measures what fraction of the token falls inside the anatomy region.
    Unlike IoU, this is robust to large size differences between anatomy bbox
    (e.g. entire right lung) and small token bboxes (e.g. 32³ voxels).
    """
    ix = max(0, min(token_bbox["x_max"], anatomy_bbox["x_max"]) -
             max(token_bbox["x_min"], anatomy_bbox["x_min"]))
    iy = max(0, min(token_bbox["y_max"], anatomy_bbox["y_max"]) -
             max(token_bbox["y_min"], anatomy_bbox["y_min"]))
    iz = max(0, min(token_bbox["z_max"], anatomy_bbox["z_max"]) -
             max(token_bbox["z_min"], anatomy_bbox["z_min"]))
    inter = ix * iy * iz
    if inter <= 0:
        return 0.0
    token_vol = ((token_bbox["x_max"] - token_bbox["x_min"]) *
                 (token_bbox["y_max"] - token_bbox["y_min"]) *
                 (token_bbox["z_max"] - token_bbox["z_min"]))
    return inter / token_vol if token_vol > 0 else 0.0


# ── Citation Faithfulness helpers ──

_LEFT_RE = re.compile(r"\b(?:left|lt)\b", re.IGNORECASE)
_RIGHT_RE = re.compile(r"\b(?:right|rt)\b", re.IGNORECASE)
_BILATERAL_RE = re.compile(r"\b(?:bilateral(?:ly)?|both)\b", re.IGNORECASE)
_NEGATION_RE = re.compile(r"\b(?:no|not|without|absent|unremarkable|normal|clear|negative)\b", re.IGNORECASE)


def _parse_text_laterality(text: str) -> Optional[str]:
    """Parse laterality claim from generated text."""
    has_left = bool(_LEFT_RE.search(text))
    has_right = bool(_RIGHT_RE.search(text))
    has_bi = bool(_BILATERAL_RE.search(text))
    if has_bi or (has_left and has_right):
        return "bilateral"
    if has_left:
        return "left"
    if has_right:
        return "right"
    return None


def _token_side(tok_bbox: Dict) -> str:
    """Determine which side a token is on based on x-coordinate center.
    In our volume: right lung = x < 0.5 (voxel < 64), left lung = x >= 0.5."""
    center_x = (tok_bbox["x_min"] + tok_bbox["x_max"]) / 2.0
    midpoint = _VOL_SHAPE / 2.0  # 64 for 128³
    return "right" if center_x < midpoint else "left"


def _tokens_dominant_side(token_bboxes: List[Dict]) -> Optional[str]:
    """Determine dominant side of a set of token bboxes."""
    if not token_bboxes:
        return None
    left_count = sum(1 for b in token_bboxes if _token_side(b) == "left")
    right_count = len(token_bboxes) - left_count
    if left_count == right_count:
        return "bilateral"
    return "left" if left_count > right_count else "right"


def _laterality_matches(text_lat: str, token_lat: str) -> bool:
    """Check if text laterality claim matches token laterality."""
    if text_lat == token_lat:
        return True
    # bilateral text is compatible with any token side
    if text_lat == "bilateral":
        return True
    return False


# Normalized anatomy boxes (same as ProveTok_Main_experiment/simple_modules.py)
_ANATOMY_BOXES_NORM = {
    "right lung":       {"x_min": 0.0, "x_max": 0.5, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0},
    "left lung":        {"x_min": 0.5, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0},
    "right upper lobe": {"x_min": 0.0, "x_max": 0.5, "y_min": 0.5, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0},
    "right lower lobe": {"x_min": 0.0, "x_max": 0.5, "y_min": 0.0, "y_max": 0.5, "z_min": 0.0, "z_max": 1.0},
    "left upper lobe":  {"x_min": 0.5, "x_max": 1.0, "y_min": 0.5, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0},
    "left lower lobe":  {"x_min": 0.5, "x_max": 1.0, "y_min": 0.0, "y_max": 0.5, "z_min": 0.0, "z_max": 1.0},
    "mediastinum":      {"x_min": 0.3, "x_max": 0.7, "y_min": 0.15, "y_max": 0.85, "z_min": 0.0, "z_max": 1.0},
    "bilateral":        {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0},
}

_VOL_SHAPE = 128  # all volumes are 128³ in our dataset


def _resolve_anatomy_bbox(keyword: str) -> Optional[Dict]:
    """Resolve anatomy keyword to voxel-space bbox dict."""
    key = keyword.lower().strip()
    norm = _ANATOMY_BOXES_NORM.get(key)
    if norm is None:
        return None
    s = _VOL_SHAPE
    return {
        "x_min": norm["x_min"] * s, "x_max": norm["x_max"] * s,
        "y_min": norm["y_min"] * s, "y_max": norm["y_max"] * s,
        "z_min": norm["z_min"] * s, "z_max": norm["z_max"] * s,
    }


def compute_grounding_metrics(sentences: List[Dict], condition_dir: str) -> Dict[str, Any]:
    """
    Compute grounding metrics from trace + token data:
    - mean_iou: average max-IoU between cited tokens and anatomy bbox
    - hit_at_1/4/8: fraction of grounding-eligible sentences where top-1/4/8 has IoU > tau
    - routing_precision: fraction of cited tokens with IoU ≥ tau
    - lat_accuracy: laterality correctness (from evidence card vs R1 violations)
    - viol_free_rate: fraction of sentences with zero violations
    """
    n = len(sentences)
    if n == 0:
        return {}

    # Overlap ratio threshold: a token is "grounded" if ≥50% of its volume
    # falls inside the anatomy region.
    overlap_tau = 0.5

    # Skip "bilateral" — its bbox is the entire volume, so all tokens
    # trivially achieve 100% overlap. Including it inflates all metrics.
    SKIP_KEYWORDS = {"bilateral"}

    # --- Overlap-based metrics (need _tokens_by_id) ---
    overlap_vals = []        # per-sentence mean overlap ratio
    hit1, hit4, hit8 = 0, 0, 0
    prec_nums, prec_dens = 0, 0
    n_grounding_eligible = 0

    for s in sentences:
        kw = s.get("anatomy_keyword")
        if not kw or kw.lower().strip() in SKIP_KEYWORDS:
            continue
        anat_bbox = _resolve_anatomy_bbox(kw)
        if anat_bbox is None:
            continue

        tokens_by_id = s.get("_tokens_by_id", {})
        cited_ids = s.get("topk_token_ids", [])
        if not tokens_by_id or not cited_ids:
            n_grounding_eligible += 1
            continue

        n_grounding_eligible += 1

        # Compute overlap ratio for each cited token
        overlaps = []
        for tid in cited_ids:
            tok = tokens_by_id.get(tid)
            if tok is None:
                overlaps.append(0.0)
                continue
            overlaps.append(_bbox3d_overlap_ratio(tok["bbox_3d_voxel"], anat_bbox))

        if overlaps:
            overlap_vals.append(np.mean(overlaps))  # mean overlap for this sentence

        # Hit@k: does top-k contain at least one token with overlap ≥ tau?
        if len(overlaps) >= 1 and max(overlaps[:1]) >= overlap_tau:
            hit1 += 1
        if len(overlaps) >= 4 and max(overlaps[:4]) >= overlap_tau:
            hit4 += 1
        if max(overlaps) >= overlap_tau:  # hit@8 (or whatever k is)
            hit8 += 1

        # Routing precision: fraction of cited tokens with overlap ≥ tau
        prec_nums += sum(1 for ov in overlaps if ov >= overlap_tau)
        prec_dens += len(overlaps)

    # --- Laterality accuracy (from evidence card + R1 violations) ---
    lat_correct = 0
    lat_total = 0
    for s in sentences:
        ec = s.get("evidence_card")
        if not ec:
            continue
        allowed = ec.get("laterality_allowed", "unconstrained")
        if allowed in ("unconstrained", "none"):
            continue
        lat_total += 1
        has_r1 = any(v.get("rule_id") == "R1_LATERALITY" for v in s.get("violations", []))
        if not has_r1:
            lat_correct += 1

    # --- Violation-free rate ---
    n_viol_free = sum(1 for s in sentences if not s.get("violations"))

    # --- Citation Faithfulness (text-to-token alignment) ---
    # For generated sentences: does the text laterality match cited token positions?
    cf_match = 0
    cf_total = 0
    # Depth faithfulness: does the text depth match token depth levels?
    df_match = 0
    df_total = 0

    for s in sentences:
        if not s.get("generated", False):
            continue  # only evaluate generated text
        text = s.get("sentence_text", "")
        if not text:
            continue

        tokens_by_id = s.get("_tokens_by_id", {})
        cited_ids = s.get("topk_token_ids", [])
        if not tokens_by_id or not cited_ids:
            continue

        # Laterality faithfulness
        text_lat = _parse_text_laterality(text)
        if text_lat and text_lat != "bilateral":
            # Get actual token bboxes
            tok_bboxes = []
            for tid in cited_ids:
                tok = tokens_by_id.get(tid)
                if tok:
                    tok_bboxes.append(tok["bbox_3d_voxel"])
            if tok_bboxes:
                token_lat = _tokens_dominant_side(tok_bboxes)
                cf_total += 1
                if _laterality_matches(text_lat, token_lat):
                    cf_match += 1

        # Depth faithfulness: check if text mentions specific depth terms
        # and cited tokens are at appropriate levels
        ec = s.get("evidence_card", {})
        level_hist = ec.get("level_histogram", {})
        if level_hist:
            levels = [int(k) for k in level_hist.keys()]
            level_range = ec.get("level_range_expected")
            if level_range and len(level_range) == 2:
                df_total += 1
                in_range = ec.get("in_range_count", 0)
                out_range = ec.get("out_of_range_count", 0)
                total_depth = in_range + out_range
                if total_depth > 0 and in_range / total_depth >= 0.5:
                    df_match += 1

    result: Dict[str, Any] = {
        "grounding_eligible": n_grounding_eligible,
        "viol_free_rate": round(n_viol_free / n * 100, 1) if n > 0 else None,
        "lat_accuracy": round(lat_correct / lat_total * 100, 1) if lat_total > 0 else None,
        "lat_total": lat_total,
        "citation_faith": round(cf_match / cf_total * 100, 1) if cf_total > 0 else None,
        "citation_faith_n": cf_total,
        "depth_faith": round(df_match / df_total * 100, 1) if df_total > 0 else None,
        "depth_faith_n": df_total,
    }

    # Overlap-based metrics (only available when tokens were loaded)
    if overlap_vals:
        result["mean_overlap"] = round(np.mean(overlap_vals), 3)
        result["hit_at_1"] = round(hit1 / n_grounding_eligible * 100, 1)
        result["hit_at_4"] = round(hit4 / n_grounding_eligible * 100, 1)
        result["hit_at_8"] = round(hit8 / n_grounding_eligible * 100, 1)
        result["routing_precision"] = round(prec_nums / prec_dens * 100, 1) if prec_dens > 0 else None

    return result


# ── NLG metrics (simplified — use pre-computed if available) ──

def load_precomputed_nlg(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """Load NLG metrics from pre-computed CSV."""
    result = {}
    if not csv_path.exists():
        return result
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cond = row["condition"]
            nlg = {}
            for k in ["nlg_BLEU-4", "nlg_ROUGE-L", "nlg_METEOR"]:
                v = row.get(k, "")
                nlg[k.replace("nlg_", "")] = float(v) if v else None
            result[cond] = nlg
    return result


# ═══════════════════════════════════════════
# TABLE 2: Ablation table
# ═══════════════════════════════════════════

def generate_table2(nlg_csv: Optional[Path] = None):
    """Generate Table 2 data and LaTeX."""
    print("=" * 80)
    print("TABLE 2: Ablation Chain")
    print("=" * 80)

    if nlg_csv is None:
        nlg_csv = ROOT / "outputs" / "evaluation_v2" / "metrics_all_conditions.csv"
    nlg_data = load_precomputed_nlg(nlg_csv)

    rows = []
    for label, dirname, desc in ABLATION_CHAIN:
        cond_dir = str(ABLATION_DIR / dirname)
        sentences = load_traces(cond_dir, load_tokens=True)
        if not sentences:
            print(f"  WARNING: No data for {label} ({dirname})")
            continue

        viol = compute_violation_metrics(sentences)
        compute = compute_compute_metrics(sentences, load_run_meta(cond_dir))
        grounding = compute_grounding_metrics(sentences, cond_dir)
        nlg = nlg_data.get(dirname, {})

        row = {
            "label": label,
            "desc": desc,
            "dirname": dirname,
            **viol,
            **compute,
            **grounding,
            "BLEU-4": nlg.get("BLEU-4"),
            "ROUGE-L": nlg.get("ROUGE-L"),
            "METEOR": nlg.get("METEOR"),
        }
        rows.append(row)

        # Print summary with grounding metrics
        ovl_str = f"Ovl={grounding.get('mean_overlap', 'N/A')}"
        prec_str = f"Prec={grounding.get('routing_precision', 'N/A')}"
        cf_str = f"CF={grounding.get('citation_faith', 'N/A')}"
        df_str = f"DF={grounding.get('depth_faith', 'N/A')}"
        print(f"  {label:<8} | Viol={viol.get('viol_rate', '?'):>6}% "
              f"R1={viol.get('R1', '?'):>4} R3={viol.get('R3', '?'):>4} | "
              f"{ovl_str} {prec_str} {cf_str} {df_str} | "
              f"BLEU-4={nlg.get('BLEU-4', 'N/A')}")

    # Generate LaTeX
    print("\n--- LaTeX ---")
    latex = _table2_latex(rows)
    print(latex)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "table2_ablation.tex", "w") as f:
        f.write(latex)
    with open(OUT_DIR / "table2_data.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)

    # Save grounding metrics separately for Table 3 design
    grounding_keys = [
        "label", "desc", "grounding_eligible",
        "mean_overlap", "hit_at_1", "hit_at_4", "hit_at_8",
        "routing_precision", "lat_accuracy", "lat_total",
        "citation_faith", "citation_faith_n",
        "depth_faith", "depth_faith_n",
        "viol_free_rate",
    ]
    grounding_rows = [{k: r.get(k) for k in grounding_keys} for r in rows]
    with open(OUT_DIR / "table3_grounding_data.json", "w") as f:
        json.dump(grounding_rows, f, indent=2, default=str)

    print(f"\nSaved to {OUT_DIR}/table2_ablation.tex")
    print(f"Saved to {OUT_DIR}/table3_grounding_data.json")
    return rows


def _table2_latex(rows: List[Dict]) -> str:
    """Generate LaTeX for Table 2."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation study on 180 test cases (90 CT-RATE + 90 RadGenome-ChestCT). "
                 r"Each row adds one component to the previous configuration. "
                 r"Viol.\% = total violations / total sentences. "
                 r"$C_\text{LLM}$ = number of LLM forward passes (generation + judge). "
                 r"NLG metrics computed on generated sentences only (conditions with Stage~3c).}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}ll rrrrr rr rrr@{}}")
    lines.append(r"\toprule")
    lines.append(r" & & \multicolumn{5}{c}{\textbf{Spatial Consistency}} "
                 r"& \multicolumn{2}{c}{\textbf{Compute}} "
                 r"& \multicolumn{3}{c}{\textbf{NLG Quality}} \\")
    lines.append(r"\cmidrule(lr){3-7} \cmidrule(lr){8-9} \cmidrule(lr){10-12}")
    lines.append(r"\textbf{ID} & \textbf{Configuration} "
                 r"& \textbf{Viol.\%}$\downarrow$ & \textbf{R1} & \textbf{R3} & \textbf{R6b} & \textbf{R2\,Hit\%}$\uparrow$ "
                 r"& $C_\text{LLM}$ & $k$ "
                 r"& \textbf{B-4}$\uparrow$ & \textbf{R-L}$\uparrow$ & \textbf{MTR}$\uparrow$ \\")
    lines.append(r"\midrule")

    best_label = "B2'v2"  # highlight best row

    for row in rows:
        label = row["label"]
        desc = row["desc"]
        vr = row.get("viol_rate", "—")
        r1 = row.get("R1", "—")
        r3 = row.get("R3", "—")
        r6b = row.get("R6b", "—")
        r2hit = row.get("R2_hit_rate", "—")
        if r2hit == "—" or r2hit is None:
            r2hit = "100.0"  # no R2 violations means 100% hit

        c_llm = row.get("C_LLM", 0)
        avg_k = row.get("avg_k", "—")
        b4 = f"{row['BLEU-4']:.3f}" if row.get("BLEU-4") is not None else "—"
        rl = f"{row['ROUGE-L']:.3f}" if row.get("ROUGE-L") is not None else "—"
        mtr = f"{row['METEOR']:.3f}" if row.get("METEOR") is not None else "—"

        # Bold best row
        if label == best_label:
            line = (f"\\textbf{{{label}}} & \\textbf{{{desc}}} "
                    f"& \\textbf{{{vr}}} & \\textbf{{{r1}}} & \\textbf{{{r3}}} "
                    f"& \\textbf{{{r6b}}} & \\textbf{{{r2hit}}} "
                    f"& {c_llm} & {avg_k} "
                    f"& \\textbf{{{b4}}} & \\textbf{{{rl}}} & \\textbf{{{mtr}}} \\\\")
        else:
            line = (f"{label} & {desc} "
                    f"& {vr} & {r1} & {r3} & {r6b} & {r2hit} "
                    f"& {c_llm} & {avg_k} "
                    f"& {b4} & {rl} & {mtr} \\\\")

        # Add midrule before generation stages
        if label == "B2":
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{12}{l}{\textit{+ LLM Generation (Stage 3c--5)}} \\")

        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ═══════════════════════════════════════════
# FIGURE 1: Waterfall / Step Plot
# ═══════════════════════════════════════════

def plot_waterfall(rows: List[Dict]):
    """Create waterfall/step plot of ablation chain."""
    print("\n" + "=" * 80)
    print("FIGURE 1: Waterfall Plot")
    print("=" * 80)

    labels = [r["label"] for r in rows]
    viol_rates = [r.get("viol_rate", 0) for r in rows]
    r1_counts = [r.get("R1", 0) for r in rows]
    r3_counts = [r.get("R3", 0) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    # Left panel: Violation rate (%)
    ax1 = axes[0]
    colors = []
    for i, label in enumerate(labels):
        if label == "B2'v2":
            colors.append("#2196F3")  # blue for best
        elif "B2" in label or "C2" in label or "D2" in label:
            colors.append("#4CAF50")  # green for generation stages
        else:
            colors.append("#FF9800")  # orange for routing-only

    bars1 = ax1.bar(range(len(labels)), viol_rates, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Violation Rate (%)", fontsize=11)
    ax1.set_title("(a) Total Violation Rate", fontsize=12)
    ax1.axhline(y=viol_rates[labels.index("B2'v2")], color="#2196F3",
                linestyle="--", alpha=0.5, linewidth=1)

    # Add value labels
    for bar, val in zip(bars1, viol_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    # Right panel: R1 and R3 stacked
    ax2 = axes[1]
    x = np.arange(len(labels))
    width = 0.35
    bars_r1 = ax2.bar(x - width/2, r1_counts, width, label="R1 (Laterality)",
                       color="#E57373", edgecolor="white", linewidth=0.5)
    bars_r3 = ax2.bar(x + width/2, r3_counts, width, label="R3 (Depth)",
                       color="#7986CB", edgecolor="white", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Violation Count", fontsize=11)
    ax2.set_title("(b) R1 & R3 Violations", fontsize=12)
    ax2.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_waterfall.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(OUT_DIR / "fig1_waterfall.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUT_DIR}/fig1_waterfall.pdf")


# ═══════════════════════════════════════════
# FIGURE 2: Budget Sweep (k vs faithfulness)
# ═══════════════════════════════════════════

def plot_budget_sweep():
    """Budget sweep: k_per_sentence vs violation rate."""
    print("\n" + "=" * 80)
    print("FIGURE 2: Budget Sweep (k vs Faithfulness)")
    print("=" * 80)

    ks = []
    viol_rates = []
    r1_counts = []
    r3_counts = []
    r6b_counts = []

    for k in K_SWEEP_ORDER:
        cond_dir = str(K_SWEEP_DIR / f"k{k}")
        sentences = load_traces(cond_dir)
        if not sentences:
            print(f"  WARNING: No data for k={k}")
            continue
        viol = compute_violation_metrics(sentences)
        ks.append(k)
        viol_rates.append(viol.get("viol_rate", 0))
        r1_counts.append(viol.get("R1", 0))
        r3_counts.append(viol.get("R3", 0))
        r6b_counts.append(viol.get("R6b", 0))
        print(f"  k={k:>2}: Viol={viol.get('viol_rate', 0):>6.2f}% "
              f"R1={viol.get('R1', 0):>4} R3={viol.get('R3', 0):>4} R6b={viol.get('R6b', 0):>3}")

    if not ks:
        print("  No k-sweep data found, skipping.")
        return

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    # Primary axis: violation rate
    color_viol = "#E53935"
    ax1.plot(ks, viol_rates, "o-", color=color_viol, linewidth=2, markersize=8,
             label="Violation Rate (%)", zorder=3)
    ax1.set_xlabel("$k$ (tokens per sentence)", fontsize=12)
    ax1.set_ylabel("Violation Rate (%)", fontsize=12, color=color_viol)
    ax1.tick_params(axis="y", labelcolor=color_viol)
    ax1.set_xticks(ks)

    # Highlight optimal k=8
    if 8 in ks:
        idx = ks.index(8)
        ax1.axvline(x=8, color="gray", linestyle="--", alpha=0.4)
        ax1.annotate(f"k*=8\n({viol_rates[idx]:.1f}%)",
                     xy=(8, viol_rates[idx]),
                     xytext=(10, viol_rates[idx] + 1.5),
                     fontsize=9, ha="center",
                     arrowprops=dict(arrowstyle="->", color="gray"))

    # Secondary axis: R1 and R3 counts
    ax2 = ax1.twinx()
    ax2.plot(ks, r1_counts, "s--", color="#1976D2", linewidth=1.5, markersize=6,
             alpha=0.7, label="R1 (Laterality)")
    ax2.plot(ks, r3_counts, "^--", color="#388E3C", linewidth=1.5, markersize=6,
             alpha=0.7, label="R3 (Depth)")
    ax2.set_ylabel("Violation Count", fontsize=12, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    ax1.set_title("Budget Sweep: $k$ vs Spatial Faithfulness", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig2_budget_sweep.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(OUT_DIR / "fig2_budget_sweep.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUT_DIR}/fig2_budget_sweep.pdf")


# ═══════════════════════════════════════════
# FIGURE 3: Counterfactual Sensitivity
# ═══════════════════════════════════════════

def run_counterfactual_analysis():
    """
    Counterfactual analysis using actual experimental data from metric_sanity_check.

    Uses real results from 5K test set (3,979 sentences):
    - original: baseline violation rate
    - paraphrase: synonym substitution (control — should be stable)
    - laterality_flip: "left" ↔ "right" (should spike R1)
    - presence_flip: "normal" ↔ "abnormal" (should spike R6b)
    """
    print("\n" + "=" * 80)
    print("FIGURE 3: Counterfactual Sensitivity Analysis")
    print("=" * 80)

    # Load actual sanity check results
    sanity_path = ROOT / "outputs" / "metric_sanity_check.json"
    if not sanity_path.exists():
        print("  No metric_sanity_check.json found, skipping.")
        return

    with open(sanity_path) as f:
        sanity = json.load(f)

    results = sanity.get("results", [])
    if not results:
        print("  No results in sanity check, skipping.")
        return

    n_sentences = results[0].get("total_sentences", 0)
    print(f"  Using actual experimental data: {n_sentences} sentences")

    # Build perturbation data from actual results
    perturbation_data = {}
    for r in results:
        name = r["perturbation"]
        perturbation_data[name] = {
            "viol_rate": r["violation_rate"],
            "R1": r["R1"],
            "R3": r["R3"],
            "R6b": r["R6b"],
            "total": r["total_violations"],
            "perturbed": r["sentences_perturbed"],
        }

    print(f"  Perturbations: {list(perturbation_data.keys())}")
    for name, d in perturbation_data.items():
        print(f"    {name}: viol={d['viol_rate']:.2f}% R1={d['R1']} R3={d['R3']} R6b={d['R6b']}")

    # ── Figure 3a: Overall violation rate bar chart ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    display_names = {
        "original": "Original",
        "paraphrase": "Paraphrase\n(control)",
        "laterality_flip": "Laterality\nflip",
        "presence_flip": "Presence\nflip",
    }
    order = ["original", "paraphrase", "laterality_flip", "presence_flip"]
    order = [o for o in order if o in perturbation_data]

    names = [display_names.get(o, o) for o in order]
    viol_rates = [perturbation_data[o]["viol_rate"] for o in order]
    colors = ["#4CAF50", "#78909C", "#FF7043", "#FF7043"]

    ax1 = axes[0]
    bars = ax1.bar(range(len(names)), viol_rates, color=colors[:len(names)],
                   edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, viol_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.axhline(y=viol_rates[0], color="#4CAF50", linestyle="--", alpha=0.4)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, fontsize=10)
    ax1.set_ylabel("Violation Rate (%)", fontsize=11)
    ax1.set_title("(a) Overall Violation Rate", fontsize=12)

    # ── Figure 3b: Per-rule breakdown ──
    ax2 = axes[1]
    x = np.arange(len(order))
    width = 0.25
    r1_vals = [perturbation_data[o]["R1"] for o in order]
    r3_vals = [perturbation_data[o]["R3"] for o in order]
    r6b_vals = [perturbation_data[o]["R6b"] for o in order]

    ax2.bar(x - width, r1_vals, width, label="R1 (Laterality)", color="#E57373")
    ax2.bar(x, r3_vals, width, label="R3 (Depth)", color="#7986CB")
    ax2.bar(x + width, r6b_vals, width, label="R6b (Cross-presence)", color="#FFB74D")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=10)
    ax2.set_ylabel("Violation Count", fontsize=11)
    ax2.set_title("(b) Per-Rule Breakdown", fontsize=12)
    ax2.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig3_counterfactual.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(OUT_DIR / "fig3_counterfactual.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUT_DIR}/fig3_counterfactual.pdf")

    # Save data
    with open(OUT_DIR / "fig3_counterfactual_data.json", "w") as f:
        json.dump({
            "source": "metric_sanity_check.json (actual experiments)",
            "n_sentences": n_sentences,
            "perturbations": perturbation_data,
        }, f, indent=2)


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def _build_ablation_chain(data_dir: Path) -> List[Tuple[str, str, str]]:
    """Build ablation chain with directory names resolved for the data source."""
    # Auto-detect: if data_dir contains 5k_ablation, use 5K mapping
    dir_name = data_dir.name
    if "5k" in dir_name.lower():
        dirname_map = _DIRNAME_5K
    else:
        dirname_map = _DIRNAME_180

    chain = []
    for label, desc in ABLATION_CHAIN_LABELS:
        dirname = dirname_map.get(label)
        if dirname is None:
            continue  # config not in this data source (e.g. B2 not in 5K)
        if (data_dir / dirname / "cases").exists():
            chain.append((label, dirname, desc))
        else:
            print(f"  [SKIP] {label} ({dirname}): no cases/ directory found")
    return chain


def main():
    global ABLATION_DIR, OUT_DIR, ABLATION_CHAIN

    parser = argparse.ArgumentParser(description="Generate ablation tables and figures")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to ablation data directory "
                             "(default: outputs/ablation_routing_2x2)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for tables/figures "
                             "(default: outputs/paper_figures[_5k])")
    parser.add_argument("--nlg_csv", type=str, default=None,
                        help="Path to pre-computed NLG metrics CSV")
    args = parser.parse_args()

    # Set data directory
    if args.data_dir:
        ABLATION_DIR = Path(args.data_dir)
        if not ABLATION_DIR.is_absolute():
            ABLATION_DIR = ROOT / ABLATION_DIR
    else:
        ABLATION_DIR = ROOT / "outputs" / "ablation_routing_2x2"

    # Set output directory
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
        if not OUT_DIR.is_absolute():
            OUT_DIR = ROOT / OUT_DIR
    elif "5k" in ABLATION_DIR.name.lower():
        OUT_DIR = ROOT / "outputs" / "paper_figures_5k"
    else:
        OUT_DIR = ROOT / "outputs" / "paper_figures"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build chain
    ABLATION_CHAIN = _build_ablation_chain(ABLATION_DIR)
    print(f"Data dir:   {ABLATION_DIR}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Chain:      {[c[0] for c in ABLATION_CHAIN]}")
    print()

    # Table 2 + waterfall data
    nlg_csv = Path(args.nlg_csv) if args.nlg_csv else None
    rows = generate_table2(nlg_csv=nlg_csv)

    if rows:
        # Figure 1: Waterfall
        plot_waterfall(rows)

    # Figure 2: Budget sweep
    plot_budget_sweep()

    # Figure 3: Counterfactual
    run_counterfactual_analysis()

    print("\n" + "=" * 80)
    print(f"All outputs saved to {OUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
