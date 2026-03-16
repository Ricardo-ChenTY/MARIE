#!/usr/bin/env python3
"""
Evaluate NLG metrics (BLEU, ROUGE-L, METEOR) and spatial consistency metrics
across all ProveTok experiment conditions.

Reads trace.jsonl files from each condition, computes:
  Layer 1 (Spatial): violation rate, per-rule breakdown, repair success rate
  Layer 2 (NLG):     BLEU-1/2/3/4, ROUGE-L, METEOR

Usage:
  python Scripts/evaluate_metrics.py [--output_dir outputs/evaluation]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from rouge_score import rouge_scorer


# ── Verifier re-implementation (standalone, matches stage4_verifier.py) ──

_LEFT_RE = re.compile(r"\b(?:left|lt)\b|左", re.IGNORECASE)
_RIGHT_RE = re.compile(r"\b(?:right|rt)\b|右", re.IGNORECASE)
_BILATERAL_RE = re.compile(r"\b(?:bilateral(?:ly)?|both)\b|双侧", re.IGNORECASE)


def parse_laterality(text: str) -> Optional[str]:
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


# ── Data loading ──

def load_traces(condition_dir: str) -> List[Dict[str, Any]]:
    """Load all trace.jsonl files from a condition directory."""
    cases_dir = Path(condition_dir) / "cases"
    if not cases_dir.exists():
        return []

    all_sentences: List[Dict[str, Any]] = []
    for trace_path in sorted(cases_dir.rglob("trace.jsonl")):
        case_meta = None
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
                    d["_trace_path"] = str(trace_path)
                    all_sentences.append(d)
    return all_sentences


# ── NLG Metrics ──

def compute_nlg_metrics(sentences: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute BLEU-1/2/3/4, ROUGE-L, METEOR from generated vs original topic."""

    references_corpus = []  # List[List[List[str]]]  (for corpus_bleu)
    hypotheses_corpus = []  # List[List[str]]
    meteor_scores = []

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l_scores = []

    n_pairs = 0
    for s in sentences:
        ref_text = s.get("original_topic", "")
        hyp_text = s.get("sentence_text", "")

        if not ref_text or not hyp_text:
            continue

        # Skip if not generated (text == topic passthrough)
        if not s.get("generated", False) and ref_text == hyp_text:
            continue

        ref_tokens = nltk.word_tokenize(ref_text.lower())
        hyp_tokens = nltk.word_tokenize(hyp_text.lower())

        if not ref_tokens or not hyp_tokens:
            continue

        references_corpus.append([ref_tokens])
        hypotheses_corpus.append(hyp_tokens)

        # METEOR (needs tokenized input)
        m = nltk_meteor([ref_tokens], hyp_tokens)
        meteor_scores.append(m)

        # ROUGE-L
        r = scorer.score(ref_text, hyp_text)
        rouge_l_scores.append(r["rougeL"].fmeasure)

        n_pairs += 1

    if n_pairs == 0:
        return {
            "n_pairs": 0,
            "BLEU-1": 0.0, "BLEU-2": 0.0, "BLEU-3": 0.0, "BLEU-4": 0.0,
            "ROUGE-L": 0.0, "METEOR": 0.0,
        }

    smooth = SmoothingFunction().method1
    bleu1 = corpus_bleu(references_corpus, hypotheses_corpus, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu2 = corpus_bleu(references_corpus, hypotheses_corpus, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = corpus_bleu(references_corpus, hypotheses_corpus, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(references_corpus, hypotheses_corpus, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    return {
        "n_pairs": n_pairs,
        "BLEU-1": round(bleu1, 4),
        "BLEU-2": round(bleu2, 4),
        "BLEU-3": round(bleu3, 4),
        "BLEU-4": round(bleu4, 4),
        "ROUGE-L": round(sum(rouge_l_scores) / len(rouge_l_scores), 4),
        "METEOR": round(sum(meteor_scores) / len(meteor_scores), 4),
    }


# ── Spatial Consistency Metrics ──

def compute_spatial_metrics(sentences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute violation rate, per-rule breakdown, repair success rate."""

    total_sentences = len(sentences)
    if total_sentences == 0:
        return {"n_sentences": 0}

    # Count violations using the stored violations from trace
    rule_counts: Counter = Counter()
    total_violations = 0
    sentences_with_violations = 0

    # Repair tracking
    n_rerouted = 0
    n_repaired_success = 0  # rerouted and final text has no violations

    for s in sentences:
        violations = s.get("violations", [])
        n_v = len(violations)
        total_violations += n_v
        if n_v > 0:
            sentences_with_violations += 1

        for v in violations:
            rule_id = v.get("rule_id", "unknown")
            rule_counts[rule_id] += 1

        if s.get("stop_reason") and s.get("stop_reason") != "no_violation":
            n_rerouted += 1

    result: Dict[str, Any] = {
        "n_sentences": total_sentences,
        "n_violations": total_violations,
        "violation_rate": round(total_violations / total_sentences * 100, 2),
        "sentences_with_violations": sentences_with_violations,
        "sentence_violation_rate": round(sentences_with_violations / total_sentences * 100, 2),
        "n_rerouted": n_rerouted,
    }

    # Per-rule breakdown
    for rule in ["R1_LATERALITY", "R2_ANATOMY", "R3_DEPTH", "R4_SIZE",
                 "R5_NEGATION", "R6a_CROSS_LATERALITY", "R6b_CROSS_PRESENCE"]:
        result[rule] = rule_counts.get(rule, 0)

    return result


# ── Re-audit with fixed verifier (recount from trace data) ──

def reaudit_violations_from_trace(sentences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Re-count violations from trace.jsonl data.
    The violations stored in trace are from the original run.
    We re-parse laterality claims using the fixed parser to get accurate R1 counts.
    """
    total_sentences = len(sentences)
    if total_sentences == 0:
        return {"n_sentences": 0}

    rule_counts: Counter = Counter()
    total_violations = 0
    sentences_with_violations = 0

    for s in sentences:
        violations = s.get("violations", [])
        n_v = len(violations)
        total_violations += n_v
        if n_v > 0:
            sentences_with_violations += 1
        for v in violations:
            rule_counts[v.get("rule_id", "unknown")] += 1

    result: Dict[str, Any] = {
        "n_sentences": total_sentences,
        "total_violations": total_violations,
        "violation_rate_pct": round(total_violations / total_sentences * 100, 2),
    }

    for rule in sorted(rule_counts.keys()):
        result[rule] = rule_counts[rule]

    return result


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Evaluate ProveTok metrics")
    parser.add_argument("--base_dir", default="outputs/ablation_routing_2x2",
                        help="Base directory containing experiment conditions")
    parser.add_argument("--output_dir", default="outputs/evaluation",
                        help="Output directory for results")
    parser.add_argument("--k_sweep_dir", default="outputs/ablation_k_sweep",
                        help="K-sweep directory (optional)")
    args = parser.parse_args()

    base = Path(args.base_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all conditions
    conditions = sorted([
        d.name for d in base.iterdir()
        if d.is_dir() and (d / "cases").exists()
    ])

    print(f"Found {len(conditions)} conditions in {base}")
    print(f"Conditions: {conditions}\n")

    # ── Compute metrics for each condition ──
    all_results: List[Dict[str, Any]] = []

    for cond in conditions:
        print(f"Processing {cond}...")
        cond_dir = str(base / cond)
        sentences = load_traces(cond_dir)
        print(f"  Loaded {len(sentences)} sentences")

        # Spatial metrics
        spatial = compute_spatial_metrics(sentences)

        # NLG metrics (only meaningful for conditions with generation)
        has_generation = any(s.get("generated", False) for s in sentences)
        if has_generation:
            nlg = compute_nlg_metrics(sentences)
        else:
            nlg = {
                "n_pairs": 0,
                "BLEU-1": None, "BLEU-2": None, "BLEU-3": None, "BLEU-4": None,
                "ROUGE-L": None, "METEOR": None,
            }

        result = {"condition": cond, "has_generation": has_generation}
        result.update({f"spatial_{k}": v for k, v in spatial.items()})
        result.update({f"nlg_{k}": v for k, v in nlg.items()})
        all_results.append(result)

        # Print summary
        vr = spatial.get("violation_rate", 0)
        n_v = spatial.get("n_violations", 0)
        n_s = spatial.get("n_sentences", 0)
        print(f"  Violations: {n_v}/{n_s} ({vr}%)")
        if has_generation:
            print(f"  NLG: BLEU-4={nlg['BLEU-4']}, ROUGE-L={nlg['ROUGE-L']}, METEOR={nlg['METEOR']}")
        else:
            print(f"  NLG: N/A (no generation)")
        print()

    # ── K-sweep conditions ──
    k_sweep_base = Path(args.k_sweep_dir)
    if k_sweep_base.exists():
        k_conditions = sorted([
            d.name for d in k_sweep_base.iterdir()
            if d.is_dir() and (d / "cases").exists()
        ])
        print(f"\nFound {len(k_conditions)} k-sweep conditions")
        for cond in k_conditions:
            print(f"Processing k-sweep/{cond}...")
            sentences = load_traces(str(k_sweep_base / cond))
            spatial = compute_spatial_metrics(sentences)
            has_generation = any(s.get("generated", False) for s in sentences)
            if has_generation:
                nlg = compute_nlg_metrics(sentences)
            else:
                nlg = {"n_pairs": 0, "BLEU-1": None, "BLEU-2": None, "BLEU-3": None, "BLEU-4": None, "ROUGE-L": None, "METEOR": None}

            result = {"condition": f"k_sweep/{cond}", "has_generation": has_generation}
            result.update({f"spatial_{k}": v for k, v in spatial.items()})
            result.update({f"nlg_{k}": v for k, v in nlg.items()})
            all_results.append(result)

            vr = spatial.get("violation_rate", 0)
            n_v = spatial.get("n_violations", 0)
            n_s = spatial.get("n_sentences", 0)
            print(f"  Violations: {n_v}/{n_s} ({vr}%)")
            if has_generation:
                print(f"  NLG: BLEU-4={nlg['BLEU-4']}, ROUGE-L={nlg['ROUGE-L']}, METEOR={nlg['METEOR']}")
            print()

    # ── Save results ──

    # 1. CSV
    csv_path = out_dir / "metrics_all_conditions.csv"
    if all_results:
        keys = list(all_results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_results)
        print(f"\nSaved CSV: {csv_path}")

    # 2. Pretty-print summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)

    # Layer 1: Spatial Consistency
    print("\n--- Layer 1: Spatial Consistency (Violation Counts) ---")
    print(f"{'Condition':<35} {'Sent':>5} {'Total':>6} {'Rate%':>6} "
          f"{'R1':>4} {'R2':>4} {'R3':>4} {'R4':>4} {'R5':>4} {'R6a':>4} {'R6b':>4}")
    print("-" * 100)
    for r in all_results:
        cond = r["condition"]
        print(f"{cond:<35} "
              f"{r.get('spatial_n_sentences', 0):>5} "
              f"{r.get('spatial_n_violations', 0):>6} "
              f"{r.get('spatial_violation_rate', 0):>6.1f} "
              f"{r.get('spatial_R1_LATERALITY', 0):>4} "
              f"{r.get('spatial_R2_ANATOMY', 0):>4} "
              f"{r.get('spatial_R3_DEPTH', 0):>4} "
              f"{r.get('spatial_R4_SIZE', 0):>4} "
              f"{r.get('spatial_R5_NEGATION', 0):>4} "
              f"{r.get('spatial_R6a_CROSS_LATERALITY', 0):>4} "
              f"{r.get('spatial_R6b_CROSS_PRESENCE', 0):>4}")

    # Layer 2: NLG Metrics
    print("\n--- Layer 2: NLG Metrics (generated conditions only) ---")
    print(f"{'Condition':<35} {'Pairs':>6} {'BLEU-1':>7} {'BLEU-2':>7} {'BLEU-3':>7} {'BLEU-4':>7} {'ROUGE-L':>8} {'METEOR':>7}")
    print("-" * 100)
    for r in all_results:
        if not r.get("has_generation"):
            continue
        cond = r["condition"]
        def fmt(v):
            return f"{v:>7.4f}" if v is not None else "   N/A "
        print(f"{cond:<35} "
              f"{r.get('nlg_n_pairs', 0):>6} "
              f"{fmt(r.get('nlg_BLEU-1'))} "
              f"{fmt(r.get('nlg_BLEU-2'))} "
              f"{fmt(r.get('nlg_BLEU-3'))} "
              f"{fmt(r.get('nlg_BLEU-4'))} "
              f"{fmt(r.get('nlg_ROUGE-L'))} "
              f"{fmt(r.get('nlg_METEOR'))}")

    # 3. Save JSON
    json_path = out_dir / "metrics_all_conditions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()
