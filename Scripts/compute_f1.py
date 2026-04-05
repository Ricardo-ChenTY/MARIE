#!/usr/bin/env python3
"""
Compute token-level F1 (word overlap precision / recall / F1) for MARIE
generated conditions. This is the standard "clinical F1" metric used in
radiology report generation papers (R2Gen, CT2Rep, etc.).

Usage:
  python Scripts/compute_f1.py --base_dir outputs/5k_ablation
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import nltk


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
                    all_sentences.append(d)
    return all_sentences


def compute_token_f1(sentences: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute micro-averaged token-level precision, recall, F1.

    For each (reference, hypothesis) pair:
      - Tokenize both into word tokens (lowercased)
      - P = |hyp ∩ ref| / |hyp|
      - R = |hyp ∩ ref| / |ref|
      - F1 = 2PR / (P+R)

    Returns micro-averaged (sum numerators / sum denominators) values.
    """
    total_tp = 0
    total_hyp_len = 0
    total_ref_len = 0
    n_pairs = 0

    for s in sentences:
        ref_text = s.get("original_topic", "")
        hyp_text = s.get("sentence_text", "")
        if not ref_text or not hyp_text:
            continue
        if not s.get("generated", False) and ref_text == hyp_text:
            continue

        ref_tokens = nltk.word_tokenize(ref_text.lower())
        hyp_tokens = nltk.word_tokenize(hyp_text.lower())
        if not ref_tokens or not hyp_tokens:
            continue

        ref_counter = Counter(ref_tokens)
        hyp_counter = Counter(hyp_tokens)
        overlap = sum((ref_counter & hyp_counter).values())

        total_tp += overlap
        total_hyp_len += len(hyp_tokens)
        total_ref_len += len(ref_tokens)
        n_pairs += 1

    if n_pairs == 0:
        return {"n_pairs": 0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0}

    precision = total_tp / total_hyp_len if total_hyp_len > 0 else 0.0
    recall = total_tp / total_ref_len if total_ref_len > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "n_pairs": n_pairs,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute token-level F1")
    parser.add_argument("--base_dir", default="outputs/5k_ablation")
    args = parser.parse_args()

    base = Path(args.base_dir)
    gen_conditions = ["B2_evcard_v1", "B2_evcard_v2", "C2_evcard_v1_judge", "D2_repair"]

    print(f"{'Condition':<30} {'Pairs':>6} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 70)

    results = {}
    for cond in gen_conditions:
        cond_dir = base / cond
        if not cond_dir.exists():
            print(f"{cond:<30} NOT FOUND")
            continue
        sentences = load_traces(str(cond_dir))
        metrics = compute_token_f1(sentences)
        results[cond] = metrics
        print(f"{cond:<30} {metrics['n_pairs']:>6} {metrics['Precision']:>10.4f} {metrics['Recall']:>10.4f} {metrics['F1']:>10.4f}")

    out_path = base.parent / "evaluation_5k" / "f1_scores.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

