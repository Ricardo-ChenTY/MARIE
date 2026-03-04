#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


def _iter_trace_files(run_dir: Path) -> Iterable[Path]:
    cases_dir = run_dir / "cases"
    if not cases_dir.exists():
        return []
    return cases_dir.glob("*/*/trace.jsonl")


def summarize_run(run_dir: Path) -> Dict[str, object]:
    run_meta = {}
    run_meta_path = run_dir / "run_meta.json"
    if run_meta_path.exists():
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))

    sentence_total = 0
    sentence_with_violation = 0
    violations_total = 0
    rule_counter: Counter[str] = Counter()
    dataset_sent = defaultdict(int)
    dataset_sent_violate = defaultdict(int)

    for tf in _iter_trace_files(run_dir):
        dataset = tf.parts[-3]
        with tf.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("type") != "sentence":
                    continue
                sentence_total += 1
                dataset_sent[dataset] += 1
                violations = obj.get("violations", []) or []
                if violations:
                    sentence_with_violation += 1
                    dataset_sent_violate[dataset] += 1
                violations_total += len(violations)
                for v in violations:
                    if isinstance(v, dict):
                        rid = str(v.get("rule_id", "UNKNOWN"))
                        rule_counter[rid] += 1

    return {
        "run_dir": str(run_dir),
        "r2_mode": run_meta.get("r2_mode"),
        "r2_min_support_ratio": run_meta.get("r2_min_support_ratio"),
        "cp_strict": run_meta.get("cp_strict"),
        "sentence_total": int(sentence_total),
        "sentence_with_violation": int(sentence_with_violation),
        "violation_sentence_rate": float(sentence_with_violation / max(1, sentence_total)),
        "violations_total": int(violations_total),
        "R1_LATERALITY": int(rule_counter.get("R1_LATERALITY", 0)),
        "R2_ANATOMY": int(rule_counter.get("R2_ANATOMY", 0)),
        "R3_DEPTH": int(rule_counter.get("R3_DEPTH", 0)),
        "R4_SIZE": int(rule_counter.get("R4_SIZE", 0)),
        "R5_NEGATION": int(rule_counter.get("R5_NEGATION", 0)),
        "ctrate_sentence_rate": float(
            dataset_sent_violate["ctrate"] / max(1, dataset_sent["ctrate"])
        ),
        "radgenome_sentence_rate": float(
            dataset_sent_violate["radgenome"] / max(1, dataset_sent["radgenome"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize R2 sweep run folders (r2_ratio_*)")
    parser.add_argument("--sweep_root", type=str, required=True, help="Root folder containing r2_ratio_* run dirs.")
    parser.add_argument(
        "--glob",
        type=str,
        default="r2_ratio_*",
        help="Subdir glob under sweep_root. Default: r2_ratio_*",
    )
    parser.add_argument("--save_csv", type=str, default=None, help="Optional output CSV path.")
    args = parser.parse_args()

    root = Path(args.sweep_root).expanduser().resolve()
    run_dirs: List[Path] = sorted([p for p in root.glob(args.glob) if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run dirs found under {root} with glob={args.glob}")

    rows = [summarize_run(r) for r in run_dirs]
    df = pd.DataFrame(rows).sort_values(by=["r2_min_support_ratio", "run_dir"], ascending=[False, True])
    print(df.to_string(index=False))

    if args.save_csv:
        out = Path(args.save_csv).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
