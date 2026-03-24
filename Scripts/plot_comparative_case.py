#!/usr/bin/env python3
"""
Generate comparative qualitative figure: A0 (baseline) vs B2'v2 (best) on the same case.

Shows how baseline routes to wrong tokens with violations,
while ProveTok routes correctly with evidence-card constraints.

Output: outputs/paper_figures/fig5_comparative.pdf/png

Usage:
    python Scripts/plot_comparative_case.py
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "paper_figures"
VOLUME_ROOT = ROOT / "dataset_5k" / "CT-RATE" / "train"

# ── Configs to compare ──
CONFIGS = {
    "A0": {
        "label": "A0 (Identity W + IoU Routing)",
        "dir": ROOT / "outputs" / "5k_ablation" / "A0_identity_spatial",
        "color": "#d32f2f",       # red accent
        "box_bg": "mistyrose",
    },
    "B2'v2": {
        "label": "B2'v2 (Trained W + E1 + Evidence Card v2)",
        "dir": ROOT / "outputs" / "5k_ablation" / "B2_evcard_v2",
        "color": "#2e7d32",       # green accent
        "box_bg": "honeydew",
    },
}

# ── Case & sentence to visualize ──
DATASET = "ctrate"
CASE_ID = "train_10754_c_1"
# Sentence index 5 has R3 violation in A0 but clean in B2'v2
TARGET_SENT_INDEX = 5
# Fallback: also try sentence 0 (right lung) if 5 doesn't work well
FALLBACK_SENT_INDEX = 0


def load_volume(path: Path) -> np.ndarray:
    if nib is None:
        raise ImportError("nibabel required")
    img = nib.load(str(path))
    return np.array(img.dataobj, dtype=np.float32)


def load_tokens(case_dir: Path) -> list:
    p = case_dir / "tokens.json"
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


def load_trace(case_dir: Path) -> list:
    p = case_dir / "trace.jsonl"
    sentences = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            if d.get("type") == "sentence":
                sentences.append(d)
    return sentences


def find_best_slice(tokens: list, cited_ids: list, vol_shape: tuple) -> int:
    token_map = {t["token_id"]: t for t in tokens}
    z_centers = []
    for tid in cited_ids:
        t = token_map.get(tid)
        if t and "bbox_3d_voxel" in t:
            bb = t["bbox_3d_voxel"]
            z_centers.append((bb["z_min"] + bb["z_max"]) / 2.0)
    if z_centers:
        return int(np.median(z_centers))
    return vol_shape[2] // 2


def pick_sentence(sentences: list, target_idx: int, fallback_idx: int) -> dict:
    """Pick the target sentence by index, fallback if needed."""
    for s in sentences:
        if s.get("sentence_index") == target_idx:
            return s
    for s in sentences:
        if s.get("sentence_index") == fallback_idx:
            return s
    return sentences[0]


def draw_ct_with_tokens(ax, slice_img, tokens, cited_ids, z_idx, cfg_color,
                        violations, title, score_range=None):
    """Draw CT slice with token overlay and violation indicators."""
    token_map = {t["token_id"]: t for t in tokens}

    ax.imshow(slice_img.T, cmap="gray", origin="lower", aspect="equal")

    # Color: green for no violation, red for violation
    has_violation = bool(violations)
    edge_color = "#d32f2f" if has_violation else "#2e7d32"

    for i, tid in enumerate(cited_ids):
        t = token_map.get(tid)
        if t is None or "bbox_3d_voxel" not in t:
            continue
        bb = t["bbox_3d_voxel"]
        if bb["z_min"] <= z_idx <= bb["z_max"]:
            # Use colormap for different tokens
            c = plt.cm.tab10(i / max(len(cited_ids), 1))
            rect = patches.Rectangle(
                (bb["x_min"], bb["y_min"]),
                bb["x_max"] - bb["x_min"],
                bb["y_max"] - bb["y_min"],
                linewidth=2.0,
                edgecolor=c,
                facecolor=c,
                alpha=0.25,
            )
            ax.add_patch(rect)
            ax.text(bb["x_min"] + 1, bb["y_min"] + 1,
                    f"t{tid}", fontsize=6, color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.6))

    # Violation/pass badge
    if has_violation:
        viol_ids = [v.get("rule_id", "?") for v in violations]
        badge = f"VIOLATION: {', '.join(viol_ids)}"
        badge_color = "#d32f2f"
    else:
        badge = "PASS (no violations)"
        badge_color = "#2e7d32"

    ax.text(0.98, 0.02, badge, transform=ax.transAxes,
            fontsize=8, fontweight="bold", color="white",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc=badge_color, alpha=0.85))

    if score_range:
        ax.text(0.02, 0.02, f"scores: {score_range}",
                transform=ax.transAxes, fontsize=7, color="white",
                ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("x (voxels)", fontsize=8)
    ax.set_ylabel("y (voxels)", fontsize=8)
    ax.tick_params(labelsize=7)


def draw_text_card(ax, sentence, cfg_label, box_bg, is_generated):
    """Draw the text evidence card panel."""
    ax.axis("off")

    text = sentence.get("sentence_text", "")
    anatomy = sentence.get("anatomy_keyword", "N/A")
    violations = sentence.get("violations", [])
    cited_ids = sentence.get("topk_token_ids", [])
    scores = sentence.get("topk_scores", [])
    evidence_card = sentence.get("evidence_card", {})
    original_topic = sentence.get("original_topic", "")

    wrapped_text = textwrap.fill(text, width=40)
    wrapped_topic = textwrap.fill(original_topic, width=40) if original_topic else ""

    # Violation info
    if violations:
        viol_lines = []
        for v in violations:
            viol_lines.append(f"  {v.get('rule_id', '?')}: {v.get('message', '')[:55]}")
        viol_text = "\n".join(viol_lines)
    else:
        viol_text = "  None"

    # Evidence card info
    ec_text = ""
    if evidence_card:
        ssr = evidence_card.get("same_side_ratio", "?")
        ssr_str = f"{ssr:.2f}" if isinstance(ssr, (int, float)) else str(ssr)
        ec_text = (
            f"\nEvidence Card:\n"
            f"  Dominant side: {evidence_card.get('dominant_side', '?')}\n"
            f"  SSR: {ssr_str}\n"
            f"  Lat. gate: {evidence_card.get('laterality_allowed', '?')}\n"
            f"  Depth gate: {evidence_card.get('depth_allowed', '?')}"
        )

    # Score range
    if scores:
        score_str = f"[{min(scores):.3f}, {max(scores):.3f}]"
    else:
        score_str = "N/A"

    display = (
        f"{'[LLM Generated]' if is_generated else '[Original Text]'}\n"
        f"{'─' * 36}\n"
        f"{wrapped_text}\n"
        f"{'─' * 36}\n"
    )
    if wrapped_topic and is_generated:
        display += f"Original topic:\n  {wrapped_topic}\n{'─' * 36}\n"

    display += (
        f"Anatomy: {anatomy}\n"
        f"Token IDs: {cited_ids}\n"
        f"Score range: {score_str}\n"
        f"Violations:\n{viol_text}"
        f"{ec_text}"
    )

    ax.text(0.05, 0.95, display,
            transform=ax.transAxes,
            fontsize=7.5, fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor=box_bg, alpha=0.85))

    ax.set_title(cfg_label, fontsize=9, fontweight="bold")


def main():
    print(f"Case: {DATASET}/{CASE_ID}")
    volume_path = VOLUME_ROOT / f"{CASE_ID}.nii.gz"

    # Load volume (shared)
    vol = None
    if volume_path.exists():
        print(f"Loading volume: {volume_path}")
        vol = load_volume(volume_path)
        if len(vol.shape) == 4:
            vol = vol[..., 0]
        print(f"  Shape: {vol.shape}")
    else:
        print("Volume not found, using schematic")

    # Load data for each config
    config_data = {}
    for cfg_name, cfg in CONFIGS.items():
        case_dir = cfg["dir"] / "cases" / DATASET / CASE_ID
        tokens = load_tokens(case_dir)
        sentences = load_trace(case_dir)
        sent = pick_sentence(sentences, TARGET_SENT_INDEX, FALLBACK_SENT_INDEX)

        print(f"\n{cfg_name}: {cfg['label']}")
        print(f"  Sentence idx: {sent.get('sentence_index')}")
        print(f"  Text: {sent.get('sentence_text', '')[:80]}...")
        print(f"  Generated: {sent.get('generated', False)}")
        print(f"  Violations: {[v.get('rule_id') for v in sent.get('violations', [])]}")
        print(f"  Token IDs: {sent.get('topk_token_ids', [])}")

        config_data[cfg_name] = {
            "tokens": tokens,
            "sentence": sent,
            "cfg": cfg,
        }

    # ── Build figure: 2 columns × 2 rows ──
    # Row 1: CT + token overlay for each config
    # Row 2: Text card for each config
    fig = plt.figure(figsize=(14, 11))

    # Add overall title
    fig.suptitle(
        f"Qualitative Comparison: Same Case ({CASE_ID}), Same Sentence",
        fontsize=13, fontweight="bold", y=0.98
    )

    cfg_names = list(CONFIGS.keys())

    for col_idx, cfg_name in enumerate(cfg_names):
        data = config_data[cfg_name]
        cfg = data["cfg"]
        sent = data["sentence"]
        tokens = data["tokens"]
        cited_ids = sent.get("topk_token_ids", [])
        violations = sent.get("violations", [])
        scores = sent.get("topk_scores", [])

        # Find slice for this config's cited tokens
        if vol is not None:
            z_idx = find_best_slice(tokens, cited_ids, vol.shape)
            slice_img = vol[:, :, z_idx]
        else:
            z_idx = 0
            slice_img = None

        score_range = None
        if scores:
            score_range = f"{min(scores):.3f} – {max(scores):.3f}"

        # Row 1: CT + tokens
        ax_ct = fig.add_subplot(2, 2, col_idx + 1)
        if slice_img is not None:
            draw_ct_with_tokens(
                ax_ct, slice_img, tokens, cited_ids, z_idx,
                cfg["color"], violations,
                title=f"{cfg_name}: Routed Tokens (z={z_idx})",
                score_range=score_range,
            )
        else:
            ax_ct.text(0.5, 0.5, "Volume N/A", ha="center", va="center",
                      transform=ax_ct.transAxes, fontsize=14, color="gray")
            ax_ct.set_title(f"{cfg_name}: Routed Tokens")

        # Row 2: Text card
        ax_txt = fig.add_subplot(2, 2, col_idx + 3)
        draw_text_card(
            ax_txt, sent, cfg["label"], cfg["box_bg"],
            is_generated=sent.get("generated", False),
        )

    # Add arrow between columns
    fig.text(0.50, 0.72, "→", fontsize=40, ha="center", va="center",
             fontweight="bold", color="#1565c0")
    fig.text(0.50, 0.69, "Same case,\nsame sentence",
             fontsize=9, ha="center", va="top", color="#1565c0", style="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = OUT_DIR / f"fig5_comparative.{ext}"
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nSaved to {OUT_DIR}/fig5_comparative.pdf/png")


if __name__ == "__main__":
    main()
