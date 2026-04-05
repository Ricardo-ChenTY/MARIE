#!/usr/bin/env python3
"""
Closed-loop detection and repair of a laterality violation.

3-column layout (no baseline):
  (a) Before repair — violation detected: CT slice + tokens coloured by side + evidence card
  (b) Judge diagnosis — which tokens offend, suggested action
  (c) After repair — regenerated sentence passes verification (PASS)

Case: train_18661_a_2  sentence 2
  B2'v2: "The right lung shows linear atelectasis in the middle lobe medial segment."
         L:5 R:3, ratio=0.625, dominant_side=bilateral → R1_LATERALITY
  D2:    "There are linear atelectasis in the right lung middle lobe medial segment
          and left lung upper lobe lingular segment."
         L:5 R:3, ratio=0.625, dominant_side=bilateral → PASS (de-specification)

Usage:
    python Scripts/plot_case_study.py
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "redrawn_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VOLUME_PATH = ROOT / "dataset_5k" / "CT-RATE" / "train" / "train_18661_a_2.nii.gz"
ABLATION_ROOT = ROOT / "outputs" / "5k_ablation"

CASE_ID = "train_18661_a_2"
DATASET = "ctrate"
SENT_IDX = 2

COL_SAME  = "#2CA9A0"  # teal  — same-side (correct / repaired)
COL_OPP   = "#E8923F"  # orange — opposite-side (violation)
COL_PASS  = "#2CA9A0"  # teal  — all tokens correct after repair


def load_volume(path: Path) -> np.ndarray:
    if nib is None:
        raise ImportError("nibabel required: pip install nibabel")
    img = nib.load(str(path))
    vol = np.array(img.dataobj, dtype=np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    return vol


def load_case(config: str):
    """Return (tokens, sentence_record) for a given ablation config."""
    case_dir = ABLATION_ROOT / config / "cases" / DATASET / CASE_ID
    with open(case_dir / "tokens.json") as f:
        tokens = json.load(f)
    with open(case_dir / "trace.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("type") == "sentence" and obj.get("sentence_index") == SENT_IDX:
                return tokens, obj
    raise ValueError(f"Sentence {SENT_IDX} not found in {config}")


def find_best_slice(tokens, cited_ids, vol_shape):
    """Pick axial slice at median z of cited tokens."""
    token_map = {t["token_id"]: t for t in tokens}
    z_centers = []
    for tid in cited_ids:
        t = token_map.get(tid)
        if t and "bbox_3d_voxel" in t:
            bb = t["bbox_3d_voxel"]
            z_centers.append((bb["z_min"] + bb["z_max"]) / 2.0)
    if z_centers:
        median_z = np.median(z_centers)
        max_z = max(t["bbox_3d_voxel"]["z_max"] for t in tokens if "bbox_3d_voxel" in t)
        return int(median_z / max_z * vol_shape[2])
    return vol_shape[2] // 2


def make_colour_map(tokens, cited_ids, claim_side="right"):
    """Colour tokens: same-side as claim → teal, opposite → orange."""
    token_map = {t["token_id"]: t for t in tokens}
    max_x = max(t["bbox_3d_voxel"]["x_max"] for t in tokens if "bbox_3d_voxel" in t)
    mid_x = max_x / 2.0

    cmap = {}
    for tid in cited_ids:
        t = token_map.get(tid)
        if t and "bbox_3d_voxel" in t:
            x_center = (t["bbox_3d_voxel"]["x_min"] + t["bbox_3d_voxel"]["x_max"]) / 2
            if claim_side == "right":
                on_claimed_side = x_center < mid_x
            else:
                on_claimed_side = x_center >= mid_x
            cmap[tid] = COL_SAME if on_claimed_side else COL_OPP
    return cmap


def draw_ct_panel(ax, vol, z_idx, tokens, cited_ids, colour_map, midline_x, title):
    """Draw CT axial slice with token bboxes overlaid."""
    sl = vol[:, :, z_idx].T
    wl, ww = -600, 1500
    vmin, vmax = wl - ww / 2, wl + ww / 2
    ax.imshow(sl, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, aspect="equal")

    token_map = {t["token_id"]: t for t in tokens}
    max_x = max(t["bbox_3d_voxel"]["x_max"] for t in tokens if "bbox_3d_voxel" in t)
    max_y = max(t["bbox_3d_voxel"]["y_max"] for t in tokens if "bbox_3d_voxel" in t)
    max_z = max(t["bbox_3d_voxel"]["z_max"] for t in tokens if "bbox_3d_voxel" in t)
    sx, sy, sz = vol.shape[0] / max_x, vol.shape[1] / max_y, vol.shape[2] / max_z

    for tid in cited_ids:
        t = token_map.get(tid)
        if t is None or "bbox_3d_voxel" not in t:
            continue
        bb = t["bbox_3d_voxel"]
        if not (bb["z_min"] * sz <= z_idx <= bb["z_max"] * sz):
            continue

        x0, y0 = bb["x_min"] * sx, bb["y_min"] * sy
        w, h = (bb["x_max"] - bb["x_min"]) * sx, (bb["y_max"] - bb["y_min"]) * sy
        col = colour_map.get(tid, "#AAAAAA")
        ax.add_patch(patches.Rectangle(
            (x0, y0), w, h, linewidth=1.8, edgecolor=col, facecolor=col, alpha=0.25))
        ax.text(x0 + 2, y0 + 2, f"t{tid}", fontsize=5.5, color="white",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc=col, alpha=0.7))

    ax.axvline(midline_x, linestyle="--", linewidth=0.8, color="white", alpha=0.5)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.tick_params(labelsize=6)


def draw_info_card(ax, sentence, box_color, show_judge=False, is_repaired=False):
    """Draw text info card below CT panel."""
    ax.axis("off")
    text = sentence.get("sentence_text", "")
    ec = sentence.get("evidence_card", {})
    violations = sentence.get("violations", [])

    wrapped = textwrap.fill(text, width=40)
    lines = [wrapped, ""]

    if ec:
        ssr = ec.get("same_side_ratio", "?")
        ssr_str = f"{ssr:.2f}" if isinstance(ssr, (int, float)) else str(ssr)
        lines.append(f"Cited: {ec.get('cited_tokens', '?')} tokens")
        lines.append(f"  L: {ec.get('left_count', '?')}  R: {ec.get('right_count', '?')}  "
                     f"ratio: {ssr_str}")
        lines.append(f"  Side: {ec.get('dominant_side', '?')}")
        lines.append("")

    if violations:
        for v in violations:
            lines.append(f"FAIL: {v.get('rule_id', '?')}")
    elif is_repaired:
        lines.append("PASS (no violations)")
    else:
        lines.append("Verifier: PASS")

    if show_judge:
        judges = sentence.get("stage5_judgements", [])
        for j in judges:
            if j.get("confirmed"):
                lines.append("")
                lines.append(f"Judge: CONFIRMED")
                lines.append(f"Action: {j.get('suggested_action', '?')}")
                reason = j.get("reasoning", "")
                if reason:
                    lines.append(textwrap.fill(reason, width=40)[:120])

    ax.text(0.05, 0.95, "\n".join(lines),
            transform=ax.transAxes, fontsize=7.5, fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.85))


def main():
    print("Loading volume...")
    vol = load_volume(VOLUME_PATH) if VOLUME_PATH.exists() else None

    print("Loading B2'v2 (before repair)...")
    tokens_b2, sent_b2 = load_case("B2_evcard_v2")

    print("Loading D2 (after repair)...")
    tokens_d2, sent_d2 = load_case("D2_repair")

    cited_b2 = sent_b2.get("topk_token_ids", [])[:8]
    cited_d2 = sent_d2.get("topk_token_ids", [])[:8]

    cmap_before = make_colour_map(tokens_b2, cited_b2, claim_side="right")
    cmap_after = {tid: COL_PASS for tid in cited_d2}

    if vol is not None:
        z_idx = find_best_slice(tokens_b2, cited_b2, vol.shape)
        midline_vol = vol.shape[0] / 2.0
    else:
        z_idx = 48
        midline_vol = 512

    print(f"  z_idx={z_idx}, midline={midline_vol:.0f}")

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5),
                             gridspec_kw={"height_ratios": [3, 2],
                                          "hspace": 0.22, "wspace": 0.20})

    if vol is not None:
        draw_ct_panel(axes[0, 0], vol, z_idx, tokens_b2, cited_b2,
                      cmap_before, midline_vol,
                      "(a) Before Repair — Violation Detected")
    draw_info_card(axes[1, 0], sent_b2, box_color="#FFF0E0")

    if vol is not None:
        draw_ct_panel(axes[0, 1], vol, z_idx, tokens_b2, cited_b2,
                      cmap_before, midline_vol,
                      "(b) Judge Diagnosis")
    draw_info_card(axes[1, 1], sent_d2, box_color="#FFF8E0", show_judge=True)

    if vol is not None:
        draw_ct_panel(axes[0, 2], vol, z_idx, tokens_d2, cited_d2,
                      cmap_after, midline_vol,
                      "(c) After Repair — PASS")
    draw_info_card(axes[1, 2], sent_d2, box_color="#E0F5E0", is_repaired=True)

    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COL_SAME,
               markersize=10, label='Same-side token (correct)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COL_OPP,
               markersize=10, label='Opposite-side token (offending)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=9, frameon=True, edgecolor="#CCCCCC",
               bbox_to_anchor=(0.5, 0.005))

    for ext in ("png", "pdf"):
        out = OUT_DIR / f"fig_case_study.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
