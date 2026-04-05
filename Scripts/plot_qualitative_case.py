#!/usr/bin/env python3
"""
Generate qualitative case figures for the paper (positive + negative examples).

Shows CT axial slices with overlaid token bounding boxes and generated sentences.
Generates two figures:
  - fig4a_qualitative_positive.pdf/png  (no violations)
  - fig4b_qualitative_negative.pdf/png  (with violations)

Usage:
    python Scripts/plot_qualitative_case.py
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
CASES_ROOT = ROOT / "outputs" / "stage0_5_5k" / "test" / "cases"
VOLUME_ROOT = ROOT / "dataset_5k" / "CT-RATE" / "train"

POSITIVE_CASE = {"dataset": "ctrate", "case_id": "train_10754_c_1"}
NEGATIVE_CASE = {"dataset": "ctrate", "case_id": "train_9163_a_1"}


def load_volume(path: Path) -> np.ndarray:
    """Load CT volume from .nii.gz file."""
    if nib is None:
        raise ImportError("nibabel required: pip install nibabel")
    img = nib.load(str(path))
    vol = np.array(img.dataobj, dtype=np.float32)
    return vol


def load_tokens(case_dir: Path) -> list:
    """Load token bounding boxes from tokens.json."""
    p = case_dir / "tokens.json"
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


def load_trace(case_dir: Path) -> list:
    """Load sentence records from trace.jsonl."""
    p = case_dir / "trace.jsonl"
    sentences = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            if d.get("type") == "sentence":
                sentences.append(d)
    return sentences


def pick_positive_sentence(sentences: list) -> dict:
    """Pick the best sentence for positive example (generated, with anatomy, no violations)."""
    candidates = [s for s in sentences
                  if s.get("generated", False)
                  and s.get("anatomy_keyword")
                  and not s.get("violations")]
    if candidates:
        specific = [s for s in candidates if s["anatomy_keyword"] != "bilateral"]
        if specific:
            return specific[0]
        return candidates[0]
    gen = [s for s in sentences if s.get("generated", False)]
    return gen[0] if gen else sentences[0]


def pick_negative_sentence(sentences: list) -> dict:
    """Pick the best sentence for negative example (generated, WITH violations)."""
    candidates = [s for s in sentences
                  if s.get("generated", False)
                  and s.get("violations")]
    if candidates:
        lat = [s for s in candidates
               if any(v.get("rule_id", "").startswith("R1") for v in s["violations"])]
        if lat:
            return lat[0]
        return candidates[0]
    gen = [s for s in sentences if s.get("generated", False)]
    return gen[0] if gen else sentences[0]


def find_best_slice(vol_shape: tuple, tokens: list, cited_ids: list) -> int:
    """Find the axial slice (z-index) that best shows the cited tokens."""
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


def _draw_panel(fig, pos, vol, slice_img, z_idx, tokens, cited_ids, sentence,
                panel_labels=("a", "b", "c"), title_suffix="",
                box_color="lightyellow"):
    """Draw a 3-panel row: (a) CT slice, (b) token overlay, (c) text card."""
    token_map = {t["token_id"]: t for t in tokens}
    text = sentence.get("sentence_text", "")
    anatomy = sentence.get("anatomy_keyword", "")
    violations = sentence.get("violations", [])

    if vol is not None and slice_img is not None:
        ax1 = fig.add_subplot(*pos, 1)
        ax1.imshow(slice_img.T, cmap="gray", origin="lower", aspect="equal")
        ax1.set_title(f"({panel_labels[0]}) CT Axial Slice (z={z_idx})", fontsize=10)
        ax1.set_xlabel("x (voxels)", fontsize=8)
        ax1.set_ylabel("y (voxels)", fontsize=8)
        ax1.tick_params(labelsize=7)

        ax2 = fig.add_subplot(*pos, 2)
        ax2.imshow(slice_img.T, cmap="gray", origin="lower", aspect="equal")

        colors_cited = plt.cm.Set1(np.linspace(0, 1, max(len(cited_ids), 1)))
        for i, tid in enumerate(cited_ids):
            t = token_map.get(tid)
            if t is None or "bbox_3d_voxel" not in t:
                continue
            bb = t["bbox_3d_voxel"]
            if bb["z_min"] <= z_idx <= bb["z_max"]:
                rect = patches.Rectangle(
                    (bb["x_min"], bb["y_min"]),
                    bb["x_max"] - bb["x_min"],
                    bb["y_max"] - bb["y_min"],
                    linewidth=1.5,
                    edgecolor=colors_cited[i % len(colors_cited)],
                    facecolor=colors_cited[i % len(colors_cited)],
                    alpha=0.25,
                )
                ax2.add_patch(rect)
                ax2.text(bb["x_min"] + 1, bb["y_min"] + 1,
                         f"t{tid}", fontsize=6, color="white",
                         fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))

        ax2.set_title(f"({panel_labels[1]}) Routed Tokens (k={len(cited_ids)})", fontsize=10)
        ax2.set_xlabel("x (voxels)", fontsize=8)
        ax2.set_ylabel("y (voxels)", fontsize=8)
        ax2.tick_params(labelsize=7)
    else:
        ax1 = fig.add_subplot(*pos, 1)
        ax1.text(0.5, 0.5, "CT Volume\n(not available)",
                ha="center", va="center", fontsize=14, color="gray",
                transform=ax1.transAxes)
        ax1.set_title(f"({panel_labels[0]}) CT Axial Slice", fontsize=10)

        ax2 = fig.add_subplot(*pos, 2)
        for i, tid in enumerate(cited_ids):
            t = token_map.get(tid)
            if t is None or "bbox_3d_voxel" not in t:
                continue
            bb = t["bbox_3d_voxel"]
            rect = patches.Rectangle(
                (bb["x_min"], bb["y_min"]),
                bb["x_max"] - bb["x_min"],
                bb["y_max"] - bb["y_min"],
                linewidth=1.5, edgecolor=f"C{i}", facecolor=f"C{i}", alpha=0.3,
            )
            ax2.add_patch(rect)
            ax2.text(bb["x_min"], bb["y_min"], f"t{tid}", fontsize=7, color=f"C{i}")
        all_x = [t["bbox_3d_voxel"]["x_max"] for t in tokens if "bbox_3d_voxel" in t]
        all_y = [t["bbox_3d_voxel"]["y_max"] for t in tokens if "bbox_3d_voxel" in t]
        if all_x and all_y:
            ax2.set_xlim(0, max(all_x))
            ax2.set_ylim(0, max(all_y))
        ax2.set_title(f"({panel_labels[1]}) Routed Tokens (k={len(cited_ids)})", fontsize=10)
        ax2.set_aspect("equal")

    ax3 = fig.add_subplot(*pos, 3)
    ax3.axis("off")

    evidence_card = sentence.get("evidence_card", {})
    ec_text = ""
    if evidence_card:
        ssr = evidence_card.get("same_side_ratio", "?")
        ssr_str = f"{ssr:.2f}" if isinstance(ssr, (int, float)) else str(ssr)
        ec_text = (
            f"Evidence Card:\n"
            f"  Cited tokens: {evidence_card.get('cited_tokens', '?')}\n"
            f"  Dominant side: {evidence_card.get('dominant_side', '?')}\n"
            f"  Same-side ratio: {ssr_str}\n"
            f"  Laterality gate: {evidence_card.get('laterality_allowed', '?')}\n"
            f"  Depth gate: {evidence_card.get('depth_allowed', '?')}\n"
        )

    wrapped_text = textwrap.fill(text, width=42)
    viol_text = "None" if not violations else "\n  ".join(
        f"{v.get('rule_id', '?')}: {v.get('message', '')[:60]}" for v in violations
    )

    display_text = (
        f"Generated Sentence:\n"
        f"{'─' * 38}\n"
        f"{wrapped_text}\n\n"
        f"{'─' * 38}\n"
        f"Anatomy: {anatomy}\n"
        f"Token IDs: {cited_ids}\n"
        f"Violations: {viol_text}\n\n"
        f"{ec_text}"
    )

    ax3.text(0.05, 0.95, display_text,
             transform=ax3.transAxes,
             fontsize=8, fontfamily="monospace",
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.8))
    label = f"({panel_labels[2]}) Generated Output"
    if title_suffix:
        label += f" {title_suffix}"
    ax3.set_title(label, fontsize=10)


def _load_case_data(dataset: str, case_id: str):
    """Load all data for a case. Returns (case_dir, tokens, sentences, vol, slice_img, z_idx)."""
    case_dir = CASES_ROOT / dataset / case_id
    volume_path = VOLUME_ROOT / f"{case_id}.nii.gz"

    tokens = load_tokens(case_dir)
    sentences = load_trace(case_dir)

    vol = None
    slice_img = None
    z_idx = None
    if volume_path.exists():
        vol = load_volume(volume_path)
        if len(vol.shape) == 4:
            vol = vol[..., 0]

    return case_dir, tokens, sentences, vol


def plot_single_case(case_def: dict, pick_fn, fig_name: str,
                     title_suffix: str, box_color: str, panel_labels: tuple):
    """Generate a single qualitative figure."""
    dataset = case_def["dataset"]
    case_id = case_def["case_id"]
    print(f"\n{'=' * 60}")
    print(f"Generating {fig_name}: {dataset}/{case_id}")

    case_dir, tokens, sentences, vol = _load_case_data(dataset, case_id)
    if not sentences:
        print(f"  No sentences found in {case_dir}")
        return

    sentence = pick_fn(sentences)
    cited_ids = sentence.get("topk_token_ids", [])
    text = sentence.get("sentence_text", "")
    anatomy = sentence.get("anatomy_keyword", "")
    violations = sentence.get("violations", [])

    print(f"  Sentence: {text[:80]}...")
    print(f"  Anatomy: {anatomy}")
    print(f"  Violations: {[v.get('rule_id') for v in violations] if violations else 'None'}")
    print(f"  Cited tokens: {cited_ids}")

    slice_img = None
    z_idx = None
    if vol is not None:
        z_idx = find_best_slice(vol.shape, tokens, cited_ids)
        slice_img = vol[:, :, z_idx]
        print(f"  Volume shape: {vol.shape}, slice z={z_idx}")
    else:
        print(f"  Volume not available, using schematic layout")

    fig = plt.figure(figsize=(14, 5))
    _draw_panel(fig, (1, 3), vol, slice_img, z_idx, tokens, cited_ids, sentence,
                panel_labels=panel_labels, title_suffix=title_suffix,
                box_color=box_color)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = OUT_DIR / f"{fig_name}.{ext}"
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved to {OUT_DIR}/{fig_name}.pdf/png")


def plot_combined():
    """Generate a combined 2-row figure with positive (top) and negative (bottom)."""
    print(f"\n{'=' * 60}")
    print("Generating combined figure: fig4_qualitative_combined")

    _, tokens_pos, sents_pos, vol_pos = _load_case_data(
        POSITIVE_CASE["dataset"], POSITIVE_CASE["case_id"])
    sent_pos = pick_positive_sentence(sents_pos)
    cited_pos = sent_pos.get("topk_token_ids", [])

    _, tokens_neg, sents_neg, vol_neg = _load_case_data(
        NEGATIVE_CASE["dataset"], NEGATIVE_CASE["case_id"])
    sent_neg = pick_negative_sentence(sents_neg)
    cited_neg = sent_neg.get("topk_token_ids", [])

    fig = plt.figure(figsize=(14, 10))

    slice_pos = None
    z_pos = None
    if vol_pos is not None:
        if len(vol_pos.shape) == 4:
            vol_pos = vol_pos[..., 0]
        z_pos = find_best_slice(vol_pos.shape, tokens_pos, cited_pos)
        slice_pos = vol_pos[:, :, z_pos]

    _draw_panel(fig, (2, 3), vol_pos, slice_pos, z_pos,
                tokens_pos, cited_pos, sent_pos,
                panel_labels=("a", "b", "c"),
                title_suffix="[Pass]",
                box_color="honeydew")

    slice_neg = None
    z_neg = None
    if vol_neg is not None:
        if len(vol_neg.shape) == 4:
            vol_neg = vol_neg[..., 0]
        z_neg = find_best_slice(vol_neg.shape, tokens_neg, cited_neg)
        slice_neg = vol_neg[:, :, z_neg]

    token_map_neg = {t["token_id"]: t for t in tokens_neg}
    text_neg = sent_neg.get("sentence_text", "")
    anatomy_neg = sent_neg.get("anatomy_keyword", "")
    violations_neg = sent_neg.get("violations", [])

    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    if vol_neg is not None and slice_neg is not None:
        ax4.imshow(slice_neg.T, cmap="gray", origin="lower", aspect="equal")
        ax4.set_title(f"(d) CT Axial Slice (z={z_neg})", fontsize=10)
        ax4.set_xlabel("x (voxels)", fontsize=8)
        ax4.set_ylabel("y (voxels)", fontsize=8)
        ax4.tick_params(labelsize=7)

        ax5.imshow(slice_neg.T, cmap="gray", origin="lower", aspect="equal")
        colors_cited = plt.cm.Set1(np.linspace(0, 1, max(len(cited_neg), 1)))
        for i, tid in enumerate(cited_neg):
            t = token_map_neg.get(tid)
            if t is None or "bbox_3d_voxel" not in t:
                continue
            bb = t["bbox_3d_voxel"]
            if bb["z_min"] <= z_neg <= bb["z_max"]:
                rect = patches.Rectangle(
                    (bb["x_min"], bb["y_min"]),
                    bb["x_max"] - bb["x_min"],
                    bb["y_max"] - bb["y_min"],
                    linewidth=1.5,
                    edgecolor=colors_cited[i % len(colors_cited)],
                    facecolor=colors_cited[i % len(colors_cited)],
                    alpha=0.25,
                )
                ax5.add_patch(rect)
                ax5.text(bb["x_min"] + 1, bb["y_min"] + 1,
                         f"t{tid}", fontsize=6, color="white", fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))
        ax5.set_title(f"(e) Routed Tokens (k={len(cited_neg)})", fontsize=10)
        ax5.set_xlabel("x (voxels)", fontsize=8)
        ax5.set_ylabel("y (voxels)", fontsize=8)
        ax5.tick_params(labelsize=7)
    else:
        ax4.text(0.5, 0.5, "CT Volume\n(not available)",
                ha="center", va="center", fontsize=14, color="gray",
                transform=ax4.transAxes)
        ax4.set_title("(d) CT Axial Slice", fontsize=10)

        for i, tid in enumerate(cited_neg):
            t = token_map_neg.get(tid)
            if t is None or "bbox_3d_voxel" not in t:
                continue
            bb = t["bbox_3d_voxel"]
            rect = patches.Rectangle(
                (bb["x_min"], bb["y_min"]),
                bb["x_max"] - bb["x_min"],
                bb["y_max"] - bb["y_min"],
                linewidth=1.5, edgecolor=f"C{i}", facecolor=f"C{i}", alpha=0.3,
            )
            ax5.add_patch(rect)
            ax5.text(bb["x_min"], bb["y_min"], f"t{tid}", fontsize=7, color=f"C{i}")
        all_x = [t["bbox_3d_voxel"]["x_max"] for t in tokens_neg if "bbox_3d_voxel" in t]
        all_y = [t["bbox_3d_voxel"]["y_max"] for t in tokens_neg if "bbox_3d_voxel" in t]
        if all_x and all_y:
            ax5.set_xlim(0, max(all_x))
            ax5.set_ylim(0, max(all_y))
        ax5.set_title(f"(e) Routed Tokens (k={len(cited_neg)})", fontsize=10)
        ax5.set_aspect("equal")

    ax6.axis("off")
    evidence_card = sent_neg.get("evidence_card", {})
    ec_text = ""
    if evidence_card:
        ssr = evidence_card.get("same_side_ratio", "?")
        ssr_str = f"{ssr:.2f}" if isinstance(ssr, (int, float)) else str(ssr)
        ec_text = (
            f"Evidence Card:\n"
            f"  Cited tokens: {evidence_card.get('cited_tokens', '?')}\n"
            f"  Dominant side: {evidence_card.get('dominant_side', '?')}\n"
            f"  Same-side ratio: {ssr_str}\n"
            f"  Laterality gate: {evidence_card.get('laterality_allowed', '?')}\n"
            f"  Depth gate: {evidence_card.get('depth_allowed', '?')}\n"
        )
    wrapped_text = textwrap.fill(text_neg, width=42)
    viol_text = "\n  ".join(
        f"{v.get('rule_id', '?')}: {v.get('message', '')[:60]}" for v in violations_neg
    ) if violations_neg else "None"

    display_text = (
        f"Generated Sentence:\n"
        f"{'─' * 38}\n"
        f"{wrapped_text}\n\n"
        f"{'─' * 38}\n"
        f"Anatomy: {anatomy_neg}\n"
        f"Token IDs: {cited_neg}\n"
        f"Violations: {viol_text}\n\n"
        f"{ec_text}"
    )
    ax6.text(0.05, 0.95, display_text,
             transform=ax6.transAxes,
             fontsize=8, fontfamily="monospace",
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="mistyrose", alpha=0.8))
    ax6.set_title("(f) Generated Output [Violation]", fontsize=10)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = OUT_DIR / f"fig4_qualitative_combined.{ext}"
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved to {OUT_DIR}/fig4_qualitative_combined.pdf/png")


def main():
    plot_single_case(
        POSITIVE_CASE, pick_positive_sentence,
        fig_name="fig4a_qualitative_positive",
        title_suffix="[Pass]",
        box_color="honeydew",
        panel_labels=("a", "b", "c"),
    )
    plot_single_case(
        NEGATIVE_CASE, pick_negative_sentence,
        fig_name="fig4b_qualitative_negative",
        title_suffix="[Violation]",
        box_color="mistyrose",
        panel_labels=("d", "e", "f"),
    )

    plot_combined()

    plot_single_case(
        POSITIVE_CASE, pick_positive_sentence,
        fig_name="fig4_qualitative",
        title_suffix="",
        box_color="lightyellow",
        panel_labels=("a", "b", "c"),
    )

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
