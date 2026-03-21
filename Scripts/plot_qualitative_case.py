#!/usr/bin/env python3
"""
Generate qualitative case figure for the paper.

Shows a CT axial slice with overlaid token bounding boxes and the generated sentence.

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

# ── Case selection ──
CASE_ID = "train_10754_c_1"
DATASET = "ctrate"
CASE_DIR = ROOT / "outputs" / "stage0_5_5k" / "test" / "cases" / DATASET / CASE_ID
VOLUME_PATH = ROOT / "dataset_5k" / "CT-RATE" / "train" / f"{CASE_ID}.nii.gz"


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


def pick_best_sentence(sentences: list) -> dict:
    """Pick the best sentence for visualization (generated, with anatomy, no violations)."""
    candidates = [s for s in sentences
                  if s.get("generated", False)
                  and s.get("anatomy_keyword")
                  and not s.get("violations")]
    if candidates:
        # Prefer specific anatomy (not bilateral)
        specific = [s for s in candidates if s["anatomy_keyword"] != "bilateral"]
        if specific:
            return specific[0]
        return candidates[0]
    # Fallback: any generated sentence
    gen = [s for s in sentences if s.get("generated", False)]
    return gen[0] if gen else sentences[0]


def find_best_slice(vol: np.ndarray, tokens: list, cited_ids: list) -> int:
    """Find the axial slice (z-index) that best shows the cited tokens."""
    # Cited token z-ranges (in voxel coordinates)
    token_map = {t["token_id"]: t for t in tokens}
    z_centers = []
    for tid in cited_ids:
        t = token_map.get(tid)
        if t and "bbox_3d_voxel" in t:
            bb = t["bbox_3d_voxel"]
            z_centers.append((bb["z_min"] + bb["z_max"]) / 2.0)
    if z_centers:
        return int(np.median(z_centers))
    return vol.shape[2] // 2  # fallback: middle slice


def plot_case():
    """Generate the qualitative case figure."""
    print("Loading data...")

    # Load trace
    sentences = load_trace(CASE_DIR)
    if not sentences:
        print(f"No sentences found in {CASE_DIR}")
        return

    sentence = pick_best_sentence(sentences)
    cited_ids = sentence.get("topk_token_ids", [])
    text = sentence.get("sentence_text", "")
    anatomy = sentence.get("anatomy_keyword", "")
    print(f"Selected sentence: anatomy={anatomy}")
    print(f"  Text: {text[:100]}...")
    print(f"  Cited tokens: {cited_ids}")

    # Load tokens
    tokens = load_tokens(CASE_DIR)
    token_map = {t["token_id"]: t for t in tokens}

    # Check if volume exists
    has_volume = VOLUME_PATH.exists()
    if has_volume:
        print(f"Loading volume: {VOLUME_PATH}")
        vol = load_volume(VOLUME_PATH)
        print(f"  Volume shape: {vol.shape}")
    else:
        print(f"Volume not found at {VOLUME_PATH}, using placeholder")
        vol = None

    # ── Create figure ──
    fig = plt.figure(figsize=(14, 5))

    if vol is not None:
        # Find best axial slice
        z_idx = find_best_slice(vol, tokens, cited_ids)
        # Volume is typically (D, H, W) or (X, Y, Z)
        # Take an axial slice
        if len(vol.shape) == 4:
            vol = vol[..., 0]  # drop channel dim if present

        # Try different axis orders
        d, h, w = vol.shape
        slice_img = vol[:, :, z_idx]  # axial slice at z_idx

        # ── Panel (a): CT slice without overlay ──
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(slice_img.T, cmap="gray", origin="lower", aspect="equal")
        ax1.set_title(f"(a) CT Axial Slice (z={z_idx})", fontsize=11)
        ax1.set_xlabel("x (voxels)")
        ax1.set_ylabel("y (voxels)")

        # ── Panel (b): CT slice with token bbox overlay ──
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(slice_img.T, cmap="gray", origin="lower", aspect="equal")

        # Draw bounding boxes for cited tokens
        colors_cited = plt.cm.Set1(np.linspace(0, 1, max(len(cited_ids), 1)))
        for i, tid in enumerate(cited_ids):
            t = token_map.get(tid)
            if t is None or "bbox_3d_voxel" not in t:
                continue
            bb = t["bbox_3d_voxel"]
            # Only draw if this slice is within the token's z-range
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
                # Add small label
                ax2.text(bb["x_min"] + 1, bb["y_min"] + 1,
                         f"t{tid}", fontsize=6, color="white",
                         fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))

        ax2.set_title(f"(b) Routed Tokens (k={len(cited_ids)})", fontsize=11)
        ax2.set_xlabel("x (voxels)")
        ax2.set_ylabel("y (voxels)")

    else:
        # No volume — show token layout schematically
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.text(0.5, 0.5, "CT Volume\n(not available)",
                ha="center", va="center", fontsize=14, color="gray",
                transform=ax1.transAxes)
        ax1.set_title("(a) CT Axial Slice", fontsize=11)

        ax2 = fig.add_subplot(1, 3, 2)
        # Draw token bboxes schematically
        for i, tid in enumerate(cited_ids):
            t = token_map.get(tid)
            if t is None or "bbox_3d_voxel" not in t:
                continue
            bb = t["bbox_3d_voxel"]
            rect = patches.Rectangle(
                (bb["x_min"], bb["y_min"]),
                bb["x_max"] - bb["x_min"],
                bb["y_max"] - bb["y_min"],
                linewidth=1.5,
                edgecolor=f"C{i}",
                facecolor=f"C{i}",
                alpha=0.3,
            )
            ax2.add_patch(rect)
            ax2.text(bb["x_min"], bb["y_min"], f"t{tid}", fontsize=7, color=f"C{i}")

        # Set axis limits from all token bboxes
        all_x = [t["bbox_3d_voxel"]["x_max"] for t in tokens if "bbox_3d_voxel" in t]
        all_y = [t["bbox_3d_voxel"]["y_max"] for t in tokens if "bbox_3d_voxel" in t]
        if all_x and all_y:
            ax2.set_xlim(0, max(all_x))
            ax2.set_ylim(0, max(all_y))
        ax2.set_title(f"(b) Routed Tokens (k={len(cited_ids)})", fontsize=11)
        ax2.set_aspect("equal")

    # ── Panel (c): Generated text with evidence card ──
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis("off")

    # Build text content
    evidence_card = sentence.get("evidence_card", {})
    ec_text = ""
    if evidence_card:
        ec_text = (
            f"Evidence Card:\n"
            f"  Cited tokens: {evidence_card.get('cited_tokens', '?')}\n"
            f"  Dominant side: {evidence_card.get('dominant_side', '?')}\n"
            f"  Same-side ratio: {evidence_card.get('same_side_ratio', '?'):.2f}\n"
            f"  Laterality gate: {evidence_card.get('laterality_allowed', '?')}\n"
            f"  Depth gate: {evidence_card.get('depth_allowed', '?')}\n"
        )

    wrapped_text = textwrap.fill(text, width=45)
    violations = sentence.get("violations", [])
    viol_text = "None" if not violations else "; ".join(
        v.get("rule_id", "?") for v in violations
    )

    display_text = (
        f"Generated Sentence:\n"
        f"{'─' * 40}\n"
        f"{wrapped_text}\n\n"
        f"{'─' * 40}\n"
        f"Anatomy: {anatomy}\n"
        f"Token IDs: {cited_ids}\n"
        f"Violations: {viol_text}\n\n"
        f"{ec_text}"
    )

    ax3.text(0.05, 0.95, display_text,
             transform=ax3.transAxes,
             fontsize=9, fontfamily="monospace",
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax3.set_title("(c) Generated Output", fontsize=11)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig4_qualitative.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(OUT_DIR / "fig4_qualitative.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nSaved to {OUT_DIR}/fig4_qualitative.pdf")


if __name__ == "__main__":
    plot_case()
