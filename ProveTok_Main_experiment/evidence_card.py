"""
Evidence Card: structured summary of cited token spatial properties.

Shared by Stage 3c (generation constraint) and Stage 5 (repair decision).
Computes laterality distribution, depth histogram, and dominant attributes
from the set of cited evidence tokens.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .types import EvidenceToken


@dataclass
class EvidenceCard:
    """Structured summary of a set of cited evidence tokens."""

    cited_count: int = 0

    # Laterality distribution
    left_count: int = 0
    right_count: int = 0
    cross_count: int = 0  # tokens straddling the midline
    dominant_side: str = "unknown"  # "left" | "right" | "bilateral" | "mixed" | "unknown"
    same_side_ratio: float = 1.0  # max(left, right) / (left + right), or 1.0 if none

    # Depth / level distribution
    level_histogram: Dict[int, int] = field(default_factory=dict)
    level_range_expected: Optional[Tuple[int, int]] = None
    in_range_count: int = 0
    out_of_range_count: int = 0

    def to_prompt_dict(self) -> Dict[str, object]:
        """Serialize to a dict suitable for inclusion in LLM prompts."""
        d: Dict[str, object] = {
            "cited_tokens": self.cited_count,
            "left_count": self.left_count,
            "right_count": self.right_count,
            "dominant_side": self.dominant_side,
            "same_side_ratio": round(self.same_side_ratio, 3),
            "level_histogram": {str(k): v for k, v in sorted(self.level_histogram.items())},
        }
        if self.level_range_expected is not None:
            d["level_range_expected"] = list(self.level_range_expected)
        return d

    def laterality_allowed(self) -> str:
        """
        Return what laterality terms the generated text is allowed to use.

        - "left" or "right": evidence clearly supports one side
        - "bilateral": evidence clearly supports both sides
        - "none": evidence is mixed or insufficient; do NOT mention laterality
        """
        if self.dominant_side in ("left", "right"):
            return self.dominant_side
        if self.dominant_side == "bilateral":
            return "bilateral"
        # mixed or unknown => do not mention laterality
        return "none"


def _token_side(token: EvidenceToken, x_mid: float, tol: float) -> str:
    """Classify token as left/right/cross based on center x vs midline."""
    cx = token.bbox.center()[0]
    if cx > x_mid + tol:
        return "left"
    if cx < x_mid - tol:
        return "right"
    return "cross"


def build_evidence_card(
    tokens: Sequence[EvidenceToken],
    x_mid: float,
    lateral_tolerance: float = 0.0,
    expected_level_range: Optional[Tuple[int, int]] = None,
    same_side_threshold: float = 0.8,
) -> EvidenceCard:
    """
    Build an EvidenceCard from a set of cited tokens.

    Args:
        tokens: the cited evidence tokens (top-k from routing)
        x_mid: x-coordinate of the body midline (volume_width / 2)
        lateral_tolerance: tolerance band around midline for "cross" classification
        expected_level_range: (lo, hi) expected depth levels from sentence plan
        same_side_threshold: ratio threshold above which we declare a dominant side.
            E.g., 0.8 means 80% of non-cross tokens must be on one side.
    """
    card = EvidenceCard(cited_count=len(tokens))

    if not tokens:
        return card

    # --- Laterality ---
    for tok in tokens:
        side = _token_side(tok, x_mid, lateral_tolerance)
        if side == "left":
            card.left_count += 1
        elif side == "right":
            card.right_count += 1
        else:
            card.cross_count += 1

    non_cross = card.left_count + card.right_count
    if non_cross > 0:
        majority = max(card.left_count, card.right_count)
        card.same_side_ratio = majority / non_cross

        if card.same_side_ratio >= same_side_threshold:
            if card.left_count >= card.right_count:
                card.dominant_side = "left"
            else:
                card.dominant_side = "right"
        elif card.left_count > 0 and card.right_count > 0:
            # Both sides present but neither dominant enough
            min_side = min(card.left_count, card.right_count)
            if min_side / non_cross >= 0.3:
                # Substantial presence on both sides => bilateral
                card.dominant_side = "bilateral"
            else:
                card.dominant_side = "mixed"
        else:
            card.dominant_side = "mixed"
    else:
        # All tokens on midline
        card.same_side_ratio = 1.0
        card.dominant_side = "unknown"

    # --- Depth / Level ---
    level_counter: Counter = Counter()
    for tok in tokens:
        level_counter[tok.level] += 1
    card.level_histogram = dict(level_counter)

    card.level_range_expected = expected_level_range
    if expected_level_range is not None:
        lo, hi = expected_level_range
        card.in_range_count = sum(1 for tok in tokens if lo <= tok.level <= hi)
        card.out_of_range_count = len(tokens) - card.in_range_count

    return card
