from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .config import VerifierConfig
from .math_utils import clamp
from .types import BBox3D, EvidenceToken, RuleViolation, SentenceAudit, SentenceOutput, SentencePlan


LEFT_WORDS = ("left", "lt", "左")
RIGHT_WORDS = ("right", "rt", "右")
BILATERAL_WORDS = ("bilateral", "both", "双侧")
NEGATION_WORDS = ("no ", "without", "not ", "未见", "无")
POSITIVE_FINDING_WORDS = (
    "nodule",
    "mass",
    "lesion",
    "opacity",
    "consolidation",
    "infiltrate",
    "effusion",
    "edema",
    "pneumothorax",
    "ground glass",
    "atelectasis",
    "embol",
    "thrombus",
    "fibrosis",
)


def parse_laterality(text: str) -> Optional[str]:
    t = text.lower()
    has_left = any(w in t for w in LEFT_WORDS)
    has_right = any(w in t for w in RIGHT_WORDS)
    has_bi = any(w in t for w in BILATERAL_WORDS)
    if has_bi or (has_left and has_right):
        return "bilateral"
    if has_left:
        return "left"
    if has_right:
        return "right"
    return None


def detect_negation(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in NEGATION_WORDS)


def token_side(token: EvidenceToken, x_mid: float, tol: float) -> str:
    """
    CP uses center-vs-midline side check.
    """
    x_center = token.bbox.center()[0]
    if x_center > x_mid + tol:
        return "left"
    if x_center < x_mid - tol:
        return "right"
    return "cross"


@dataclass
class Verifier:
    cfg: VerifierConfig
    anatomy_bbox_resolver: Callable[[Optional[str]], Optional[BBox3D]]
    volume_shape: Optional[Tuple[int, int, int]] = None

    def _global_midline_x(self, tokens: Sequence[EvidenceToken]) -> float:
        if self.volume_shape is not None:
            # volume_shape is [D,H,W], x-axis maps to W.
            return float(self.volume_shape[2]) / 2.0
        x_min = min(t.bbox.x_min for t in tokens)
        x_max = max(t.bbox.x_max for t in tokens)
        return (x_min + x_max) / 2.0

    def audit_sentence(
        self,
        sentence: SentenceOutput,
        plan: SentencePlan,
        all_tokens: Sequence[EvidenceToken],
        token_map: Dict[int, EvidenceToken],
    ) -> SentenceAudit:
        cited = [token_map[tid] for tid in sentence.citations if tid in token_map]
        violations: List[RuleViolation] = []
        x_mid = self._global_midline_x(all_tokens)
        side_claim = parse_laterality(sentence.text)
        negated = plan.is_negated or detect_negation(sentence.text)

        # R1 laterality consistency
        r1_skipped = (self.cfg.r1_negation_exempt and negated) or bool(
            plan.anatomy_keyword
            and plan.anatomy_keyword.lower() in self.cfg.r1_skip_midline_keywords
        )
        if side_claim in ("left", "right") and not r1_skipped:
            same_ids: List[int] = []
            opp_ids: List[int] = []
            for tok in cited:
                side = token_side(tok, x_mid, self.cfg.lateral_tolerance)
                if side == side_claim:
                    same_ids.append(tok.token_id)
                elif side != "cross":
                    opp_ids.append(tok.token_id)
            non_cross = len(same_ids) + len(opp_ids)
            same_ratio = len(same_ids) / non_cross if non_cross > 0 else 1.0
            if same_ratio < self.cfg.r1_min_same_side_ratio:
                violations.append(
                    RuleViolation(
                        sentence_index=sentence.sentence_index,
                        rule_id="R1_LATERALITY",
                        severity=self.cfg.severity_by_rule["R1_LATERALITY"],
                        message=f"Laterality mismatch with claim={side_claim} (same_side_ratio={same_ratio:.3f}).",
                        token_ids=opp_ids,
                    )
                )

        # R2 anatomy IoU consistency
        anatomy_bbox = self.anatomy_bbox_resolver(plan.anatomy_keyword)
        r2_skipped = bool(
            plan.anatomy_keyword
            and plan.anatomy_keyword.lower() in self.cfg.r2_skip_keywords
        )
        if anatomy_bbox is not None and cited and not r2_skipped:
            iou_by_tok = {tok.token_id: float(tok.bbox.iou(anatomy_bbox)) for tok in cited}
            if self.cfg.use_max_iou_for_r2:
                max_iou = max(iou_by_tok.values())
                if max_iou < self.cfg.tau_anatomy_iou:
                    sev = self.cfg.severity_by_rule["R2_ANATOMY"] * (
                        1.0 - (max_iou / max(self.cfg.tau_anatomy_iou, 1e-8))
                    )
                    violations.append(
                        RuleViolation(
                            sentence_index=sentence.sentence_index,
                            rule_id="R2_ANATOMY",
                            severity=clamp(sev, 0.0, 1.0),
                            message="Anatomy IoU below threshold (max IoU mode).",
                            token_ids=[tok.token_id for tok in cited],
                        )
                    )
            else:
                bad_ids = [tid for tid, iou in iou_by_tok.items() if iou < self.cfg.tau_anatomy_iou]
                good_ratio = 1.0 - (len(bad_ids) / max(len(cited), 1))
                if good_ratio < self.cfg.r2_min_support_ratio:
                    gap_terms = [
                        (self.cfg.tau_anatomy_iou - iou) / max(self.cfg.tau_anatomy_iou, 1e-8)
                        for iou in iou_by_tok.values()
                        if iou < self.cfg.tau_anatomy_iou
                    ]
                    mean_gap = float(sum(gap_terms) / max(len(gap_terms), 1))
                    ratio_gap = float(self.cfg.r2_min_support_ratio - good_ratio)
                    sev_scale = clamp(max(mean_gap, ratio_gap), 0.0, 1.0)
                    sev = self.cfg.severity_by_rule["R2_ANATOMY"] * sev_scale
                    violations.append(
                        RuleViolation(
                            sentence_index=sentence.sentence_index,
                            rule_id="R2_ANATOMY",
                            severity=clamp(sev, 0.0, 1.0),
                            message=(
                                f"Anatomy support ratio={good_ratio:.3f} below required "
                                f"{self.cfg.r2_min_support_ratio:.3f}."
                            ),
                            token_ids=bad_ids,
                        )
                    )

        # R3 depth-level consistency
        if plan.expected_level_range and cited:
            low, high = plan.expected_level_range
            bad_ids = [tok.token_id for tok in cited if not (low <= tok.level <= high)]
            if bad_ids:
                violations.append(
                    RuleViolation(
                        sentence_index=sentence.sentence_index,
                        rule_id="R3_DEPTH",
                        severity=self.cfg.severity_by_rule["R3_DEPTH"],
                        message=f"Expected token levels in [{low}, {high}].",
                        token_ids=bad_ids,
                    )
                )

        # R4 size/range consistency
        if not self.cfg.r4_disabled and plan.expected_volume_range and cited:
            union_box = BBox3D.union_all([tok.bbox for tok in cited])
            if union_box is not None:
                v = union_box.volume()
                lo, hi = plan.expected_volume_range
                if v < lo or v > hi:
                    violations.append(
                        RuleViolation(
                            sentence_index=sentence.sentence_index,
                            rule_id="R4_SIZE",
                            severity=self.cfg.severity_by_rule["R4_SIZE"],
                            message=f"Union bbox volume={v:.3f} outside [{lo:.3f}, {hi:.3f}].",
                            token_ids=[tok.token_id for tok in cited],
                        )
                    )

        # R5 negation handling
        if negated and cited:
            conflicted = [
                tok for tok in cited if float(tok.metadata.get("negation_conflict", 0.0)) > 0.0
            ]
            sev = 0.0
            msg = "Negated sentence cites evidence likely supporting positive finding."
            if conflicted:
                sev = max(float(tok.metadata.get("negation_conflict", 0.0)) for tok in conflicted)
                sev = clamp(sev, 0.0, 1.0) * self.cfg.severity_by_rule["R5_NEGATION"]
            elif self.cfg.r5_fallback_lexicon:
                t = sentence.text.lower()
                has_positive_word = any(w in t for w in POSITIVE_FINDING_WORDS)
                if has_positive_word:
                    conflicted = list(cited)
                    sev = clamp(self.cfg.r5_fallback_severity, 0.0, 1.0) * self.cfg.severity_by_rule["R5_NEGATION"]
                    msg = "Negation fallback: positive finding lexicon detected in negated sentence."

            if conflicted:
                violations.append(
                    RuleViolation(
                        sentence_index=sentence.sentence_index,
                        rule_id="R5_NEGATION",
                        severity=clamp(sev, 0.0, 1.0),
                        message=msg,
                        token_ids=[tok.token_id for tok in conflicted],
                    )
                )

        return SentenceAudit(
            sentence_index=sentence.sentence_index,
            passed=(len(violations) == 0),
            violations=violations,
        )

    def audit_all(
        self,
        sentence_outputs: Sequence[SentenceOutput],
        plans: Sequence[SentencePlan],
        all_tokens: Sequence[EvidenceToken],
    ) -> List[SentenceAudit]:
        token_map = {t.token_id: t for t in all_tokens}
        audits: List[SentenceAudit] = []
        for s_out, plan in zip(sentence_outputs, plans):
            audits.append(self.audit_sentence(s_out, plan, all_tokens, token_map))
        return audits

    def cross_sentence_check(
        self,
        sentence_outputs: Sequence[SentenceOutput],
        plans: Sequence[SentencePlan],
    ) -> List[RuleViolation]:
        """
        Cross-sentence consistency check (R6).

        Detects inter-sentence contradictions:
          - R6a LATERALITY: same anatomy keyword, conflicting laterality claims
          - R6b PRESENCE: same anatomy keyword, one positive and one negated
        """
        violations: List[RuleViolation] = []

        # Group sentences by anatomy keyword
        by_anatomy: Dict[str, List[Tuple[SentenceOutput, SentencePlan]]] = {}
        for s_out, plan in zip(sentence_outputs, plans):
            kw = (plan.anatomy_keyword or "").lower().strip()
            if not kw:
                continue
            by_anatomy.setdefault(kw, []).append((s_out, plan))

        for kw, group in by_anatomy.items():
            if len(group) < 2:
                continue

            # R6a: laterality conflict
            lateralities: Dict[str, List[int]] = {}
            for s_out, _ in group:
                side = parse_laterality(s_out.text)
                if side and side != "bilateral":
                    lateralities.setdefault(side, []).append(s_out.sentence_index)

            if "left" in lateralities and "right" in lateralities:
                left_ids = lateralities["left"]
                right_ids = lateralities["right"]
                violations.append(RuleViolation(
                    sentence_index=left_ids[0],
                    rule_id="R6a_CROSS_LATERALITY",
                    severity=0.7,
                    message=(
                        f"Cross-sentence laterality conflict for '{kw}': "
                        f"sentences {left_ids} claim left, {right_ids} claim right"
                    ),
                    token_ids=[],
                ))

            # R6b: presence/absence conflict
            pos_sents: List[int] = []
            neg_sents: List[int] = []
            for s_out, plan in group:
                is_neg = plan.is_negated or detect_negation(s_out.text)
                has_finding = any(
                    w in s_out.text.lower() for w in POSITIVE_FINDING_WORDS
                )
                if is_neg and has_finding:
                    neg_sents.append(s_out.sentence_index)
                elif has_finding:
                    pos_sents.append(s_out.sentence_index)

            if pos_sents and neg_sents:
                violations.append(RuleViolation(
                    sentence_index=pos_sents[0],
                    rule_id="R6b_CROSS_PRESENCE",
                    severity=0.8,
                    message=(
                        f"Cross-sentence presence/absence conflict for '{kw}': "
                        f"sentences {pos_sents} positive, {neg_sents} negated"
                    ),
                    token_ids=[],
                ))

        return violations
