"""
Stage 3c: Token-Gated LLM Report Generation (CP .tex core contribution).

Given:
  - top-k Evidence Tokens routed by Stage 3
  - sentence plan (anatomy keyword, topic)
  - trained W_proj (optional; falls back to identity)

Output:
  - LLM-generated report sentence conditioned on spatial token context

Two modes:
  1. text-only: format token metadata as structured text prompt (no W_proj needed)
  2. visual (VLM): extract CT voxel patches → send to Qwen2-VL / LLaVA (requires volume)

Usage (standalone generation):
    python -m ProveTok_Main_experiment.stage3c_generator \
        --trace_jsonl outputs/.../trace.jsonl \
        --tokens_pt  outputs/.../tokens.pt \
        --llm_backend ollama \
        --llm_model qwen2.5:7b
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .evidence_card import EvidenceCard
from .types import EvidenceToken, SentencePlan


# Spatial localization words to strip during de-specification fallback.
_SPATIAL_WORDS = [
    # laterality
    r"\bright\b", r"\bleft\b", r"\bbilateral(ly)?\b",
    # axial position
    r"\bupper\b", r"\blower\b", r"\bmiddle\b",
    r"\banterior\b", r"\bposterior\b",
    r"\bsuperior\b", r"\binferior\b",
    r"\bmedial\b", r"\blateral\b",
    # lobe-specific
    r"\bapical\b", r"\bbasal\b",
    r"\bhilar\b", r"\bperihilar\b", r"\bsubpleural\b",
    # common compound
    r"\bright-sided\b", r"\bleft-sided\b",
]
_SPATIAL_RE = re.compile("|".join(_SPATIAL_WORDS), re.IGNORECASE)


def despecify_text(text: str) -> str:
    """Remove spatial localization words from text (de-specification fallback)."""
    result = _SPATIAL_RE.sub("", text)
    # Collapse multiple spaces
    result = re.sub(r"  +", " ", result).strip()
    return result


# ── Targeted repair functions (used by Stage 5 repair executor) ──

_LATERALITY_WORDS = [
    r"\bright\b", r"\bleft\b", r"\bbilateral(ly)?\b",
    r"\bright-sided\b", r"\bleft-sided\b",
    r"\bunilateral(ly)?\b",
]
_LATERALITY_RE = re.compile("|".join(_LATERALITY_WORDS), re.IGNORECASE)

_DEPTH_WORDS = [
    r"\bupper\b", r"\blower\b", r"\bmiddle\b",
    r"\bapical\b", r"\bbasal\b",
    r"\bsuperior\b", r"\binferior\b",
]
_DEPTH_RE = re.compile("|".join(_DEPTH_WORDS), re.IGNORECASE)


def drop_laterality(text: str) -> str:
    """Remove laterality terms from text (targeted repair for R1 violations)."""
    result = _LATERALITY_RE.sub("", text)
    return re.sub(r"  +", " ", result).strip()


def drop_depth(text: str) -> str:
    """Remove depth/level terms from text (targeted repair for R3 violations)."""
    result = _DEPTH_RE.sub("", text)
    return re.sub(r"  +", " ", result).strip()


@dataclass
class GeneratorConfig:
    # Backend: "ollama" | "openai" | "anthropic" | "huggingface"
    backend: str = "ollama"
    model: str = "qwen2.5:7b"
    ollama_host: str = "http://localhost:11434"
    timeout_s: float = 60.0
    temperature: float = 0.3
    max_tokens: int = 256
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    # HuggingFace backend options
    hf_device_map: str = "auto"
    hf_torch_dtype: str = "bfloat16"
    hf_token: Optional[str] = None
    # If True, include bbox coordinates in the prompt
    include_bbox: bool = True
    # If True, include split_score and level in the prompt
    include_scores: bool = True


_SYSTEM_PROMPT = (
    "You are an expert radiologist writing a CT report. "
    "You will be given evidence tokens extracted from a CT scan — each token "
    "describes a spatial region with its anatomical location and image features. "
    "You will also receive an evidence summary that tells you what spatial "
    "attributes (laterality, depth) are supported by the evidence. "
    "Generate a single concise radiology report sentence that accurately describes "
    "the findings for the specified anatomy region. "
    "IMPORTANT CONSTRAINTS:\n"
    "- Only mention laterality (left/right/bilateral) if the evidence summary "
    "explicitly confirms it. If dominant_side is 'mixed' or 'unknown', do NOT "
    "include any laterality terms.\n"
    "- Only mention depth-related terms (upper/lower/apical/basal) if the "
    "evidence tokens consistently support that depth range.\n"
    "Write in standard radiology report style. Do not add disclaimers."
)


def _format_token_context(
    tokens: Sequence[EvidenceToken],
    anatomy_keyword: Optional[str],
    include_bbox: bool,
    include_scores: bool,
) -> str:
    """Format evidence tokens as a structured text block for the LLM prompt."""
    lines: List[str] = []
    if anatomy_keyword:
        lines.append(f"Target anatomy: {anatomy_keyword}")
    lines.append(f"Evidence tokens ({len(tokens)} total):")
    for i, tok in enumerate(tokens):
        parts = [f"  Token {i+1} (id={tok.token_id}, level={tok.level})"]
        if include_scores:
            parts.append(f"    split_score={tok.split_score:.3f}")
        if include_bbox:
            b = tok.bbox
            cx, cy, cz = b.center()
            parts.append(
                f"    center_xyz=({cx:.1f}, {cy:.1f}, {cz:.1f}), "
                f"vol={b.volume():.0f} voxels³"
            )
        lines.extend(parts)
    return "\n".join(lines)


def _format_evidence_card(card: EvidenceCard) -> str:
    """Format evidence card as a structured text block for the LLM prompt."""
    allowed = card.laterality_allowed()
    lines = [
        "Evidence summary:",
        f"  cited_tokens: {card.cited_count}",
        f"  left_count: {card.left_count}, right_count: {card.right_count}, cross_count: {card.cross_count}",
        f"  dominant_side: {card.dominant_side} (same_side_ratio: {card.same_side_ratio:.3f})",
        f"  laterality_allowed: {allowed}",
        f"  level_histogram: {dict(sorted(card.level_histogram.items()))}",
    ]
    if card.level_range_expected is not None:
        lo, hi = card.level_range_expected
        lines.append(f"  level_range_expected: [{lo}, {hi}]")
        lines.append(f"  in_range: {card.in_range_count}, out_of_range: {card.out_of_range_count}")
    return "\n".join(lines)


def _build_generation_prompt(
    plan: SentencePlan,
    tokens: Sequence[EvidenceToken],
    include_bbox: bool,
    include_scores: bool,
    history: Optional[List[str]] = None,
    evidence_card: Optional[EvidenceCard] = None,
) -> str:
    ctx = _format_token_context(tokens, plan.anatomy_keyword, include_bbox, include_scores)
    negation_note = " (negated finding expected)" if plan.is_negated else ""
    parts: List[str] = []
    if history:
        parts.append("Previously generated sentences:")
        for i, h in enumerate(history, 1):
            parts.append(f"  {i}. {h}")
        parts.append("")
    parts.append(f"Topic sentence to generate{negation_note}: \"{plan.topic}\"")
    parts.append("")
    parts.append(ctx)
    parts.append("")
    if evidence_card is not None:
        parts.append(_format_evidence_card(evidence_card))
        parts.append("")
    parts.append(
        "Write one concise radiology report sentence for this finding. "
        "Maintain consistency with previously generated sentences. "
        "Respect the laterality_allowed constraint from the evidence summary."
    )
    return "\n".join(parts)


@dataclass
class GeneratedSentence:
    sentence_index: int
    original_topic: str
    generated_text: str
    token_ids_used: List[int]
    error: Optional[str] = None


class Stage3cGenerator:
    """
    Stage 3c: LLM generates report sentence conditioned on evidence token context.

    This implements the CP .tex Token-Gated Generation concept:
    - Evidence tokens from Stage 2/3 become the "visual context"
    - LLM generates grounded report text from spatial token descriptions
    - With trained W_proj, token features are semantically aligned to text space
    """

    def __init__(self, cfg: GeneratorConfig) -> None:
        self.cfg = cfg
        self._client: Any = None
        self._hf_pipe: Any = None
        if cfg.backend == "openai":
            import openai  # type: ignore
            kw: Dict[str, Any] = {}
            if cfg.openai_api_key:
                kw["api_key"] = cfg.openai_api_key
            self._client = openai.OpenAI(**kw)
        elif cfg.backend == "anthropic":
            import anthropic  # type: ignore
            kw = {}
            if cfg.anthropic_api_key:
                kw["api_key"] = cfg.anthropic_api_key
            self._client = anthropic.Anthropic(**kw)
        elif cfg.backend == "huggingface":
            import os
            import torch  # type: ignore
            from transformers import pipeline  # type: ignore
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(cfg.hf_torch_dtype, torch.bfloat16)
            kw = dict(
                task="text-generation",
                model=cfg.model,
                torch_dtype=dtype,
                device_map=cfg.hf_device_map,
            )
            is_local = os.path.isdir(cfg.model)
            if not is_local and cfg.hf_token:
                kw["token"] = cfg.hf_token
            self._hf_pipe = pipeline(**kw)
            src = "local" if is_local else "HuggingFace Hub"
            print(f"[Stage 3c] Loaded model ({src}): {cfg.model}")

    # ------------------------------------------------------------------
    # LLM backend calls
    # ------------------------------------------------------------------

    def _call_ollama(self, user_prompt: str) -> str:
        import json as _json
        import urllib.request

        payload = _json.dumps(
            {
                "model": self.cfg.model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": self.cfg.temperature,
                    "num_predict": self.cfg.max_tokens,
                },
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.cfg.ollama_host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
            data = _json.loads(resp.read())
        return str(data.get("message", {}).get("content", ""))

    def _call_openai(self, user_prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            timeout=self.cfg.timeout_s,
        )
        return str(resp.choices[0].message.content)

    def _call_anthropic(self, user_prompt: str) -> str:
        resp = self._client.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return str(resp.content[0].text)

    def _call_huggingface(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        do_sample = self.cfg.temperature > 0.0
        outputs = self._hf_pipe(
            messages,
            max_new_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature if do_sample else None,
            do_sample=do_sample,
        )
        generated = outputs[0]["generated_text"]
        if isinstance(generated, list):
            return str(generated[-1].get("content", ""))
        return str(generated)

    def _call_llm(self, user_prompt: str) -> str:
        if self.cfg.backend == "ollama":
            return self._call_ollama(user_prompt)
        if self.cfg.backend == "openai":
            return self._call_openai(user_prompt)
        if self.cfg.backend == "anthropic":
            return self._call_anthropic(user_prompt)
        if self.cfg.backend == "huggingface":
            return self._call_huggingface(user_prompt)
        raise ValueError(f"Unknown backend: {self.cfg.backend!r}")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_sentence(
        self,
        plan: SentencePlan,
        cited_tokens: Sequence[EvidenceToken],
        history: Optional[List[str]] = None,
        evidence_card: Optional[EvidenceCard] = None,
    ) -> GeneratedSentence:
        """Generate one report sentence conditioned on cited tokens and history."""
        prompt = _build_generation_prompt(
            plan,
            cited_tokens,
            self.cfg.include_bbox,
            self.cfg.include_scores,
            history=history,
            evidence_card=evidence_card,
        )
        try:
            text = self._call_llm(prompt).strip()
            # Take only the first sentence if LLM generates multiple
            first = text.split("\n")[0].strip()
            if first:
                text = first
            return GeneratedSentence(
                sentence_index=plan.sentence_index,
                original_topic=plan.topic,
                generated_text=text,
                token_ids_used=[t.token_id for t in cited_tokens],
            )
        except Exception as e:
            return GeneratedSentence(
                sentence_index=plan.sentence_index,
                original_topic=plan.topic,
                generated_text=plan.topic,  # fallback: return original topic
                token_ids_used=[t.token_id for t in cited_tokens],
                error=str(e),
            )

    def generate_report(
        self,
        plans: Sequence[SentencePlan],
        sentence_citations: Dict[int, List[int]],
        all_tokens: Sequence[EvidenceToken],
    ) -> List[GeneratedSentence]:
        """
        Generate all sentences for a report.

        Args:
            plans: sentence plans (from ReportSentencePlanner)
            sentence_citations: {sentence_index: [token_ids]} from Stage 3 routing
            all_tokens: full token bank for the case
        """
        token_map = {t.token_id: t for t in all_tokens}
        results: List[GeneratedSentence] = []
        for plan in plans:
            cited_ids = sentence_citations.get(plan.sentence_index, [])
            cited = [token_map[tid] for tid in cited_ids if tid in token_map]
            results.append(self.generate_sentence(plan, cited))
        return results
