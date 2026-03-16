from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .evidence_card import EvidenceCard
from .types import RuleViolation, SentenceAudit, SentenceOutput


@dataclass
class JudgeVerdict:
    rule_id: str
    confirmed: bool
    adjusted_severity: float
    reasoning: str
    suggested_action: str = ""  # "drop_laterality" | "reroute_same_side" | "drop_depth" | "merge_conflict" | ""
    offending_span: str = ""


@dataclass
class SentenceJudgement:
    sentence_index: int
    verdicts: List[JudgeVerdict] = field(default_factory=list)

    def max_confirmed_severity(self) -> float:
        sevs = [v.adjusted_severity for v in self.verdicts if v.confirmed]
        return max(sevs) if sevs else 0.0

    def any_confirmed(self) -> bool:
        return any(v.confirmed for v in self.verdicts)


_SYSTEM_PROMPT = (
    "You are an expert analytical judge system for CT radiology report consistency. "
    "You will be given a sentence from a CT report, an automated rule violation, "
    "and an evidence summary showing the spatial distribution of cited tokens. "
    "Your task is to decide if the violation is genuine and suggest a repair action.\n\n"
    "You MUST output strictly valid JSON matching this structure:\n"
    '{"confirmed": true/false, "severity": 0.0, "suggested_action": "string", '
    '"offending_span": "string or null", "reasoning": "brief reason"}\n\n'
    "severity must be 0.0 if not confirmed, otherwise between 0.0 and 1.0.\n"
    "suggested_action must be empty string if not confirmed, otherwise exactly ONE of:\n"
    '  - "drop_laterality": remove laterality terms and rewrite without specifying side\n'
    '  - "reroute_same_side": keep only dominant-side tokens and re-route\n'
    '  - "drop_depth": remove depth/level terms (upper/lower/apical/basal) and rewrite\n'
    '  - "merge_conflict": cross-sentence conflict, merge or de-duplicate assertions\n\n'
    "offending_span is the exact substring from the sentence that causes the violation, "
    "or null if not applicable.\n"
    "Do not output anything other than the JSON object."
)


def _build_user_prompt(
    sentence_text: str,
    rule_id: str,
    message: str,
    evidence_card: Optional[EvidenceCard] = None,
) -> str:
    parts = [
        f'Radiology report sentence: "{sentence_text}"',
        f"Automated rule: {rule_id}",
        f"Violation details: {message}",
    ]
    if evidence_card is not None:
        import json as _json
        parts.append(f"Evidence summary: {_json.dumps(evidence_card.to_prompt_dict())}")
    parts.append("")
    parts.append("Is this a genuine violation? Reply with JSON only.")
    return "\n".join(parts)


_VALID_ACTIONS = {"drop_laterality", "reroute_same_side", "drop_depth", "merge_conflict"}


def _parse_verdict(rule_id: str, raw: str, fallback_severity: float) -> JudgeVerdict:
    """Parse LLM JSON response, with graceful fallback on parse failure."""
    try:
        m = re.search(r"\{[^}]+\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            confirmed = bool(data.get("confirmed", False))
            severity = float(data.get("severity", fallback_severity if confirmed else 0.0))
            severity = max(0.0, min(1.0, severity))
            reasoning = str(data.get("reasoning", ""))
            suggested_action = str(data.get("suggested_action", ""))
            if suggested_action not in _VALID_ACTIONS:
                suggested_action = ""
            offending_span = data.get("offending_span") or ""
            return JudgeVerdict(
                rule_id=rule_id,
                confirmed=confirmed,
                adjusted_severity=severity,
                reasoning=reasoning,
                suggested_action=suggested_action if confirmed else "",
                offending_span=str(offending_span) if confirmed else "",
            )
    except Exception:
        pass
    # Fallback: heuristic from raw text
    lowered = raw.lower()
    confirmed = '"confirmed": true' in lowered or "confirmed: true" in lowered
    return JudgeVerdict(
        rule_id=rule_id,
        confirmed=confirmed,
        adjusted_severity=fallback_severity if confirmed else 0.0,
        reasoning=raw[:300],
    )


@dataclass
class LLMJudgeConfig:
    # Backend selection: "ollama" | "openai" | "anthropic" | "huggingface"
    backend: str = "ollama"
    # Model name (backend-specific)
    model: str = "qwen2.5:7b"
    # CP .tex penalty: S'_i = S_i * (1 - alpha * sev_i)
    alpha: float = 0.5
    # Ollama host
    ollama_host: str = "http://localhost:11434"
    timeout_s: float = 30.0
    temperature: float = 0.0
    # API keys (if not set, reads from env)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    # HuggingFace backend options
    hf_device_map: str = "auto"       # "auto" | "cpu" | "cuda"
    hf_torch_dtype: str = "bfloat16"  # "bfloat16" | "float16" | "float32"
    hf_token: Optional[str] = None    # HF access token (for gated models like Llama-3)
    # When LLM call fails, fall back to confirming the original violation
    fail_open: bool = True


class LLMJudge:
    """
    Stage 5: LLM-as-Semantic-Judge.

    For each sentence with Stage-4 violations, call an LLM to confirm or
    dismiss each violation. Confirmed violations are used to apply the CP .tex
    score penalty: S'_i = S_i * (1 - alpha * sev_i).
    """

    def __init__(self, cfg: LLMJudgeConfig) -> None:
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
            kw: Dict[str, Any] = dict(
                task="text-generation",
                model=cfg.model,
                torch_dtype=dtype,
                device_map=cfg.hf_device_map,
            )
            # Only pass token for remote HuggingFace Hub models (not local paths)
            is_local = os.path.isdir(cfg.model)
            if not is_local and cfg.hf_token:
                kw["token"] = cfg.hf_token
            self._hf_pipe = pipeline(**kw)
            src = "local" if is_local else "HuggingFace Hub"
            print(f"[Stage 5] Loaded model ({src}): {cfg.model}")

    # ------------------------------------------------------------------
    # LLM backend calls
    # ------------------------------------------------------------------

    def _call_ollama(self, user_prompt: str) -> str:
        import urllib.request

        payload = json.dumps(
            {
                "model": self.cfg.model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": self.cfg.temperature},
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.cfg.ollama_host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
            data = json.loads(resp.read())
        return str(data.get("message", {}).get("content", ""))

    def _call_openai(self, user_prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.cfg.temperature,
            timeout=self.cfg.timeout_s,
        )
        return str(resp.choices[0].message.content)

    def _call_anthropic(self, user_prompt: str) -> str:
        resp = self._client.messages.create(
            model=self.cfg.model,
            max_tokens=256,
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
            max_new_tokens=256,
            temperature=self.cfg.temperature if do_sample else None,
            do_sample=do_sample,
        )
        # transformers pipeline returns full conversation; last entry is model reply
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
        raise ValueError(f"Unknown LLM backend: {self.cfg.backend!r}")

    # ------------------------------------------------------------------
    # Judging logic
    # ------------------------------------------------------------------

    def judge_violations(
        self,
        sentence_text: str,
        violations: Sequence[RuleViolation],
        evidence_card: Optional[EvidenceCard] = None,
    ) -> List[JudgeVerdict]:
        """Ask the LLM to confirm or dismiss each violation for a single sentence."""
        verdicts: List[JudgeVerdict] = []
        for v in violations:
            prompt = _build_user_prompt(sentence_text, v.rule_id, v.message, evidence_card)
            try:
                raw = self._call_llm(prompt)
                verdicts.append(_parse_verdict(v.rule_id, raw, v.severity))
            except Exception as e:
                if self.cfg.fail_open:
                    # Conservative: keep original violation
                    verdicts.append(
                        JudgeVerdict(
                            rule_id=v.rule_id,
                            confirmed=True,
                            adjusted_severity=v.severity,
                            reasoning=f"LLM call failed ({e}); defaulting to confirmed.",
                        )
                    )
                else:
                    # Optimistic: dismiss on failure
                    verdicts.append(
                        JudgeVerdict(
                            rule_id=v.rule_id,
                            confirmed=False,
                            adjusted_severity=0.0,
                            reasoning=f"LLM call failed ({e}); defaulting to dismissed.",
                        )
                    )
        return verdicts

    def reroute_scores(
        self,
        scores: Dict[int, float],
        verdicts: List[JudgeVerdict],
    ) -> Dict[int, float]:
        """
        CP .tex Stage-5 penalty: S'_i = S_i * (1 - alpha * sev_i).
        Uses the maximum confirmed severity across all violations for the sentence.
        """
        confirmed_sevs = [v.adjusted_severity for v in verdicts if v.confirmed]
        if not confirmed_sevs:
            return scores
        max_sev = max(confirmed_sevs)
        penalty = max(0.0, 1.0 - self.cfg.alpha * max_sev)
        return {tid: s * penalty for tid, s in scores.items()}

    def reroute_scores_log_smooth(
        self,
        scores: Dict[int, float],
        verdicts: List[JudgeVerdict],
        gamma: float,
        violations: Optional[Sequence["RuleViolation"]] = None,
    ) -> Dict[int, float]:
        """
        Log-smooth penalty per paper: r'_i = r_i - γ·ln(1 + sev_i).

        Per-token mode (when violations provided): each token is penalized only
        by the max severity of confirmed violations that cite it.
        Sentence-level fallback (no violations): uniform penalty from max severity.
        """
        confirmed_sevs = [v.adjusted_severity for v in verdicts if v.confirmed]
        if not confirmed_sevs:
            return scores

        # Build per-token severity map from violation token_ids
        if violations is not None:
            # Match verdicts to violations by rule_id to get token_ids
            verdict_map = {v.rule_id: v for v in verdicts if v.confirmed}
            token_sev: Dict[int, float] = {}
            for viol in violations:
                vd = verdict_map.get(viol.rule_id)
                if vd is None:
                    continue
                for tid in viol.token_ids:
                    token_sev[tid] = max(token_sev.get(tid, 0.0), vd.adjusted_severity)

            if token_sev:
                result = {}
                for tid, s in scores.items():
                    sev = token_sev.get(tid, 0.0)
                    result[tid] = s - gamma * math.log(1.0 + sev)
                return result

        # Fallback: sentence-level uniform penalty
        max_sev = max(confirmed_sevs)
        penalty = gamma * math.log(1.0 + max_sev)
        return {tid: s - penalty for tid, s in scores.items()}

    def judge_all(
        self,
        sentence_outputs: Sequence[SentenceOutput],
        audits: Sequence[SentenceAudit],
        evidence_cards: Optional[Dict[int, EvidenceCard]] = None,
    ) -> Dict[int, SentenceJudgement]:
        """
        Run judge over all sentences with violations.
        Returns mapping sentence_index -> SentenceJudgement.
        """
        s_map = {s.sentence_index: s for s in sentence_outputs}
        results: Dict[int, SentenceJudgement] = {}
        for audit in audits:
            judgement = SentenceJudgement(sentence_index=audit.sentence_index)
            if audit.violations:
                sentence = s_map.get(audit.sentence_index)
                text = sentence.text if sentence else ""
                card = evidence_cards.get(audit.sentence_index) if evidence_cards else None
                judgement.verdicts = self.judge_violations(text, audit.violations, evidence_card=card)
            results[audit.sentence_index] = judgement
        return results
