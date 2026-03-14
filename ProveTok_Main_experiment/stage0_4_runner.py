from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import ProveTokConfig
from .simple_modules import ReportSentencePlanner, RuleBasedAnatomyResolver
from .stage0_scorer import DeterministicArtifactScorer
from .stage1_swinunetr_encoder import FrozenSwinUNETREncoder
from .stage2_octree_splitter import AdaptiveOctreeSplitter
from .stage3_router import Router
from .stage3c_generator import Stage3cGenerator, despecify_text
from .stage4_verifier import Verifier
from .stage5_llm_judge import LLMJudge
from .token_bank_io import save_token_bank_case
from .types import BBox3D, EvidenceToken, SentenceOutput


@dataclass
class Stage04Components:
    artifact_scorer: DeterministicArtifactScorer
    encoder: FrozenSwinUNETREncoder
    splitter: AdaptiveOctreeSplitter
    planner: ReportSentencePlanner
    anatomy_resolver: RuleBasedAnatomyResolver
    router: Router
    verifier: Verifier
    llm_judge: Optional[LLMJudge] = None
    generator: Optional[Stage3cGenerator] = None


def run_case_stage0_4(
    case_id: str,
    report_text: str,
    volume,
    spacing_xyz_mm: Tuple[float, float, float],
    out_case_dir: str,
    cfg: ProveTokConfig,
    comp: Stage04Components,
) -> Dict[str, object]:
    out_dir = Path(out_case_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_state = comp.artifact_scorer.score(volume, case_id=case_id)
    encoded = comp.encoder.encode(volume, case_id=case_id)
    tokens = comp.splitter.build_tokens(
        volume=volume,
        encoded=encoded,
        artifact_state=artifact_state,
        token_budget_b=cfg.split.token_budget_b,
    )

    d, h, w = volume.shape
    global_bbox_voxel = BBox3D(x_min=0.0, x_max=float(w), y_min=0.0, y_max=float(h), z_min=0.0, z_max=float(d))
    save_token_bank_case(
        out_case_dir=str(out_dir),
        tokens=tokens,
        cfg=cfg,
        spacing_xyz_mm=spacing_xyz_mm,
        encoder_name=comp.encoder.model.__class__.__name__,
        global_bbox_voxel=global_bbox_voxel,
    )

    comp.planner.set_report(report_text)
    comp.anatomy_resolver.volume_shape = volume.shape
    comp.verifier.volume_shape = tuple(int(x) for x in volume.shape)
    plans = comp.planner.plan(tokens)

    token_map = {t.token_id: t for t in tokens}

    sentence_outputs: List[SentenceOutput] = []
    sentence_logs: List[Dict[str, object]] = []
    generated_history: List[str] = []  # accumulate for sentence history context
    for plan in plans:
        q_s = comp.router.text_encoder(plan.topic)
        anatomy_bbox = comp.anatomy_resolver(plan.anatomy_keyword)
        scores = comp.router.score_tokens(plan.topic, tokens, anatomy_bbox)
        topk_ids = sorted(scores.keys(), key=lambda tid: (-scores[tid], tid))[: cfg.router.k_per_sentence]
        topk_scores = [float(scores[tid]) for tid in topk_ids]

        # Stage 3c: LLM generation conditioned on routed tokens
        gen_text = plan.topic
        gen_flag = False
        gen_error: Optional[str] = None
        if comp.generator is not None:
            cited_tokens = [token_map[tid] for tid in topk_ids if tid in token_map]
            gen_result = comp.generator.generate_sentence(
                plan, cited_tokens, history=generated_history or None
            )
            gen_text = gen_result.generated_text
            gen_flag = gen_result.error is None
            gen_error = gen_result.error

        sentence_outputs.append(
            SentenceOutput(
                sentence_index=plan.sentence_index,
                text=gen_text,
                citations=topk_ids,
                route_scores=scores,
                original_topic=plan.topic,
                generated=gen_flag,
                generation_error=gen_error,
            )
        )
        # Accumulate history for subsequent sentences
        generated_history.append(gen_text)

        sentence_logs.append(
            {
                "sentence_index": int(plan.sentence_index),
                "sentence_text": gen_text,
                "original_topic": plan.topic,
                "generated": gen_flag,
                "generation_error": gen_error,
                "anatomy_keyword": plan.anatomy_keyword,
                "q_s": [float(x) for x in q_s],
                "topk_token_ids": [int(x) for x in topk_ids],
                "topk_scores": topk_scores,
            }
        )

    audits = comp.verifier.audit_all(sentence_outputs, plans, tokens)

    # Cross-sentence consistency check (R6)
    cross_violations = comp.verifier.cross_sentence_check(sentence_outputs, plans)
    if cross_violations:
        # Attach cross-sentence violations to the relevant audits
        audit_map = {a.sentence_index: a for a in audits}
        for cv in cross_violations:
            a = audit_map.get(cv.sentence_index)
            if a is not None:
                a.violations.append(cv)
                a.passed = False

    # Stage 5: LLM judge — confirm/dismiss violations, log-smooth penalty, re-top-k, regenerate
    stage5_judgements: Dict[int, object] = {}
    gamma = cfg.reroute.gamma_penalty
    max_retry = cfg.reroute.max_retry

    if comp.llm_judge is not None:
        judgements = comp.llm_judge.judge_all(sentence_outputs, audits)
        for idx, s_out in enumerate(sentence_outputs):
            j = judgements.get(s_out.sentence_index)
            if j is None or not j.any_confirmed():
                continue

            # Per-token log-smooth penalty: r'_i = r_i - γ·ln(1 + sev_i)
            audit = next((a for a in audits if a.sentence_index == s_out.sentence_index), None)
            penalized = comp.llm_judge.reroute_scores_log_smooth(
                s_out.route_scores, j.verdicts, gamma,
                violations=audit.violations if audit else None,
            )
            # Re-top-k: select new top-k from penalized scores
            new_topk_ids = sorted(
                penalized.keys(), key=lambda tid: (-penalized[tid], tid)
            )[: cfg.router.k_per_sentence]

            s_out.route_scores = penalized
            s_out.citations = new_topk_ids
            s_out.rerouted = True
            s_out.stop_reason = "log_smooth_reroute"

            # Regenerate with new citations (if generator available and retry > 0)
            if comp.generator is not None and max_retry > 0:
                plan = plans[idx]
                cited_tokens = [token_map[tid] for tid in new_topk_ids if tid in token_map]
                # Build history from all other sentences (exclude current)
                regen_history = [
                    so.text for so in sentence_outputs if so.sentence_index != s_out.sentence_index
                ] or None
                gen_result = comp.generator.generate_sentence(plan, cited_tokens, history=regen_history)
                s_out.text = gen_result.generated_text
                s_out.generated = gen_result.error is None
                s_out.generation_error = gen_result.error

                # Re-verify after reroute; if still failing → de-specify fallback
                re_audit = comp.verifier.audit_all([s_out], [plan], tokens)
                if re_audit and re_audit[0].violations:
                    despec_topic = despecify_text(plan.topic)
                    if despec_topic != plan.topic:
                        despec_plan = type(plan)(
                            sentence_index=plan.sentence_index,
                            topic=despec_topic,
                            anatomy_keyword=plan.anatomy_keyword,
                            expected_level_range=plan.expected_level_range,
                            expected_volume_range=plan.expected_volume_range,
                            is_negated=plan.is_negated,
                        )
                        gen_result = comp.generator.generate_sentence(
                            despec_plan, cited_tokens, history=regen_history,
                        )
                        s_out.text = gen_result.generated_text
                        s_out.generated = gen_result.error is None
                        s_out.generation_error = gen_result.error
                        s_out.stop_reason = "de_specified"

            stage5_judgements[s_out.sentence_index] = [
                {
                    "rule_id": v.rule_id,
                    "confirmed": v.confirmed,
                    "adjusted_severity": v.adjusted_severity,
                    "reasoning": v.reasoning,
                }
                for v in j.verdicts
            ]

    violations_by_sentence = {a.sentence_index: [asdict(v) for v in a.violations] for a in audits}
    s_out_map = {s.sentence_index: s for s in sentence_outputs}
    for row in sentence_logs:
        si = row["sentence_index"]
        row["violations"] = violations_by_sentence.get(si, [])
        row["stage5_judgements"] = stage5_judgements.get(si, [])
        s = s_out_map.get(si)
        if s is not None and s.rerouted:
            row["rerouted_citations"] = [int(x) for x in s.citations]
            row["sentence_text"] = s.text  # update with regenerated text
            row["stop_reason"] = s.stop_reason

    b_plan = cfg.router.planning_budget(cfg.split.token_budget_b)
    trace_jsonl = out_dir / "trace.jsonl"
    with trace_jsonl.open("w", encoding="utf-8") as f:
        case_meta = {
            "type": "case_meta",
            "case_id": case_id,
            "B": int(cfg.split.token_budget_b),
            "k": int(cfg.router.k_per_sentence),
            "B_plan": int(b_plan),
            "lambda_spatial": float(cfg.router.lambda_spatial),
            "tau_IoU": float(cfg.verifier.tau_anatomy_iou),
            "ell_coarse": int(cfg.router.planning_level_cutoff),
            "beta": float(cfg.split.beta),
            "n_sentences": len(sentence_logs),
        }
        f.write(json.dumps(case_meta, ensure_ascii=False) + "\n")
        for s in sentence_logs:
            payload = {"type": "sentence", **s}
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    n_violate = sum(len(a.violations) for a in audits)
    n_judge_confirmed = sum(
        1
        for v_list in stage5_judgements.values()
        if isinstance(v_list, list) and any(v.get("confirmed") for v in v_list)  # type: ignore[union-attr]
    )
    n_generated = sum(1 for s in sentence_outputs if s.generated)
    n_rerouted = sum(1 for s in sentence_outputs if s.rerouted)
    n_despecified = sum(1 for s in sentence_outputs if s.stop_reason == "de_specified")
    return {
        "case_id": case_id,
        "n_tokens": len(tokens),
        "n_sentences": len(sentence_logs),
        "n_violations": int(n_violate),
        "n_judge_confirmed": int(n_judge_confirmed),
        "n_generated": int(n_generated),
        "n_rerouted": int(n_rerouted),
        "n_despecified": int(n_despecified),
        "n_cross_violations": len(cross_violations),
        "trace_jsonl": str(trace_jsonl),
    }
