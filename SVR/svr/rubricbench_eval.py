from __future__ import annotations

import asyncio
import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Sequence

import torch

from svr.bank import rubric_similarity
from svr.data import (
    DEFAULT_REFERENCE_RUBRIC_FIELD_CANDIDATES,
    load_preference_examples,
)
from svr.inference import NoOpRubricRewriter, SVRInferenceEngine
from svr.schema import (
    PairwisePrediction,
    PairwiseRubricComparison,
    PreferenceExample,
    RubricItem,
)
from svr.utils import (
    dump_json,
    ensure_dir,
    gold_aligned_margin,
    importance_weight,
    orient_preference_pair,
    preferred_position,
    public_path,
    semantic_prediction_payload,
)

def _normalize_domain(value: Any) -> str:
    if not isinstance(value, str):
        return "unknown"
    normalized = " ".join(value.strip().split()).lower()
    return normalized or "unknown"


def _example_domain(example: PreferenceExample) -> str:
    return _normalize_domain(example.meta.get("domain"))


def _build_identity_payload(example: PreferenceExample) -> dict[str, Any]:
    raw_record = example.raw_record if isinstance(example.raw_record, dict) else {}
    return {
        "example_id": example.example_id,
        "record_idx": example.meta.get("record_idx"),
        "domain": _example_domain(example),
        "raw_domain": example.meta.get("domain"),
        "case_id": raw_record.get("case_id"),
        "source_id": raw_record.get("id"),
        "chosen_candidate": raw_record.get("chosen_candidate"),
        "preferred_candidate": raw_record.get("preferred_candidate"),
    }


def _prediction_payload(
    *,
    example: PreferenceExample,
    prediction,
    correct: bool,
    gold_side: str,
) -> dict[str, Any]:
    payload = _build_identity_payload(example)
    payload.update(
        {
            "skipped": False,
            "correct": correct,
            "preferred_position": preferred_position(
                prediction.preferred_side,
                gold_side=gold_side,
            ),
            "margin_vs_chosen": gold_aligned_margin(
                prediction.weighted_margin,
                gold_side,
            ),
            "prediction": semantic_prediction_payload(
                prediction,
                gold_side=gold_side,
            ),
        }
    )
    return payload


def _oriented_example(
    example: PreferenceExample,
    *,
    salt: str,
) -> tuple[PreferenceExample, str]:
    oriented = orient_preference_pair(
        example_id=example.example_id,
        positive_response=example.chosen_response,
        negative_response=example.rejected_response,
        salt=salt,
    )
    return (
        PreferenceExample(
            example_id=example.example_id,
            prompt_text=example.prompt_text,
            prompt_messages=example.prompt_messages,
            chosen_response=oriented.response_a,
            rejected_response=oriented.response_b,
            candidate_responses=list(example.candidate_responses),
            self_rubrics=list(example.self_rubrics),
            reference_rubrics=list(example.reference_rubrics),
            difficulty_analysis=example.difficulty_analysis,
            meta=dict(example.meta),
            raw_record=example.raw_record,
        ),
        oriented.gold_side,
    )


def _prediction_has_full_rubric_coverage(prediction: Any, expected_count: int) -> bool:
    selected_rubrics = getattr(prediction, "selected_rubrics", None)
    rubric_comparisons = getattr(prediction, "rubric_comparisons", None)
    if not isinstance(selected_rubrics, list) or not isinstance(rubric_comparisons, list):
        return False
    if len(selected_rubrics) != expected_count:
        return False
    if len(rubric_comparisons) != expected_count:
        return False
    return True


def _scorer_tie_margin(engine: SVRInferenceEngine) -> float:
    scorer_config = getattr(getattr(engine, "scorer", None), "config", None)
    tie_margin = getattr(scorer_config, "tie_margin", 0.05)
    try:
        return float(tie_margin)
    except (TypeError, ValueError):
        return 0.05


def _selection_text(item: dict[str, Any]) -> str:
    return str(item.get("bank_text") or item.get("text") or "")


def _is_still_malformed_rubric_text(text: Any) -> bool:
    if not isinstance(text, str):
        return True
    normalized = " ".join(text.strip().split())
    if not normalized:
        return True
    lowered = normalized.lower().rstrip(".?")
    if len(normalized) < 12:
        return True
    if normalized in {"The", "All", "Does the?"}:
        return True
    if lowered in {
        "the response",
        "the answer",
        "the reply",
        "the code",
        "does the response",
        "does the answer",
        "does the reply",
        "does the code",
        "is the response",
        "is the answer",
        "is the reply",
        "is the code",
    }:
        return True
    if normalized.endswith("("):
        return True
    return False


def _sanitize_selected_rubrics(selected_rubrics: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    seen_bank_ids: set[int] = set()
    for item in selected_rubrics:
        bank_id = int(item.get("bank_id", -1))
        if bank_id in seen_bank_ids:
            continue
        text = str(item.get("text") or "").strip()
        bank_text = str(item.get("bank_text") or "").strip()
        if _is_still_malformed_rubric_text(text):
            if _is_still_malformed_rubric_text(bank_text):
                continue
            text = bank_text
        normalized = dict(item)
        normalized["text"] = text
        if bank_text:
            normalized["bank_text"] = bank_text
        cleaned.append(normalized)
        seen_bank_ids.add(bank_id)
    return cleaned


def _raw_select_rubric_pool(
    *,
    engine: SVRInferenceEngine,
    prompt_text: str,
    pool_size: int,
) -> list[dict[str, Any]]:
    matrix = engine.vectorizer.transform([prompt_text])
    tensor = torch.tensor(matrix.toarray(), dtype=torch.float32, device=engine.device)
    with torch.no_grad():
        bank_scores = engine.model.score_bank(tensor)[0]
    k = min(max(1, int(pool_size)), int(bank_scores.numel()))
    values, indices = torch.topk(bank_scores, k=k)

    selected = []
    for score, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
        entry = engine.bank[int(index)]
        selected.append(
            {
                "bank_id": entry.bank_id,
                "text": entry.text,
                "bank_text": entry.text,
                "facet": entry.facet,
                "importance": entry.importance,
                "source": entry.source,
                "grounding": entry.grounding,
                "selection_weight": float(score),
            }
        )
    return selected


def _diverse_rubric_subset(
    *,
    candidates: Sequence[dict[str, Any]],
    target_k: int,
    max_same_facet: int,
    diversity_similarity_threshold: float,
    diversity_lambda: float,
) -> list[dict[str, Any]]:
    if target_k <= 0 or not candidates:
        return []

    remaining = [dict(item) for item in candidates]
    selected: list[dict[str, Any]] = []
    facet_counts: Counter[str] = Counter()

    while remaining and len(selected) < target_k:
        eligible = [
            item
            for item in remaining
            if facet_counts[str(item.get("facet") or "correctness")] < max_same_facet
        ]
        pool = eligible if eligible else remaining

        best_item = None
        best_score = float("-inf")
        for item in pool:
            max_sim = 0.0
            if selected:
                max_sim = max(
                    rubric_similarity(_selection_text(item), _selection_text(chosen))
                    for chosen in selected
                )
            adjusted = float(item.get("selection_weight", 0.0)) - diversity_lambda * max_sim
            if max_sim >= diversity_similarity_threshold:
                adjusted -= 1.0
            if adjusted > best_score:
                best_score = adjusted
                best_item = item

        if best_item is None:
            break
        selected.append(best_item)
        facet_counts[str(best_item.get("facet") or "correctness")] += 1
        remaining = [
            item for item in remaining if int(item.get("bank_id", -1)) != int(best_item["bank_id"])
        ]

    return selected[:target_k]


def _select_diverse_rubrics(
    *,
    engine: SVRInferenceEngine,
    prompt_text: str,
    top_k: int,
    selection_pool_size: int,
    max_same_facet: int,
    diversity_similarity_threshold: float,
    diversity_lambda: float,
) -> list[dict[str, Any]]:
    pool = _raw_select_rubric_pool(
        engine=engine,
        prompt_text=prompt_text,
        pool_size=max(top_k, selection_pool_size),
    )
    chosen = _diverse_rubric_subset(
        candidates=pool,
        target_k=top_k,
        max_same_facet=max_same_facet,
        diversity_similarity_threshold=diversity_similarity_threshold,
        diversity_lambda=diversity_lambda,
    )
    rewritten = engine.rewriter.rewrite(
        prompt_text=prompt_text,
        selected_rubrics=chosen,
    )
    cleaned = _sanitize_selected_rubrics(rewritten)
    return cleaned or list(rewritten)


async def _aselect_diverse_rubrics(
    *,
    engine: SVRInferenceEngine,
    prompt_text: str,
    top_k: int,
    selection_pool_size: int,
    max_same_facet: int,
    diversity_similarity_threshold: float,
    diversity_lambda: float,
) -> list[dict[str, Any]]:
    pool = _raw_select_rubric_pool(
        engine=engine,
        prompt_text=prompt_text,
        pool_size=max(top_k, selection_pool_size),
    )
    chosen = _diverse_rubric_subset(
        candidates=pool,
        target_k=top_k,
        max_same_facet=max_same_facet,
        diversity_similarity_threshold=diversity_similarity_threshold,
        diversity_lambda=diversity_lambda,
    )
    if hasattr(engine.rewriter, "arewrite"):
        rewritten = await engine.rewriter.arewrite(
            prompt_text=prompt_text,
            selected_rubrics=chosen,
        )
    else:
        rewritten = engine.rewriter.rewrite(
            prompt_text=prompt_text,
            selected_rubrics=chosen,
        )
    cleaned = _sanitize_selected_rubrics(rewritten)
    return cleaned or list(rewritten)


def _effective_rubric_count(
    prediction: PairwisePrediction,
    *,
    min_weight: float = 1e-9,
) -> int:
    return sum(
        int(float(item.get("selection_weight", 0.0)) > min_weight)
        for item in prediction.selected_rubrics
        if isinstance(item, dict)
    )


def _rubric_vote_margin(prediction: PairwisePrediction) -> int:
    return _filtered_rubric_vote_margin(prediction)


def _filtered_rubric_vote_margin(
    prediction: PairwisePrediction,
    *,
    positive_weight_only: bool = False,
    per_facet_cap: int | None = None,
    min_weight: float = 1e-9,
) -> int:
    votes = 0
    facet_counts: Counter[str] = Counter()
    for comparison in prediction.rubric_comparisons:
        weight = float(getattr(comparison, "weight", 0.0))
        if positive_weight_only and weight <= min_weight:
            continue
        facet = str(getattr(comparison, "facet", "") or "correctness")
        if per_facet_cap is not None and per_facet_cap >= 0:
            if facet_counts[facet] >= per_facet_cap:
                continue
            facet_counts[facet] += 1
        better = str(getattr(comparison, "better", "")).strip().upper()
        if better == "A":
            votes += 1
        elif better == "B":
            votes -= 1
        else:
            delta = float(getattr(comparison, "delta", 0.0))
            if delta > 0:
                votes += 1
            elif delta < 0:
                votes -= 1
    return votes


def _margin_to_side(margin: float, *, tie_margin: float = 0.0) -> str:
    if margin > tie_margin:
        return "A"
    if margin < -tie_margin:
        return "B"
    return "tie"


def _clone_prediction_with_new_decision(
    *,
    prediction: PairwisePrediction,
    preferred_side: str,
    weighted_margin: float,
    decision_source: str,
) -> PairwisePrediction:
    return PairwisePrediction(
        preferred_side=preferred_side,
        weighted_margin=weighted_margin,
        selected_rubrics=[dict(item) for item in prediction.selected_rubrics],
        rubric_comparisons=list(prediction.rubric_comparisons),
        decision_source=decision_source,
    )


def _direct_compare_prediction(
    *,
    example: PreferenceExample,
    engine: SVRInferenceEngine,
    base_prediction: PairwisePrediction | None = None,
    decision_source: str = "direct_overall_judge",
) -> PairwisePrediction | None:
    direct_compare = getattr(engine.scorer, "_direct_compare_no_tie", None)
    if not callable(direct_compare):
        return None
    try:
        preferred_side, raw_source = direct_compare(
            prompt_messages=example.prompt_messages,
            response_a=example.chosen_response,
            response_b=example.rejected_response,
            difficulty_analysis=getattr(example, "difficulty_analysis", None),
        )
    except Exception:
        return None
    if preferred_side not in {"A", "B"}:
        return None
    template = base_prediction or PairwisePrediction(
        preferred_side=preferred_side,
        weighted_margin=0.0,
        selected_rubrics=[],
        rubric_comparisons=[],
        decision_source=raw_source,
    )
    return _clone_prediction_with_new_decision(
        prediction=template,
        preferred_side=preferred_side,
        weighted_margin=0.0,
        decision_source=decision_source or raw_source,
    )


async def _adirect_compare_prediction(
    *,
    example: PreferenceExample,
    engine: SVRInferenceEngine,
    base_prediction: PairwisePrediction | None = None,
    decision_source: str = "direct_overall_judge",
) -> PairwisePrediction | None:
    direct_compare = getattr(engine.scorer, "_adirect_compare_no_tie", None)
    if not callable(direct_compare):
        return _direct_compare_prediction(
            example=example,
            engine=engine,
            base_prediction=base_prediction,
            decision_source=decision_source,
        )
    try:
        preferred_side, raw_source = await direct_compare(
            prompt_messages=example.prompt_messages,
            response_a=example.chosen_response,
            response_b=example.rejected_response,
            difficulty_analysis=getattr(example, "difficulty_analysis", None),
        )
    except Exception:
        return None
    if preferred_side not in {"A", "B"}:
        return None
    template = base_prediction or PairwisePrediction(
        preferred_side=preferred_side,
        weighted_margin=0.0,
        selected_rubrics=[],
        rubric_comparisons=[],
        decision_source=raw_source,
    )
    return _clone_prediction_with_new_decision(
        prediction=template,
        preferred_side=preferred_side,
        weighted_margin=0.0,
        decision_source=decision_source or raw_source,
    )

def _apply_low_evidence_direct_judge(
    *,
    example: PreferenceExample,
    engine: SVRInferenceEngine,
    prediction: PairwisePrediction,
    max_effective_rubrics: int,
) -> PairwisePrediction:
    if max_effective_rubrics < 0:
        return prediction
    if _effective_rubric_count(prediction) > max_effective_rubrics:
        return prediction
    if not prediction.rubric_comparisons:
        direct_prediction = _direct_compare_prediction(
            example=example,
            engine=engine,
            base_prediction=prediction,
            decision_source="low_evidence_direct_compare",
        )
        return direct_prediction or prediction

    vote_margin = _rubric_vote_margin(prediction)
    if vote_margin > 0:
        return _clone_prediction_with_new_decision(
            prediction=prediction,
            preferred_side="A",
            weighted_margin=float(vote_margin),
            decision_source="low_evidence_vote",
        )
    if vote_margin < 0:
        return _clone_prediction_with_new_decision(
            prediction=prediction,
            preferred_side="B",
            weighted_margin=float(vote_margin),
            decision_source="low_evidence_vote",
        )
    if vote_margin == 0:
        direct_prediction = _direct_compare_prediction(
            example=example,
            engine=engine,
            base_prediction=prediction,
            decision_source="low_evidence_direct_compare",
        )
        if direct_prediction is not None:
            return direct_prediction
        return _clone_prediction_with_new_decision(
            prediction=prediction,
            preferred_side=prediction.preferred_side,
            weighted_margin=float(prediction.weighted_margin),
            decision_source="low_evidence_keep",
        )
    return prediction


async def _aapply_low_evidence_direct_judge(
    *,
    example: PreferenceExample,
    engine: SVRInferenceEngine,
    prediction: PairwisePrediction,
    max_effective_rubrics: int,
) -> PairwisePrediction:
    if max_effective_rubrics < 0:
        return prediction
    if _effective_rubric_count(prediction) > max_effective_rubrics:
        return prediction
    if not prediction.rubric_comparisons:
        direct_prediction = await _adirect_compare_prediction(
            example=example,
            engine=engine,
            base_prediction=prediction,
            decision_source="low_evidence_direct_compare",
        )
        return direct_prediction or prediction

    vote_margin = _rubric_vote_margin(prediction)
    if vote_margin > 0:
        return _clone_prediction_with_new_decision(
            prediction=prediction,
            preferred_side="A",
            weighted_margin=float(vote_margin),
            decision_source="low_evidence_vote",
        )
    if vote_margin < 0:
        return _clone_prediction_with_new_decision(
            prediction=prediction,
            preferred_side="B",
            weighted_margin=float(vote_margin),
            decision_source="low_evidence_vote",
        )
    if vote_margin == 0:
        direct_prediction = await _adirect_compare_prediction(
            example=example,
            engine=engine,
            base_prediction=prediction,
            decision_source="low_evidence_direct_compare",
        )
        if direct_prediction is not None:
            return direct_prediction
        return _clone_prediction_with_new_decision(
            prediction=prediction,
            preferred_side=prediction.preferred_side,
            weighted_margin=float(prediction.weighted_margin),
            decision_source="low_evidence_keep",
        )
    return prediction


def _apply_low_evidence_redundancy_guard(
    *,
    prediction: PairwisePrediction,
    per_facet_cap: int,
) -> PairwisePrediction:
    if prediction.decision_source not in {
        "low_evidence_vote",
        "low_evidence_direct_compare",
    }:
        return prediction
    if per_facet_cap <= 0 or not prediction.rubric_comparisons:
        return prediction

    capped_vote_margin = _filtered_rubric_vote_margin(
        prediction,
        per_facet_cap=per_facet_cap,
    )
    positive_weight_vote_margin = _filtered_rubric_vote_margin(
        prediction,
        positive_weight_only=True,
    )
    capped_side = _margin_to_side(float(capped_vote_margin))
    positive_weight_side = _margin_to_side(float(positive_weight_vote_margin))
    if capped_side not in {"A", "B"}:
        return prediction
    if positive_weight_side not in {"A", "B"}:
        return prediction
    if capped_side != positive_weight_side:
        return prediction
    if capped_side == prediction.preferred_side:
        return prediction

    return _clone_prediction_with_new_decision(
        prediction=prediction,
        preferred_side=capped_side,
        weighted_margin=float(capped_vote_margin),
        decision_source="low_evidence_facet_capped_vote",
    )


def _coverage_retry_note(*, required_count: int, top_k: int, attempt_idx: int) -> str:
    return (
        "[RubricBench eval coverage retry] "
        f"attempt={attempt_idx} current_top_k={top_k}. "
        f"You must return rubric-level judgments for all {required_count} provided "
        "rubrics exactly once. Do not omit any rubric."
    )


def _single_rubric_retry_note(
    *,
    rubric_idx: int,
    total_rubrics: int,
    top_k: int,
    attempt_idx: int,
) -> str:
    return (
        "[RubricBench eval single-rubric recovery] "
        f"rubric={rubric_idx}/{total_rubrics} current_top_k={top_k} attempt={attempt_idx}. "
        "You are judging exactly one rubric and must return exactly one rubric comparison."
    )


def _merge_recovered_predictions(
    *,
    predictions: Sequence[PairwisePrediction],
    tie_margin: float,
    decision_source: str = "rubric_margin_recovered",
) -> PairwisePrediction:
    weighted_margin = 0.0
    selected_rubrics: list[dict[str, Any]] = []
    rubric_comparisons: list[PairwiseRubricComparison] = []
    for prediction in predictions:
        weighted_margin += float(prediction.weighted_margin)
        selected_rubrics.extend(prediction.selected_rubrics)
        rubric_comparisons.extend(prediction.rubric_comparisons)

    if weighted_margin > tie_margin:
        preferred_side = "A"
    elif weighted_margin < -tie_margin:
        preferred_side = "B"
    else:
        preferred_side = "tie"

    return PairwisePrediction(
        preferred_side=preferred_side,
        weighted_margin=weighted_margin,
        selected_rubrics=selected_rubrics,
        rubric_comparisons=rubric_comparisons,
        decision_source=decision_source,
    )


def _recover_rubrics_individually(
    *,
    example: PreferenceExample,
    engine: SVRInferenceEngine,
    selected_rubrics: Sequence[dict[str, Any]],
    score_retry_times: int,
    top_k: int,
    tie_margin: float,
) -> PairwisePrediction:
    recovered: list[PairwisePrediction] = []
    failed_rubric_indices: list[int] = []
    total_rubrics = len(selected_rubrics)
    for rubric_idx, rubric in enumerate(selected_rubrics, start=1):
        prediction = None
        for attempt_idx in range(1, max(1, score_retry_times) + 1):
            prediction = engine.scorer.compare_responses(
                prompt_messages=example.prompt_messages,
                prompt_text=example.prompt_text,
                response_a=example.chosen_response,
                response_b=example.rejected_response,
                selected_rubrics=[rubric],
                rubric_note=_single_rubric_retry_note(
                    rubric_idx=rubric_idx,
                    total_rubrics=total_rubrics,
                    top_k=top_k,
                    attempt_idx=attempt_idx,
                ),
            )
            if _prediction_has_full_rubric_coverage(prediction, 1):
                break
        else:
            failed_rubric_indices.append(rubric_idx)
            continue
        recovered.append(prediction)
    if not recovered:
        direct_prediction = _direct_compare_prediction(
            example=example,
            engine=engine,
            decision_source="direct_overall_judge",
        )
        if direct_prediction is not None:
            return direct_prediction
        raise RuntimeError(
            "RubricBench eval failed single-rubric recovery with no direct judge result: "
            f"example_id={example.example_id} top_k={top_k} retries={score_retry_times}"
        )
    decision_source = (
        "rubric_margin_recovered_partial"
        if failed_rubric_indices
        else "rubric_margin_recovered"
    )
    return _merge_recovered_predictions(
        predictions=recovered,
        tie_margin=tie_margin,
        decision_source=decision_source,
    )


async def _arecover_rubrics_individually(
    *,
    example: PreferenceExample,
    engine: SVRInferenceEngine,
    selected_rubrics: Sequence[dict[str, Any]],
    score_retry_times: int,
    top_k: int,
    tie_margin: float,
) -> PairwisePrediction:
    recovered: list[PairwisePrediction] = []
    failed_rubric_indices: list[int] = []
    total_rubrics = len(selected_rubrics)
    for rubric_idx, rubric in enumerate(selected_rubrics, start=1):
        prediction = None
        for attempt_idx in range(1, max(1, score_retry_times) + 1):
            prediction = await engine.scorer.acompare_responses(
                prompt_messages=example.prompt_messages,
                prompt_text=example.prompt_text,
                response_a=example.chosen_response,
                response_b=example.rejected_response,
                selected_rubrics=[rubric],
                rubric_note=_single_rubric_retry_note(
                    rubric_idx=rubric_idx,
                    total_rubrics=total_rubrics,
                    top_k=top_k,
                    attempt_idx=attempt_idx,
                ),
            )
            if _prediction_has_full_rubric_coverage(prediction, 1):
                break
        else:
            failed_rubric_indices.append(rubric_idx)
            continue
        recovered.append(prediction)
    if not recovered:
        direct_prediction = await _adirect_compare_prediction(
            example=example,
            engine=engine,
            decision_source="direct_overall_judge",
        )
        if direct_prediction is not None:
            return direct_prediction
        raise RuntimeError(
            "RubricBench eval failed single-rubric recovery with no direct judge result: "
            f"example_id={example.example_id} top_k={top_k} retries={score_retry_times}"
        )
    decision_source = (
        "rubric_margin_recovered_partial"
        if failed_rubric_indices
        else "rubric_margin_recovered"
    )
    return _merge_recovered_predictions(
        predictions=recovered,
        tie_margin=tie_margin,
        decision_source=decision_source,
    )


def _score_pair_strict_rubricbench(
    *,
    example: PreferenceExample,
    engine: SVRInferenceEngine,
    initial_top_k: int,
    required_rubrics: int,
    score_retry_times: int,
    max_top_k: int | None,
    selection_pool_size: int,
    max_same_facet: int,
    diversity_similarity_threshold: float,
    diversity_lambda: float,
    low_evidence_max_effective_rubrics: int,
):
    bank_size = len(engine.bank)
    if bank_size <= 0:
        raise ValueError("SVR bank is empty; cannot evaluate RubricBench.")
    if bank_size < required_rubrics:
        raise ValueError(
            f"SVR bank size {bank_size} is smaller than required_rubrics={required_rubrics}."
        )

    start_top_k = max(initial_top_k, required_rubrics)
    final_max_top_k = bank_size if max_top_k is None else min(max_top_k, bank_size)
    if final_max_top_k < required_rubrics:
        raise ValueError(
            f"max_top_k={final_max_top_k} is smaller than required_rubrics={required_rubrics}."
        )

    tie_margin = _scorer_tie_margin(engine)
    top_k = min(start_top_k, final_max_top_k)

    while True:
        selected_rubrics = _select_diverse_rubrics(
            engine=engine,
            prompt_text=example.prompt_text,
            top_k=top_k,
            selection_pool_size=selection_pool_size,
            max_same_facet=max_same_facet,
            diversity_similarity_threshold=diversity_similarity_threshold,
            diversity_lambda=diversity_lambda,
        )
        expected_count = len(selected_rubrics)
        if expected_count < required_rubrics:
            if top_k < final_max_top_k:
                top_k += 1
                continue
            raise RuntimeError(
                "RubricBench selector returned too few rubrics: "
                f"example_id={example.example_id} selected={expected_count} "
                f"required={required_rubrics} max_top_k={final_max_top_k}"
            )

        prediction = engine.scorer.compare_responses(
            prompt_messages=example.prompt_messages,
            prompt_text=example.prompt_text,
            response_a=example.chosen_response,
            response_b=example.rejected_response,
            selected_rubrics=selected_rubrics,
            rubric_note=_coverage_retry_note(
                required_count=expected_count,
                top_k=top_k,
                attempt_idx=1,
            ),
        )
        if not _prediction_has_full_rubric_coverage(prediction, expected_count):
            print(
                "[SVR] rubricbench partial coverage; switching to single-rubric recovery: "
                f"example_id={example.example_id} top_k={top_k} expected={expected_count} "
                f"actual_selected={len(getattr(prediction, 'selected_rubrics', []) or [])} "
                f"actual_comp={len(getattr(prediction, 'rubric_comparisons', []) or [])}",
                flush=True,
            )
            prediction = _recover_rubrics_individually(
                example=example,
                engine=engine,
                selected_rubrics=selected_rubrics,
                score_retry_times=score_retry_times,
                top_k=top_k,
                tie_margin=tie_margin,
            )

        prediction = _apply_low_evidence_direct_judge(
            example=example,
            engine=engine,
            prediction=prediction,
            max_effective_rubrics=low_evidence_max_effective_rubrics,
        )
        prediction = _apply_low_evidence_redundancy_guard(
            prediction=prediction,
            per_facet_cap=max_same_facet,
        )

        if (
            prediction.decision_source
            in {
                "rubric_margin",
                "rubric_margin_recovered",
                "rubric_margin_recovered_partial",
                "low_evidence_vote",
                "low_evidence_facet_capped_vote",
            }
            and abs(float(prediction.weighted_margin)) > tie_margin
        ):
            return prediction
        if prediction.decision_source in {
            "low_evidence_direct_compare",
            "low_evidence_keep",
            "direct_overall_judge",
        }:
            return prediction
        if top_k >= final_max_top_k:
            return prediction

        top_k += 1


async def _ascore_pair_strict_rubricbench(
    *,
    example: PreferenceExample,
    engine: SVRInferenceEngine,
    initial_top_k: int,
    required_rubrics: int,
    score_retry_times: int,
    max_top_k: int | None,
    selection_pool_size: int,
    max_same_facet: int,
    diversity_similarity_threshold: float,
    diversity_lambda: float,
    low_evidence_max_effective_rubrics: int,
):
    bank_size = len(engine.bank)
    if bank_size <= 0:
        raise ValueError("SVR bank is empty; cannot evaluate RubricBench.")
    if bank_size < required_rubrics:
        raise ValueError(
            f"SVR bank size {bank_size} is smaller than required_rubrics={required_rubrics}."
        )

    start_top_k = max(initial_top_k, required_rubrics)
    final_max_top_k = bank_size if max_top_k is None else min(max_top_k, bank_size)
    if final_max_top_k < required_rubrics:
        raise ValueError(
            f"max_top_k={final_max_top_k} is smaller than required_rubrics={required_rubrics}."
        )

    tie_margin = _scorer_tie_margin(engine)
    top_k = min(start_top_k, final_max_top_k)

    while True:
        selected_rubrics = await _aselect_diverse_rubrics(
            engine=engine,
            prompt_text=example.prompt_text,
            top_k=top_k,
            selection_pool_size=selection_pool_size,
            max_same_facet=max_same_facet,
            diversity_similarity_threshold=diversity_similarity_threshold,
            diversity_lambda=diversity_lambda,
        )
        expected_count = len(selected_rubrics)
        if expected_count < required_rubrics:
            if top_k < final_max_top_k:
                top_k += 1
                continue
            raise RuntimeError(
                "RubricBench selector returned too few rubrics: "
                f"example_id={example.example_id} selected={expected_count} "
                f"required={required_rubrics} max_top_k={final_max_top_k}"
            )

        prediction = await engine.scorer.acompare_responses(
            prompt_messages=example.prompt_messages,
            prompt_text=example.prompt_text,
            response_a=example.chosen_response,
            response_b=example.rejected_response,
            selected_rubrics=selected_rubrics,
            rubric_note=_coverage_retry_note(
                required_count=expected_count,
                top_k=top_k,
                attempt_idx=1,
            ),
        )
        if not _prediction_has_full_rubric_coverage(prediction, expected_count):
            print(
                "[SVR] rubricbench partial coverage; switching to single-rubric recovery: "
                f"example_id={example.example_id} top_k={top_k} expected={expected_count} "
                f"actual_selected={len(getattr(prediction, 'selected_rubrics', []) or [])} "
                f"actual_comp={len(getattr(prediction, 'rubric_comparisons', []) or [])}",
                flush=True,
            )
            prediction = await _arecover_rubrics_individually(
                example=example,
                engine=engine,
                selected_rubrics=selected_rubrics,
                score_retry_times=score_retry_times,
                top_k=top_k,
                tie_margin=tie_margin,
            )

        prediction = await _aapply_low_evidence_direct_judge(
            example=example,
            engine=engine,
            prediction=prediction,
            max_effective_rubrics=low_evidence_max_effective_rubrics,
        )
        prediction = _apply_low_evidence_redundancy_guard(
            prediction=prediction,
            per_facet_cap=max_same_facet,
        )

        if (
            prediction.decision_source
            in {
                "rubric_margin",
                "rubric_margin_recovered",
                "rubric_margin_recovered_partial",
                "low_evidence_vote",
                "low_evidence_facet_capped_vote",
            }
            and abs(float(prediction.weighted_margin)) > tie_margin
        ):
            return prediction
        if prediction.decision_source in {
            "low_evidence_direct_compare",
            "low_evidence_keep",
            "direct_overall_judge",
        }:
            return prediction
        if top_k >= final_max_top_k:
            return prediction

        top_k += 1


def _skipped_payload(
    *,
    example: PreferenceExample,
    reason: str,
) -> dict[str, Any]:
    payload = _build_identity_payload(example)
    payload.update(
        {
            "skipped": True,
            "reason": reason,
            "correct": False,
            "preferred_position": None,
            "margin_vs_chosen": None,
            "prediction": None,
        }
    )
    return payload


def _summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total_items = len(records)
    usable_records = [item for item in records if not item.get("skipped")]
    usable_items = len(usable_records)
    correct_items = sum(int(bool(item.get("correct"))) for item in usable_records)
    tie_items = sum(int(item.get("preferred_position") == "tie") for item in usable_records)
    margins = [
        float(item["margin_vs_chosen"])
        for item in usable_records
        if isinstance(item.get("margin_vs_chosen"), (int, float))
    ]
    return {
        "total_items": total_items,
        "usable_items": usable_items,
        "skipped_items": total_items - usable_items,
        "correct_items": correct_items,
        "incorrect_items": usable_items - correct_items,
        "accuracy": correct_items / usable_items if usable_items else 0.0,
        "tie_rate": tie_items / usable_items if usable_items else 0.0,
        "avg_margin": sum(margins) / usable_items if usable_items else 0.0,
    }


def _summarize_by_domain(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        grouped[str(item.get("domain") or "unknown")].append(item)
    return {
        domain: _summarize_records(items)
        for domain, items in sorted(grouped.items())
    }


def _reference_rubric_payload(rubrics: Sequence[RubricItem]) -> list[dict[str, Any]]:
    selected = []
    for idx, rubric in enumerate(rubrics):
        selected.append(
            {
                "bank_id": idx,
                "text": rubric.text,
                "facet": rubric.facet,
                "importance": rubric.importance,
                "source": rubric.source,
                "grounding": rubric.grounding,
                "selection_weight": importance_weight(rubric.importance),
            }
        )
    return selected


def _evaluate_svr(
    *,
    examples: Sequence[PreferenceExample],
    engine: SVRInferenceEngine,
    top_k: int,
    required_rubrics: int,
    score_retry_times: int,
    max_top_k: int | None,
    selection_pool_size: int,
    max_same_facet: int,
    diversity_similarity_threshold: float,
    diversity_lambda: float,
    low_evidence_max_effective_rubrics: int,
    progress_log_interval: int,
    max_concurrency: int,
) -> list[dict[str, Any]]:
    if max_concurrency > 1:
        return asyncio.run(
            _evaluate_svr_async(
                examples=examples,
                engine=engine,
                top_k=top_k,
                required_rubrics=required_rubrics,
                score_retry_times=score_retry_times,
                max_top_k=max_top_k,
                selection_pool_size=selection_pool_size,
                max_same_facet=max_same_facet,
                diversity_similarity_threshold=diversity_similarity_threshold,
                diversity_lambda=diversity_lambda,
                low_evidence_max_effective_rubrics=low_evidence_max_effective_rubrics,
                progress_log_interval=progress_log_interval,
                max_concurrency=max_concurrency,
            )
        )

    records: list[dict[str, Any]] = []
    total = len(examples)
    for idx, example in enumerate(examples, start=1):
        try:
            scored_example, gold_side = _oriented_example(
                example,
                salt="rubricbench:svr",
            )
            prediction = _score_pair_strict_rubricbench(
                example=scored_example,
                engine=engine,
                initial_top_k=top_k,
                required_rubrics=required_rubrics,
                score_retry_times=score_retry_times,
                max_top_k=max_top_k,
                selection_pool_size=selection_pool_size,
                max_same_facet=max_same_facet,
                diversity_similarity_threshold=diversity_similarity_threshold,
                diversity_lambda=diversity_lambda,
                low_evidence_max_effective_rubrics=low_evidence_max_effective_rubrics,
            )
            records.append(
                _prediction_payload(
                    example=example,
                    prediction=prediction,
                    correct=prediction.preferred_side == gold_side,
                    gold_side=gold_side,
                )
            )
        except Exception as exc:  # noqa: BLE001
            reason = f"strict_eval_failure: {type(exc).__name__}: {exc}"
            print(
                "[SVR] rubricbench strict eval skipped: "
                f"example_id={example.example_id} error={type(exc).__name__}: {exc}",
                flush=True,
            )
            records.append(_skipped_payload(example=example, reason=reason))
        if progress_log_interval > 0 and (idx % progress_log_interval == 0 or idx == total):
            print(f"[SVR] rubricbench svr progress: {idx}/{total}", flush=True)
    return records


async def _evaluate_svr_async(
    *,
    examples: Sequence[PreferenceExample],
    engine: SVRInferenceEngine,
    top_k: int,
    required_rubrics: int,
    score_retry_times: int,
    max_top_k: int | None,
    selection_pool_size: int,
    max_same_facet: int,
    diversity_similarity_threshold: float,
    diversity_lambda: float,
    low_evidence_max_effective_rubrics: int,
    progress_log_interval: int,
    max_concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
    records: list[dict[str, Any] | None] = [None] * len(examples)

    async def _process(
        idx: int,
        example: PreferenceExample,
    ) -> tuple[int, dict[str, Any]]:
        async with semaphore:
            try:
                scored_example, gold_side = _oriented_example(
                    example,
                    salt="rubricbench:svr",
                )
                prediction = await _ascore_pair_strict_rubricbench(
                    example=scored_example,
                    engine=engine,
                    initial_top_k=top_k,
                    required_rubrics=required_rubrics,
                    score_retry_times=score_retry_times,
                    max_top_k=max_top_k,
                    selection_pool_size=selection_pool_size,
                    max_same_facet=max_same_facet,
                    diversity_similarity_threshold=diversity_similarity_threshold,
                    diversity_lambda=diversity_lambda,
                    low_evidence_max_effective_rubrics=low_evidence_max_effective_rubrics,
                )
                return idx, _prediction_payload(
                    example=example,
                    prediction=prediction,
                    correct=prediction.preferred_side == gold_side,
                    gold_side=gold_side,
                )
            except Exception as exc:  # noqa: BLE001
                reason = f"strict_eval_failure: {type(exc).__name__}: {exc}"
                print(
                    "[SVR] rubricbench strict eval skipped: "
                    f"example_id={example.example_id} error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                return idx, _skipped_payload(example=example, reason=reason)

    tasks = [
        asyncio.create_task(_process(idx, example))
        for idx, example in enumerate(examples)
    ]
    completed = 0
    total = len(tasks)
    for task in asyncio.as_completed(tasks):
        idx, record = await task
        records[idx] = record
        completed += 1
        if progress_log_interval > 0 and (
            completed % progress_log_interval == 0 or completed == total
        ):
            print(f"[SVR] rubricbench svr progress: {completed}/{total}", flush=True)

    return [record for record in records if record is not None]


def _evaluate_reference(
    *,
    examples: Sequence[PreferenceExample],
    engine: SVRInferenceEngine,
    progress_log_interval: int,
    max_concurrency: int,
) -> list[dict[str, Any]]:
    if max_concurrency > 1:
        return asyncio.run(
            _evaluate_reference_async(
                examples=examples,
                engine=engine,
                progress_log_interval=progress_log_interval,
                max_concurrency=max_concurrency,
            )
        )

    records: list[dict[str, Any]] = []
    total = len(examples)
    for idx, example in enumerate(examples, start=1):
        if not example.reference_rubrics:
            records.append(
                _skipped_payload(
                    example=example,
                    reason="missing reference rubrics",
                )
            )
        else:
            oriented = orient_preference_pair(
                example_id=example.example_id,
                positive_response=example.chosen_response,
                negative_response=example.rejected_response,
                salt="rubricbench:reference",
            )
            prediction = engine.scorer.compare_responses(
                prompt_messages=example.prompt_messages,
                prompt_text=example.prompt_text,
                response_a=oriented.response_a,
                response_b=oriented.response_b,
                selected_rubrics=_reference_rubric_payload(example.reference_rubrics),
                difficulty_analysis=example.difficulty_analysis,
            )
            records.append(
                _prediction_payload(
                    example=example,
                    prediction=prediction,
                    correct=prediction.preferred_side == oriented.gold_side,
                    gold_side=oriented.gold_side,
                )
            )
        if progress_log_interval > 0 and (idx % progress_log_interval == 0 or idx == total):
            print(f"[SVR] rubricbench reference progress: {idx}/{total}", flush=True)
    return records


async def _evaluate_reference_async(
    *,
    examples: Sequence[PreferenceExample],
    engine: SVRInferenceEngine,
    progress_log_interval: int,
    max_concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
    records: list[dict[str, Any] | None] = [None] * len(examples)

    async def _process(
        idx: int,
        example: PreferenceExample,
    ) -> tuple[int, dict[str, Any]]:
        async with semaphore:
            if not example.reference_rubrics:
                return idx, _skipped_payload(
                    example=example,
                    reason="missing reference rubrics",
                )
            oriented = orient_preference_pair(
                example_id=example.example_id,
                positive_response=example.chosen_response,
                negative_response=example.rejected_response,
                salt="rubricbench:reference",
            )
            prediction = await engine.scorer.acompare_responses(
                prompt_messages=example.prompt_messages,
                prompt_text=example.prompt_text,
                response_a=oriented.response_a,
                response_b=oriented.response_b,
                selected_rubrics=_reference_rubric_payload(example.reference_rubrics),
                difficulty_analysis=example.difficulty_analysis,
            )
            return idx, _prediction_payload(
                example=example,
                prediction=prediction,
                correct=prediction.preferred_side == oriented.gold_side,
                gold_side=oriented.gold_side,
            )

    tasks = [
        asyncio.create_task(_process(idx, example))
        for idx, example in enumerate(examples)
    ]
    completed = 0
    total = len(tasks)
    for task in asyncio.as_completed(tasks):
        idx, record = await task
        records[idx] = record
        completed += 1
        if progress_log_interval > 0 and (
            completed % progress_log_interval == 0 or completed == total
        ):
            print(f"[SVR] rubricbench reference progress: {completed}/{total}", flush=True)

    return [record for record in records if record is not None]


def _build_details_records(
    *,
    svr_records: Sequence[dict[str, Any]],
    reference_records: Sequence[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for idx, svr_record in enumerate(svr_records):
        detail = {
            key: svr_record.get(key)
            for key in (
                "example_id",
                "record_idx",
                "domain",
                "raw_domain",
                "case_id",
                "source_id",
                "chosen_candidate",
                "preferred_candidate",
            )
        }
        detail["svr"] = {
            key: svr_record.get(key)
            for key in (
                "skipped",
                "reason",
                "correct",
                "preferred_position",
                "margin_vs_chosen",
                "prediction",
            )
        }
        if reference_records is not None:
            reference_record = reference_records[idx]
            detail["reference"] = {
                key: reference_record.get(key)
                for key in (
                    "skipped",
                    "reason",
                    "correct",
                    "preferred_position",
                    "margin_vs_chosen",
                    "prediction",
                )
            }
        details.append(detail)
    return details


def _write_jsonl(path: str, records: Sequence[dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False))
            file_obj.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SVR model directly on RubricBench."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test-path", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--domain",
        action="append",
        default=None,
        help="Repeat to keep only selected RubricBench domains, matched case-insensitively.",
    )
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument(
        "--required-rubrics",
        type=int,
        default=6,
        help="Minimum number of rubric comparisons required at each scoring attempt.",
    )
    parser.add_argument(
        "--score-retry-times",
        type=int,
        default=5,
        help="Maximum retries per single-rubric recovery when the initial full top_k judge returns partial coverage.",
    )
    parser.add_argument(
        "--max-top-k",
        type=int,
        default=None,
        help="Increase top_k up to this value when rubric evidence is still tied. Default: bank size.",
    )
    parser.add_argument(
        "--selection-pool-size",
        type=int,
        default=18,
        help="Initial larger rubric pool size before diversity reranking.",
    )
    parser.add_argument(
        "--max-same-facet",
        type=int,
        default=2,
        help="Soft cap for how many selected rubrics may share the same facet.",
    )
    parser.add_argument(
        "--diversity-similarity-threshold",
        type=float,
        default=0.92,
        help="Similarity threshold above which near-duplicate rubrics receive an extra penalty.",
    )
    parser.add_argument(
        "--diversity-lambda",
        type=float,
        default=0.35,
        help="Penalty weight applied to max similarity during diversity reranking.",
    )
    parser.add_argument(
        "--low-evidence-max-effective-rubrics",
        type=int,
        default=2,
        help="When the selector gives at most this many non-zero-weight rubrics, fall back to rubric-vote/direct-compare aggregation.",
    )
    parser.add_argument(
        "--eval-llm-reasoning-effort",
        default="medium",
        help="Override real-LLM reasoning effort during eval to reduce long reasoning truncation.",
    )
    parser.add_argument(
        "--eval-judge-rubric-chunk-size",
        type=int,
        default=3,
        help="Override scorer chunk size during eval; smaller chunks reduce JSON truncation risk.",
    )
    parser.add_argument(
        "--eval-score-max-tokens",
        type=int,
        default=8192,
        help="Override pairwise judge max tokens during eval.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress-log-interval", type=int, default=64)
    parser.add_argument(
        "--eval-max-concurrency",
        type=int,
        default=64,
        help="Concurrent examples for eval.",
    )
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--details-path", default=None)
    parser.add_argument(
        "--disable-rubric-rewrite",
        action="store_true",
        help="Override model inference config and skip eval-time rubric rewriting.",
    )
    parser.add_argument("--with-reference-baseline", action="store_true")
    parser.add_argument(
        "--reference-rubric-field",
        action="append",
        default=list(DEFAULT_REFERENCE_RUBRIC_FIELD_CANDIDATES),
        help="Fields used when --with-reference-baseline is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_paths = list(args.test_path)
    print(
        "[SVR] RubricBench eval start: "
        f"model_dir={args.model_dir} test_paths={test_paths}",
        flush=True,
    )
    examples = load_preference_examples(
        paths=test_paths,
        self_rubric_fields=(),
        reference_rubric_fields=args.reference_rubric_field,
        limit=None,
    )
    print(f"[SVR] loaded rubricbench examples={len(examples)}", flush=True)

    allowed_domains = None
    if args.domain:
        allowed_domains = {_normalize_domain(item) for item in args.domain}
        examples = [
            example for example in examples if _example_domain(example) in allowed_domains
        ]
        print(
            f"[SVR] domain filter={sorted(allowed_domains)} kept_examples={len(examples)}",
            flush=True,
        )

    if args.limit is not None:
        examples = examples[: args.limit]
        print(f"[SVR] limited examples={len(examples)}", flush=True)

    if not examples:
        raise ValueError("No RubricBench examples left after filtering.")

    output_path = args.output_path or os.path.join(
        args.model_dir, "rubricbench_eval_summary.json"
    )
    details_path = args.details_path or os.path.join(
        args.model_dir, "rubricbench_eval_details.jsonl"
    )

    engine = SVRInferenceEngine(
        model_dir=args.model_dir,
        device=args.device,
    )
    runner = getattr(getattr(engine, "scorer", None), "config", None)
    runner = getattr(runner, "runner", None)
    if runner is not None:
        if args.eval_llm_reasoning_effort:
            runner.config.reasoning_effort = str(args.eval_llm_reasoning_effort)
            engine.inference_config["llm_reasoning_effort"] = str(
                args.eval_llm_reasoning_effort
            )
        scorer_config = getattr(engine.scorer, "config", None)
        if scorer_config is not None:
            scorer_config.chunk_size = max(1, int(args.eval_judge_rubric_chunk_size))
            scorer_config.judge_max_tokens = max(512, int(args.eval_score_max_tokens))
            engine.inference_config["judge_rubric_chunk_size"] = int(
                scorer_config.chunk_size
            )
            engine.inference_config["score_max_tokens"] = int(
                scorer_config.judge_max_tokens
            )
        print(
            "[SVR] eval LLM overrides: "
            f"reasoning_effort={engine.inference_config.get('llm_reasoning_effort')} "
            f"judge_rubric_chunk_size={engine.inference_config.get('judge_rubric_chunk_size')} "
            f"score_max_tokens={engine.inference_config.get('score_max_tokens')}",
            flush=True,
        )
    if args.disable_rubric_rewrite:
        engine.inference_config["rewrite_selected_rubrics"] = False
        engine.rewriter = NoOpRubricRewriter()
        print("[SVR] eval override: disable rubric rewrite", flush=True)
    if args.eval_max_concurrency > 1:
        print(
            f"[SVR] eval concurrency enabled: max_concurrency={args.eval_max_concurrency}",
            flush=True,
        )
    svr_records = _evaluate_svr(
        examples=examples,
        engine=engine,
        top_k=args.top_k,
        required_rubrics=args.required_rubrics,
        score_retry_times=args.score_retry_times,
        max_top_k=args.max_top_k,
        selection_pool_size=args.selection_pool_size,
        max_same_facet=args.max_same_facet,
        diversity_similarity_threshold=args.diversity_similarity_threshold,
        diversity_lambda=args.diversity_lambda,
        low_evidence_max_effective_rubrics=args.low_evidence_max_effective_rubrics,
        progress_log_interval=args.progress_log_interval,
        max_concurrency=args.eval_max_concurrency,
    )

    reference_records = None
    if args.with_reference_baseline:
        reference_records = _evaluate_reference(
            examples=examples,
            engine=engine,
            progress_log_interval=args.progress_log_interval,
            max_concurrency=args.eval_max_concurrency,
        )

    detail_records = _build_details_records(
        svr_records=svr_records,
        reference_records=reference_records,
    )
    _write_jsonl(details_path, detail_records)

    domain_counts = Counter(_example_domain(example) for example in examples)
    summary = {
        "model_dir": public_path(args.model_dir),
        "test_paths": [public_path(path) for path in test_paths],
        "output_path": public_path(output_path),
        "details_path": public_path(details_path),
        "domain_filter": sorted(allowed_domains) if allowed_domains else None,
        "dataset": {
            "total_items": len(examples),
            "domain_counts": dict(sorted(domain_counts.items())),
        },
        "inference": {
            "top_k": args.top_k,
            "required_rubrics": args.required_rubrics,
            "score_retry_times": args.score_retry_times,
            "max_top_k": args.max_top_k if args.max_top_k is not None else len(engine.bank),
            "selection_pool_size": args.selection_pool_size,
            "max_same_facet": args.max_same_facet,
            "diversity_similarity_threshold": args.diversity_similarity_threshold,
            "diversity_lambda": args.diversity_lambda,
            "low_evidence_max_effective_rubrics": args.low_evidence_max_effective_rubrics,
            "device": args.device,
            "eval_max_concurrency": args.eval_max_concurrency,
            "bank_size": len(engine.bank),
            "inference_config": dict(engine.inference_config),
        },
        "svr": {
            "summary": _summarize_records(svr_records),
            "by_domain": _summarize_by_domain(svr_records),
        },
        "reference": (
            {
                "summary": _summarize_records(reference_records),
                "by_domain": _summarize_by_domain(reference_records),
            }
            if reference_records is not None
            else None
        ),
    }
    dump_json(output_path, summary)

    print(f"Saved RubricBench summary to {output_path}")
    print(f"Saved RubricBench details to {details_path}")
    print(
        "svr_accuracy={:.4f} tie_rate={:.4f}".format(
            summary["svr"]["summary"]["accuracy"],
            summary["svr"]["summary"]["tie_rate"],
        )
    )
    if summary["reference"] is not None:
        print(
            "reference_accuracy={:.4f} tie_rate={:.4f}".format(
                summary["reference"]["summary"]["accuracy"],
                summary["reference"]["summary"]["tie_rate"],
            )
        )


if __name__ == "__main__":
    main()
