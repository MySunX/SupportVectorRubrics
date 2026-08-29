from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from svr.rewardbench_eval_base import (
    RewardBenchItem,
    add_shared_eval_args,
    astrict_score_pair,
    build_eval_engine,
    build_identity_payload,
    build_inference_summary,
    build_svr_payload,
    dump_json,
    extract_prompt_from_any,
    group_summary,
    json_safe,
    normalize_text_list,
    resolve_input_path,
    run_eval_loop,
    strict_eval_config_from_args,
    strict_score_pair,
    subset_key,
    summarize_best_of_n,
    write_jsonl,
)
from svr.utils import gold_aligned_margin, orient_preference_pair, preferred_position, public_path

SUMMARY_FILENAME = "rewardbench2_eval_summary.json"
DETAIL_FILENAME = "rewardbench2_eval_details.jsonl"


def _load_rewardbench2_items(
    *,
    path: str,
    subset_filters: set[str] | None,
    limit: int | None,
) -> list[RewardBenchItem]:
    frame = pd.read_parquet(path)
    items: list[RewardBenchItem] = []
    for record_idx, raw_record in enumerate(frame.to_dict(orient="records")):
        prompt_text, prompt_messages = extract_prompt_from_any(raw_record.get("prompt"))
        subset = str(raw_record.get("subset") or "unknown")
        if subset_filters and subset_key(subset) not in subset_filters:
            continue
        chosen_responses = normalize_text_list(raw_record.get("chosen"))
        rejected_responses = normalize_text_list(raw_record.get("rejected"))
        if not chosen_responses or not rejected_responses:
            continue
        global_idx = len(items)
        example_id = str(raw_record.get("id") if raw_record.get("id") is not None else global_idx)
        items.append(
            RewardBenchItem(
                benchmark="reward-bench-2",
                example_id=example_id,
                prompt_text=prompt_text,
                prompt_messages=prompt_messages,
                chosen_responses=chosen_responses,
                rejected_responses=rejected_responses,
                subset=subset,
                meta={
                    "record_idx": record_idx,
                    "global_idx": global_idx,
                    "source_path": public_path(path),
                    "num_correct": int(raw_record.get("num_correct") or len(chosen_responses)),
                    "num_incorrect": int(raw_record.get("num_incorrect") or len(rejected_responses)),
                    "total_completions": int(
                        raw_record.get("total_completions")
                        or (len(chosen_responses) + len(rejected_responses))
                    ),
                    "models": json_safe(raw_record.get("models")),
                    "additional_metadata": json_safe(raw_record.get("additional_metadata")),
                },
                raw_record=json_safe(raw_record),
            )
        )
        if limit is not None and len(items) >= limit:
            break
    return items


def _score_nonties_item(*, item: RewardBenchItem, engine, config) -> dict[str, Any]:
    pair_results = []
    correct_count = 0
    margins: list[float] = []
    for rejected_idx, rejected_response in enumerate(item.rejected_responses):
        pair_id = f"{item.example_id}:rejected:{rejected_idx}"
        oriented = orient_preference_pair(
            example_id=pair_id,
            positive_response=item.chosen_responses[0],
            negative_response=rejected_response,
            salt="rewardbench2:nonties",
        )
        prediction = strict_score_pair(
            example_id=pair_id,
            prompt_text=item.prompt_text,
            prompt_messages=item.prompt_messages,
            response_a=oriented.response_a,
            response_b=oriented.response_b,
            engine=engine,
            config=config,
            meta=item.meta,
            raw_record=item.raw_record,
        )
        is_correct = prediction.preferred_side == oriented.gold_side
        margin_vs_chosen = gold_aligned_margin(
            prediction.weighted_margin,
            oriented.gold_side,
        )
        correct_count += int(is_correct)
        margins.append(margin_vs_chosen)
        pair_results.append(
            {
                "rejected_idx": rejected_idx,
                "correct": is_correct,
                "preferred_position": preferred_position(
                    prediction.preferred_side,
                    gold_side=oriented.gold_side,
                ),
                "margin_vs_chosen": margin_vs_chosen,
                "svr": build_svr_payload(prediction, gold_side=oriented.gold_side),
            }
        )

    payload = build_identity_payload(item)
    payload.update(
        {
            "skipped": False,
            "correct": correct_count == len(item.rejected_responses),
            "pair_correct_count": correct_count,
            "pair_total": len(item.rejected_responses),
            "min_margin": min(margins) if margins else None,
            "max_margin": max(margins) if margins else None,
            "pair_results": pair_results,
            "score_method": "strict_chosen_vs_each_rejected",
        }
    )
    return payload


async def _ascore_nonties_item(*, item: RewardBenchItem, engine, config) -> dict[str, Any]:
    pair_results = []
    correct_count = 0
    margins: list[float] = []
    for rejected_idx, rejected_response in enumerate(item.rejected_responses):
        pair_id = f"{item.example_id}:rejected:{rejected_idx}"
        oriented = orient_preference_pair(
            example_id=pair_id,
            positive_response=item.chosen_responses[0],
            negative_response=rejected_response,
            salt="rewardbench2:nonties",
        )
        prediction = await astrict_score_pair(
            example_id=pair_id,
            prompt_text=item.prompt_text,
            prompt_messages=item.prompt_messages,
            response_a=oriented.response_a,
            response_b=oriented.response_b,
            engine=engine,
            config=config,
            meta=item.meta,
            raw_record=item.raw_record,
        )
        is_correct = prediction.preferred_side == oriented.gold_side
        margin_vs_chosen = gold_aligned_margin(
            prediction.weighted_margin,
            oriented.gold_side,
        )
        correct_count += int(is_correct)
        margins.append(margin_vs_chosen)
        pair_results.append(
            {
                "rejected_idx": rejected_idx,
                "correct": is_correct,
                "preferred_position": preferred_position(
                    prediction.preferred_side,
                    gold_side=oriented.gold_side,
                ),
                "margin_vs_chosen": margin_vs_chosen,
                "svr": build_svr_payload(prediction, gold_side=oriented.gold_side),
            }
        )

    payload = build_identity_payload(item)
    payload.update(
        {
            "skipped": False,
            "correct": correct_count == len(item.rejected_responses),
            "pair_correct_count": correct_count,
            "pair_total": len(item.rejected_responses),
            "min_margin": min(margins) if margins else None,
            "max_margin": max(margins) if margins else None,
            "pair_results": pair_results,
            "score_method": "strict_chosen_vs_each_rejected",
        }
    )
    return payload


def _compute_ties_prompt_stats(
    scores: Sequence[float],
    *,
    num_correct: int,
) -> tuple[bool, float | None, float | None]:
    correct_scores = [float(score) for score in scores[:num_correct]]
    incorrect_scores = [float(score) for score in scores[num_correct:]]
    if not correct_scores or not incorrect_scores:
        return False, None, None
    best_correct = max(correct_scores)
    worst_correct = min(correct_scores)
    best_incorrect = max(incorrect_scores)
    different_correct_margin = (
        best_correct - worst_correct if len(correct_scores) > 1 else None
    )
    correct_incorrect_margin = worst_correct - best_incorrect
    accurate = correct_incorrect_margin > 0
    return accurate, different_correct_margin, correct_incorrect_margin


def _score_ties_item(*, item: RewardBenchItem, engine, config) -> dict[str, Any]:
    completions = item.chosen_responses + item.rejected_responses
    completion_scores = [0.0] * len(completions)
    pair_results = []
    pair_count = 0
    for left_idx in range(len(completions)):
        for right_idx in range(left_idx + 1, len(completions)):
            pair_id = f"{item.example_id}:pair:{left_idx}:{right_idx}"
            oriented = orient_preference_pair(
                example_id=pair_id,
                positive_response=completions[left_idx],
                negative_response=completions[right_idx],
                salt="rewardbench2:ties",
            )
            prediction = strict_score_pair(
                example_id=pair_id,
                prompt_text=item.prompt_text,
                prompt_messages=item.prompt_messages,
                response_a=oriented.response_a,
                response_b=oriented.response_b,
                engine=engine,
                config=config,
                meta=item.meta,
                raw_record=item.raw_record,
            )
            margin = gold_aligned_margin(prediction.weighted_margin, oriented.gold_side)
            completion_scores[left_idx] += margin
            completion_scores[right_idx] -= margin
            pair_results.append(
                {
                    "left_idx": left_idx,
                    "right_idx": right_idx,
                    "preferred_position": preferred_position(
                        prediction.preferred_side,
                        gold_side=oriented.gold_side,
                        positive_label="left",
                        negative_label="right",
                    ),
                    "margin_vs_left": margin,
                    "svr": build_svr_payload(
                        prediction,
                        gold_side=oriented.gold_side,
                        positive_label="left",
                        negative_label="right",
                        margin_key="margin_vs_left",
                    ),
                }
            )
            pair_count += 1

    divisor = max(1, len(completions) - 1)
    averaged_scores = [score / divisor for score in completion_scores]
    num_correct = len(item.chosen_responses)
    derived_correct, different_correct_margin, correct_incorrect_margin = (
        _compute_ties_prompt_stats(averaged_scores, num_correct=num_correct)
    )
    sample_type = "tied"
    prompt_group_id = item.example_id
    if ":" in item.example_id:
        sample_type, prompt_group_id = item.example_id.split(":", 1)

    payload = build_identity_payload(item)
    payload.update(
        {
            "skipped": False,
            "correct": None,
            "derived_correct": derived_correct,
            "different_correct_margin": different_correct_margin,
            "correct_incorrect_margin": correct_incorrect_margin,
            "completion_scores": averaged_scores,
            "pair_total": pair_count,
            "pair_results": pair_results,
            "sample_type": sample_type,
            "prompt_group_id": prompt_group_id,
            "score_method": "strict_pairwise_round_robin_margin",
        }
    )
    return payload


async def _ascore_ties_item(*, item: RewardBenchItem, engine, config) -> dict[str, Any]:
    completions = item.chosen_responses + item.rejected_responses
    completion_scores = [0.0] * len(completions)
    pair_results = []
    pair_count = 0
    for left_idx in range(len(completions)):
        for right_idx in range(left_idx + 1, len(completions)):
            pair_id = f"{item.example_id}:pair:{left_idx}:{right_idx}"
            oriented = orient_preference_pair(
                example_id=pair_id,
                positive_response=completions[left_idx],
                negative_response=completions[right_idx],
                salt="rewardbench2:ties",
            )
            prediction = await astrict_score_pair(
                example_id=pair_id,
                prompt_text=item.prompt_text,
                prompt_messages=item.prompt_messages,
                response_a=oriented.response_a,
                response_b=oriented.response_b,
                engine=engine,
                config=config,
                meta=item.meta,
                raw_record=item.raw_record,
            )
            margin = gold_aligned_margin(prediction.weighted_margin, oriented.gold_side)
            completion_scores[left_idx] += margin
            completion_scores[right_idx] -= margin
            pair_results.append(
                {
                    "left_idx": left_idx,
                    "right_idx": right_idx,
                    "preferred_position": preferred_position(
                        prediction.preferred_side,
                        gold_side=oriented.gold_side,
                        positive_label="left",
                        negative_label="right",
                    ),
                    "margin_vs_left": margin,
                    "svr": build_svr_payload(
                        prediction,
                        gold_side=oriented.gold_side,
                        positive_label="left",
                        negative_label="right",
                        margin_key="margin_vs_left",
                    ),
                }
            )
            pair_count += 1

    divisor = max(1, len(completions) - 1)
    averaged_scores = [score / divisor for score in completion_scores]
    num_correct = len(item.chosen_responses)
    derived_correct, different_correct_margin, correct_incorrect_margin = (
        _compute_ties_prompt_stats(averaged_scores, num_correct=num_correct)
    )
    sample_type = "tied"
    prompt_group_id = item.example_id
    if ":" in item.example_id:
        sample_type, prompt_group_id = item.example_id.split(":", 1)

    payload = build_identity_payload(item)
    payload.update(
        {
            "skipped": False,
            "correct": None,
            "derived_correct": derived_correct,
            "different_correct_margin": different_correct_margin,
            "correct_incorrect_margin": correct_incorrect_margin,
            "completion_scores": averaged_scores,
            "pair_total": pair_count,
            "pair_results": pair_results,
            "sample_type": sample_type,
            "prompt_group_id": prompt_group_id,
            "score_method": "strict_pairwise_round_robin_margin",
        }
    )
    return payload


def _aggregate_ties_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "total_items": 0,
            "ref_items": 0,
            "tied_items": 0,
            "prompt_pairs": 0,
            "ref_accuracy": 0.0,
            "tied_accuracy": 0.0,
            "correctness_preferred": 0.0,
            "correctness_preferred_hard": 0.0,
            "correctness_margin_score": 0.0,
            "overall_score": 0.0,
        }

    ref_stats: dict[str, tuple[bool, float | None, float | None]] = {}
    tied_stats: dict[str, tuple[bool, float | None, float | None]] = {}
    for record in records:
        sample_type = str(record.get("sample_type") or "")
        prompt_group_id = str(record.get("prompt_group_id") or "")
        stats = (
            bool(record.get("derived_correct")),
            (
                float(record["different_correct_margin"])
                if isinstance(record.get("different_correct_margin"), (int, float))
                else None
            ),
            (
                float(record["correct_incorrect_margin"])
                if isinstance(record.get("correct_incorrect_margin"), (int, float))
                else None
            ),
        )
        if sample_type == "ref":
            ref_stats[prompt_group_id] = stats
        elif sample_type == "tied":
            tied_stats[prompt_group_id] = stats

    ref_accuracy = float(np.mean([item[0] for item in ref_stats.values()])) if ref_stats else 0.0
    tied_accuracy = float(np.mean([item[0] for item in tied_stats.values()])) if tied_stats else 0.0

    shared_prompt_ids = sorted(set(ref_stats) & set(tied_stats))
    if shared_prompt_ids:
        diff_corr_margin = np.array(
            [
                tied_stats[prompt_id][1] if tied_stats[prompt_id][1] is not None else 0.0
                for prompt_id in shared_prompt_ids
            ],
            dtype=float,
        )
        corr_incorrect_ties = np.array(
            [
                tied_stats[prompt_id][2] if tied_stats[prompt_id][2] is not None else 0.0
                for prompt_id in shared_prompt_ids
            ],
            dtype=float,
        )
        corr_incorrect_ref = np.array(
            [
                ref_stats[prompt_id][2] if ref_stats[prompt_id][2] is not None else 0.0
                for prompt_id in shared_prompt_ids
            ],
            dtype=float,
        )
        correctness_preferred = float(np.mean(corr_incorrect_ties > diff_corr_margin))
        correctness_preferred_hard = float(
            np.mean(np.minimum(corr_incorrect_ref, corr_incorrect_ties) > diff_corr_margin)
        )
        correctness_margin_score = float(
            np.mean(np.minimum(corr_incorrect_ref, corr_incorrect_ties) - diff_corr_margin)
        )
    else:
        correctness_preferred = 0.0
        correctness_preferred_hard = 0.0
        correctness_margin_score = 0.0

    overall_score = (
        0.30 * tied_accuracy
        + 0.30 * ref_accuracy
        + 0.20 * correctness_preferred
        + 0.20 * correctness_preferred_hard
        + 0.01 * correctness_margin_score
    )
    return {
        "total_items": len(records),
        "ref_items": len(ref_stats),
        "tied_items": len(tied_stats),
        "prompt_pairs": len(shared_prompt_ids),
        "ref_accuracy": ref_accuracy,
        "tied_accuracy": tied_accuracy,
        "correctness_preferred": correctness_preferred,
        "correctness_preferred_hard": correctness_preferred_hard,
        "correctness_margin_score": correctness_margin_score,
        "overall_score": float(overall_score),
        "score_method": "official_rewardbench2_ties_formula_on_strict_pairwise_scores",
    }


def _build_summary(
    *,
    items: Sequence[RewardBenchItem],
    records: Sequence[dict[str, Any]],
    output_path: str,
    details_path: str,
    model_dir: str,
    test_path: str,
    device: str,
    max_concurrency: int,
    engine,
    config,
) -> dict[str, Any]:
    subset_counts = Counter(item.subset for item in items)
    non_ties_records = [record for record in records if subset_key(record.get("subset")) != "ties"]
    ties_records = [record for record in records if subset_key(record.get("subset")) == "ties"]
    by_subset: dict[str, Any] = {}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("subset") or "unknown")].append(record)
    for subset, subset_records in sorted(grouped.items()):
        if subset_key(subset) == "ties":
            by_subset[subset] = _aggregate_ties_records(subset_records)
        else:
            by_subset[subset] = summarize_best_of_n(subset_records)

    subset_scores: dict[str, float] = {}
    for subset, payload in by_subset.items():
        if subset_key(subset) == "ties":
            subset_scores[subset] = float(payload.get("overall_score") or 0.0)
        else:
            subset_scores[subset] = float(payload.get("accuracy") or 0.0)

    rewardbench2_score = (
        sum(subset_scores.values()) / len(subset_scores) if subset_scores else 0.0
    )
    return {
        "benchmark": "reward-bench-2",
        "model_dir": public_path(model_dir),
        "test_paths": [public_path(test_path)],
        "output_path": public_path(output_path),
        "details_path": public_path(details_path),
        "dataset": {
            "total_items": len(items),
            "subset_counts": dict(sorted(subset_counts.items())),
        },
        "inference": build_inference_summary(
            engine=engine,
            config=config,
            device=device,
            max_concurrency=max_concurrency,
        ),
        "svr": {
            "summary": {
                "total_items": len(records),
                "non_ties_items": len(non_ties_records),
                "ties_items": len(ties_records),
                "non_ties_accuracy": summarize_best_of_n(non_ties_records)["accuracy"]
                if non_ties_records
                else 0.0,
                "reward_bench_2_score": rewardbench2_score,
            },
            "by_subset": by_subset,
            "subset_scores": subset_scores,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SVR model on RewardBench 2 using RubricBench-style strict judging."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--test-path",
        required=True,
        help="RewardBench 2 parquet path or a directory containing it.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--subset",
        action="append",
        default=None,
        help="Repeat to keep only selected subsets, matched case-insensitively.",
    )
    add_shared_eval_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_path = resolve_input_path(
        args.test_path,
        preferred_file_names=["test-00000-of-00001.parquet"],
    )
    subset_filters = {subset_key(item) for item in args.subset} if args.subset else None
    config = strict_eval_config_from_args(args)
    print(
        "[SVR] RewardBench 2 eval start: "
        f"model_dir={args.model_dir} test_paths={[test_path]}",
        flush=True,
    )
    items = _load_rewardbench2_items(
        path=test_path,
        subset_filters=subset_filters,
        limit=args.limit,
    )
    if not items:
        raise ValueError("No RewardBench 2 items found for the requested inputs.")
    print(
        f"[SVR] loaded reward-bench-2 items={len(items)} subsets={sorted({item.subset for item in items})}",
        flush=True,
    )

    engine = build_eval_engine(
        model_dir=args.model_dir,
        device=args.device,
        eval_llm_reasoning_effort=args.eval_llm_reasoning_effort,
        eval_judge_rubric_chunk_size=args.eval_judge_rubric_chunk_size,
        eval_score_max_tokens=args.eval_score_max_tokens,
        eval_max_concurrency=args.eval_max_concurrency,
        disable_rubric_rewrite=args.disable_rubric_rewrite,
    )
    use_async = args.eval_max_concurrency > 1
    non_ties_items = [item for item in items if subset_key(item.subset) != "ties"]
    ties_items = [item for item in items if subset_key(item.subset) == "ties"]
    records: list[dict[str, Any]] = []
    if non_ties_items:
        records.extend(
            run_eval_loop(
                items=non_ties_items,
                sync_fn=lambda item: _score_nonties_item(item=item, engine=engine, config=config),
                async_fn=lambda item: _ascore_nonties_item(item=item, engine=engine, config=config),
                progress_label="reward-bench-2 non-ties progress",
                progress_log_interval=args.progress_log_interval,
                max_concurrency=args.eval_max_concurrency,
                use_async=use_async,
            )
        )
    if ties_items:
        records.extend(
            run_eval_loop(
                items=ties_items,
                sync_fn=lambda item: _score_ties_item(item=item, engine=engine, config=config),
                async_fn=lambda item: _ascore_ties_item(item=item, engine=engine, config=config),
                progress_label="reward-bench-2 ties progress",
                progress_log_interval=args.progress_log_interval,
                max_concurrency=args.eval_max_concurrency,
                use_async=use_async,
            )
        )
    records = sorted(records, key=lambda item: int(item.get("global_idx") or 0))

    output_path = args.output_path or os.path.join(args.model_dir, SUMMARY_FILENAME)
    details_path = args.details_path or os.path.join(args.model_dir, DETAIL_FILENAME)
    write_jsonl(details_path, records)
    summary = _build_summary(
        items=items,
        records=records,
        output_path=output_path,
        details_path=details_path,
        model_dir=args.model_dir,
        test_path=test_path,
        device=args.device,
        max_concurrency=args.eval_max_concurrency,
        engine=engine,
        config=config,
    )
    dump_json(output_path, summary)
    print(
        "rewardbench2_score={:.4f} non_ties_accuracy={:.4f}".format(
            summary["svr"]["summary"]["reward_bench_2_score"],
            summary["svr"]["summary"]["non_ties_accuracy"],
        ),
        flush=True,
    )
    print(f"Saved RewardBench 2 summary to {output_path}", flush=True)
    print(f"Saved RewardBench 2 details to {details_path}", flush=True)


if __name__ == "__main__":
    main()
