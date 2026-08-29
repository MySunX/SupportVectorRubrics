from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

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
    normalize_text_list,
    resolve_input_path,
    run_eval_loop,
    strict_eval_config_from_args,
    strict_score_pair,
    subset_key,
    write_jsonl,
)
from svr.utils import gold_aligned_margin, orient_preference_pair, preferred_position, public_path

SUMMARY_FILENAME = "rmbench_eval_summary.json"
DETAIL_FILENAME = "rmbench_eval_details.jsonl"
STYLE_ORDER = ["concise", "detailed_plain", "detailed_markdown"]
RAW_TO_OFFICIAL_DOMAIN = {
    "chat": "Chat",
    "code": "Code",
    "math": "Math",
    "safety-refuse": "Safety",
    "safety-response": "Safety",
}


def _zero_matrix() -> list[list[float]]:
    return [[0.0, 0.0, 0.0] for _ in range(3)]


def _matrix_mean(matrix: Sequence[Sequence[float]]) -> float:
    values = [float(cell) for row in matrix for cell in row]
    return sum(values) / len(values) if values else 0.0


def _difficulty_metrics_from_matrix(matrix: Sequence[Sequence[float]]) -> dict[str, float]:
    diagonal = [float(matrix[idx][idx]) for idx in range(3)]
    upper = [float(matrix[row][col]) for row in range(3) for col in range(row + 1, 3)]
    lower = [float(matrix[row][col]) for row in range(3) for col in range(row)]
    return {
        "easy_accuracy": sum(lower) / len(lower) if lower else 0.0,
        "normal_accuracy": sum(diagonal) / len(diagonal) if diagonal else 0.0,
        "hard_accuracy": sum(upper) / len(upper) if upper else 0.0,
        "average_accuracy": _matrix_mean(matrix),
    }


def _official_domain(raw_domain: str) -> str:
    return RAW_TO_OFFICIAL_DOMAIN.get(str(raw_domain or "").strip(), "Other")


def _domain_matches(raw_domain: str, filters: set[str] | None) -> bool:
    if not filters:
        return True
    raw_key = subset_key(raw_domain)
    official_key = subset_key(_official_domain(raw_domain))
    return raw_key in filters or official_key in filters


def _load_rmbench_items(
    *,
    path: str,
    domain_filters: set[str] | None,
    limit: int | None,
) -> list[RewardBenchItem]:
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    if isinstance(payload, list):
        raw_records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        raw_records = payload["data"]
    else:
        raise ValueError(f"Unsupported RM-Bench payload type: {type(payload)}")

    items: list[RewardBenchItem] = []
    for record_idx, raw_record in enumerate(raw_records):
        raw_domain = str(raw_record.get("domain") or "unknown")
        if not _domain_matches(raw_domain, domain_filters):
            continue

        prompt_text, prompt_messages = extract_prompt_from_any(raw_record.get("prompt"))
        chosen_responses = normalize_text_list(raw_record.get("chosen"))
        rejected_responses = normalize_text_list(raw_record.get("rejected"))
        if len(chosen_responses) != 3 or len(rejected_responses) != 3:
            raise ValueError(
                "RM-Bench expects exactly 3 chosen and 3 rejected responses per item; "
                f"got chosen={len(chosen_responses)} rejected={len(rejected_responses)} "
                f"for record {raw_record.get('id', record_idx)}"
            )

        global_idx = len(items)
        official_domain = _official_domain(raw_domain)
        example_id = str(raw_record.get("id") if raw_record.get("id") is not None else global_idx)
        items.append(
            RewardBenchItem(
                benchmark="rm-bench",
                example_id=example_id,
                prompt_text=prompt_text,
                prompt_messages=prompt_messages,
                chosen_responses=chosen_responses,
                rejected_responses=rejected_responses,
                subset=raw_domain,
                meta={
                    "record_idx": record_idx,
                    "global_idx": global_idx,
                    "source_path": public_path(path),
                    "raw_domain": raw_domain,
                    "official_domain": official_domain,
                    "style_order": list(STYLE_ORDER),
                },
                raw_record=raw_record,
            )
        )
        if limit is not None and len(items) >= limit:
            break
    return items


def _build_item_payload(
    *,
    item: RewardBenchItem,
    accuracy_matrix: list[list[float]],
    margin_matrix: list[list[float]],
    tie_matrix: list[list[float]],
    pair_results: list[dict[str, Any]],
) -> dict[str, Any]:
    difficulty_metrics = _difficulty_metrics_from_matrix(accuracy_matrix)
    strict_prompt_pass = all(
        bool(accuracy_matrix[row][col]) for row in range(3) for col in range(3)
    )
    payload = build_identity_payload(item)
    payload.update(
        {
            "raw_domain": item.meta.get("raw_domain"),
            "official_domain": item.meta.get("official_domain"),
            "style_order": list(STYLE_ORDER),
            "skipped": False,
            "correct": strict_prompt_pass,
            "strict_prompt_pass": strict_prompt_pass,
            "pair_total": 9,
            "accuracy_matrix": accuracy_matrix,
            "margin_matrix": margin_matrix,
            "tie_matrix": tie_matrix,
            "pairwise_accuracy": difficulty_metrics["average_accuracy"],
            "pairwise_tie_rate": _matrix_mean(tie_matrix),
            **difficulty_metrics,
            "pair_results": pair_results,
            "score_method": (
                "official_rm_bench_matrix_using_strict_rubricbench_pairwise_judgments"
            ),
        }
    )
    return payload


def _score_item(*, item: RewardBenchItem, engine, config) -> dict[str, Any]:
    accuracy_matrix = _zero_matrix()
    margin_matrix = _zero_matrix()
    tie_matrix = _zero_matrix()
    pair_results: list[dict[str, Any]] = []

    for chosen_idx, chosen_response in enumerate(item.chosen_responses):
        for rejected_idx, rejected_response in enumerate(item.rejected_responses):
            pair_id = f"{item.example_id}:chosen:{chosen_idx}:rejected:{rejected_idx}"
            oriented = orient_preference_pair(
                example_id=pair_id,
                positive_response=chosen_response,
                negative_response=rejected_response,
                salt="rmbench",
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
            preferred_side = prediction.preferred_side
            is_correct = preferred_side == oriented.gold_side
            margin_vs_chosen = gold_aligned_margin(
                prediction.weighted_margin,
                oriented.gold_side,
            )
            accuracy_matrix[chosen_idx][rejected_idx] = 1.0 if is_correct else 0.0
            margin_matrix[chosen_idx][rejected_idx] = margin_vs_chosen
            tie_matrix[chosen_idx][rejected_idx] = 1.0 if preferred_side == "tie" else 0.0
            pair_results.append(
                {
                    "chosen_idx": chosen_idx,
                    "rejected_idx": rejected_idx,
                    "correct": is_correct,
                    "preferred_position": preferred_position(
                        preferred_side,
                        gold_side=oriented.gold_side,
                    ),
                    "margin_vs_chosen": margin_vs_chosen,
                    "svr": build_svr_payload(prediction, gold_side=oriented.gold_side),
                }
            )

    return _build_item_payload(
        item=item,
        accuracy_matrix=accuracy_matrix,
        margin_matrix=margin_matrix,
        tie_matrix=tie_matrix,
        pair_results=pair_results,
    )


async def _ascore_item(*, item: RewardBenchItem, engine, config) -> dict[str, Any]:
    accuracy_matrix = _zero_matrix()
    margin_matrix = _zero_matrix()
    tie_matrix = _zero_matrix()
    pair_results: list[dict[str, Any]] = []

    for chosen_idx, chosen_response in enumerate(item.chosen_responses):
        for rejected_idx, rejected_response in enumerate(item.rejected_responses):
            pair_id = f"{item.example_id}:chosen:{chosen_idx}:rejected:{rejected_idx}"
            oriented = orient_preference_pair(
                example_id=pair_id,
                positive_response=chosen_response,
                negative_response=rejected_response,
                salt="rmbench",
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
            preferred_side = prediction.preferred_side
            is_correct = preferred_side == oriented.gold_side
            margin_vs_chosen = gold_aligned_margin(
                prediction.weighted_margin,
                oriented.gold_side,
            )
            accuracy_matrix[chosen_idx][rejected_idx] = 1.0 if is_correct else 0.0
            margin_matrix[chosen_idx][rejected_idx] = margin_vs_chosen
            tie_matrix[chosen_idx][rejected_idx] = 1.0 if preferred_side == "tie" else 0.0
            pair_results.append(
                {
                    "chosen_idx": chosen_idx,
                    "rejected_idx": rejected_idx,
                    "correct": is_correct,
                    "preferred_position": preferred_position(
                        preferred_side,
                        gold_side=oriented.gold_side,
                    ),
                    "margin_vs_chosen": margin_vs_chosen,
                    "svr": build_svr_payload(prediction, gold_side=oriented.gold_side),
                }
            )

    return _build_item_payload(
        item=item,
        accuracy_matrix=accuracy_matrix,
        margin_matrix=margin_matrix,
        tie_matrix=tie_matrix,
        pair_results=pair_results,
    )


def _aggregate_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        empty_matrix = _zero_matrix()
        return {
            "total_items": 0,
            "pair_total": 0,
            "pair_correct": 0.0,
            "strict_prompt_correct": 0.0,
            "strict_prompt_accuracy": 0.0,
            "accuracy_matrix": empty_matrix,
            "margin_matrix": empty_matrix,
            "tie_matrix": empty_matrix,
            "pairwise_tie_rate": 0.0,
            **_difficulty_metrics_from_matrix(empty_matrix),
        }

    accuracy_matrix = _zero_matrix()
    margin_matrix = _zero_matrix()
    tie_matrix = _zero_matrix()
    strict_prompt_correct = 0
    pair_total = 0
    pair_correct = 0.0
    for record in records:
        strict_prompt_correct += int(bool(record.get("strict_prompt_pass")))
        pair_total += int(record.get("pair_total") or 0)
        for row in range(3):
            for col in range(3):
                acc_value = float(record["accuracy_matrix"][row][col])
                margin_value = float(record["margin_matrix"][row][col])
                tie_value = float(record["tie_matrix"][row][col])
                accuracy_matrix[row][col] += acc_value
                margin_matrix[row][col] += margin_value
                tie_matrix[row][col] += tie_value
                pair_correct += acc_value

    total_items = len(records)
    for row in range(3):
        for col in range(3):
            accuracy_matrix[row][col] /= total_items
            margin_matrix[row][col] /= total_items
            tie_matrix[row][col] /= total_items

    return {
        "total_items": total_items,
        "pair_total": pair_total,
        "pair_correct": pair_correct,
        "strict_prompt_correct": strict_prompt_correct,
        "strict_prompt_accuracy": strict_prompt_correct / total_items if total_items else 0.0,
        "accuracy_matrix": accuracy_matrix,
        "margin_matrix": margin_matrix,
        "tie_matrix": tie_matrix,
        "pairwise_tie_rate": _matrix_mean(tie_matrix),
        **_difficulty_metrics_from_matrix(accuracy_matrix),
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
    raw_domain_counts = Counter(str(item.meta.get("raw_domain") or "unknown") for item in items)
    official_domain_counts = Counter(
        str(item.meta.get("official_domain") or "Other") for item in items
    )

    by_raw_domain_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_official_domain_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_raw_domain_records[str(record.get("raw_domain") or "unknown")].append(record)
        by_official_domain_records[str(record.get("official_domain") or "Other")].append(record)

    by_raw_domain = {
        domain: _aggregate_records(domain_records)
        for domain, domain_records in sorted(by_raw_domain_records.items())
    }
    by_official_domain = {
        domain: _aggregate_records(domain_records)
        for domain, domain_records in sorted(by_official_domain_records.items())
    }

    overall = _aggregate_records(records)
    covered_domains = [
        domain
        for domain in ["Chat", "Code", "Math", "Safety"]
        if domain in by_official_domain
    ]
    domain_scores = {
        domain: float(by_official_domain[domain]["average_accuracy"]) for domain in covered_domains
    }
    difficulty_scores = {
        "easy_accuracy": (
            sum(float(by_official_domain[domain]["easy_accuracy"]) for domain in covered_domains)
            / len(covered_domains)
            if covered_domains
            else 0.0
        ),
        "normal_accuracy": (
            sum(float(by_official_domain[domain]["normal_accuracy"]) for domain in covered_domains)
            / len(covered_domains)
            if covered_domains
            else 0.0
        ),
        "hard_accuracy": (
            sum(float(by_official_domain[domain]["hard_accuracy"]) for domain in covered_domains)
            / len(covered_domains)
            if covered_domains
            else 0.0
        ),
    }
    rm_bench_score = (
        sum(domain_scores.values()) / len(domain_scores) if domain_scores else 0.0
    )

    return {
        "benchmark": "rm-bench",
        "model_dir": public_path(model_dir),
        "test_paths": [public_path(test_path)],
        "output_path": public_path(output_path),
        "details_path": public_path(details_path),
        "dataset": {
            "total_items": len(items),
            "raw_domain_counts": dict(sorted(raw_domain_counts.items())),
            "official_domain_counts": dict(sorted(official_domain_counts.items())),
            "style_order": list(STYLE_ORDER),
        },
        "inference": build_inference_summary(
            engine=engine,
            config=config,
            device=device,
            max_concurrency=max_concurrency,
        ),
        "svr": {
            "summary": {
                **overall,
                "rm_bench_score": rm_bench_score,
                "micro_pairwise_accuracy": overall["average_accuracy"],
                "domain_macro_accuracy": rm_bench_score,
                "official_domain_scores": domain_scores,
                "difficulty_macro_scores": difficulty_scores,
                "leaderboard_note": (
                    "Official RM-Bench overall is the equal-weight average of "
                    "Chat, Code, Math, and Safety domain average accuracies."
                ),
            },
            "by_official_domain": by_official_domain,
            "by_raw_domain": by_raw_domain,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SVR model on RM-Bench using RubricBench-style strict judging."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--test-path",
        required=True,
        help="RM-Bench json path or a directory containing total_dataset.json.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--domain",
        action="append",
        default=None,
        help=(
            "Repeat to keep only selected RM-Bench domains, matched case-insensitively. "
            "Supports both raw domains (e.g. safety-refuse) and official domains "
            "(e.g. Safety)."
        ),
    )
    add_shared_eval_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_path = resolve_input_path(
        args.test_path,
        preferred_file_names=["total_dataset.json"],
    )
    domain_filters = {subset_key(item) for item in args.domain} if args.domain else None
    config = strict_eval_config_from_args(args)
    print(
        f"[SVR] RM-Bench eval start: model_dir={args.model_dir} test_paths={[test_path]}",
        flush=True,
    )

    items = _load_rmbench_items(
        path=test_path,
        domain_filters=domain_filters,
        limit=args.limit,
    )
    if not items:
        raise ValueError("No RM-Bench items selected for evaluation.")
    print(
        "[SVR] loaded rm-bench items={} raw_domains={}".format(
            len(items),
            dict(sorted(Counter(str(item.meta.get("raw_domain")) for item in items).items())),
        ),
        flush=True,
    )

    engine = build_eval_engine(
        model_dir=args.model_dir,
        device=args.device,
        eval_llm_reasoning_effort=args.eval_llm_reasoning_effort,
        eval_judge_rubric_chunk_size=args.eval_judge_rubric_chunk_size,
        eval_score_max_tokens=args.eval_score_max_tokens,
        eval_max_concurrency=args.eval_max_concurrency,
        disable_rubric_rewrite=bool(args.disable_rubric_rewrite),
    )
    use_async = args.eval_max_concurrency > 1
    records = run_eval_loop(
        items=items,
        sync_fn=lambda item: _score_item(item=item, engine=engine, config=config),
        async_fn=lambda item: _ascore_item(item=item, engine=engine, config=config),
        progress_label="rm-bench progress",
        progress_log_interval=int(args.progress_log_interval),
        max_concurrency=int(args.eval_max_concurrency),
        use_async=use_async,
    )

    output_path = args.output_path or os.path.join(args.model_dir, SUMMARY_FILENAME)
    details_path = args.details_path or os.path.join(args.model_dir, DETAIL_FILENAME)
    summary = _build_summary(
        items=items,
        records=records,
        output_path=output_path,
        details_path=details_path,
        model_dir=args.model_dir,
        test_path=test_path,
        device=args.device,
        max_concurrency=int(args.eval_max_concurrency),
        engine=engine,
        config=config,
    )
    dump_json(output_path, summary)
    write_jsonl(details_path, records)
    print(f"Saved RM-Bench summary to {output_path}", flush=True)
    print(f"Saved RM-Bench details to {details_path}", flush=True)
    print(
        "rm_bench_score={:.4f} micro_pairwise_accuracy={:.4f} strict_prompt_accuracy={:.4f}".format(
            float(summary["svr"]["summary"]["rm_bench_score"]),
            float(summary["svr"]["summary"]["micro_pairwise_accuracy"]),
            float(summary["svr"]["summary"]["strict_prompt_accuracy"]),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
