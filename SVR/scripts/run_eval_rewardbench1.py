from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

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
    normalize_text,
    resolve_input_path,
    run_eval_loop,
    strict_eval_config_from_args,
    strict_score_pair,
    subset_key,
    summarize_simple_pairwise,
    write_jsonl,
)
from svr.utils import gold_aligned_margin, orient_preference_pair, preferred_position, public_path

SUMMARY_FILENAME = "rewardbench_eval_summary.json"
DETAIL_FILENAME = "rewardbench_eval_details.jsonl"
REWARD_BENCH_SECTION_EXAMPLE_COUNTS = {
    "alpacaeval-easy": 100,
    "alpacaeval-length": 95,
    "alpacaeval-hard": 95,
    "mt-bench-easy": 28,
    "mt-bench-med": 40,
    "mt-bench-hard": 37,
    "math-prm": 984,
    "refusals-dangerous": 100,
    "refusals-offensive": 100,
    "llmbar-natural": 100,
    "llmbar-adver-neighbor": 134,
    "llmbar-adver-GPTInst": 92,
    "llmbar-adver-GPTOut": 47,
    "llmbar-adver-manual": 46,
    "xstest-should-refuse": 154,
    "xstest-should-respond": 250,
    "donotanswer": 136,
    "hep-cpp": 164,
    "hep-go": 164,
    "hep-java": 164,
    "hep-js": 164,
    "hep-python": 164,
    "hep-rust": 164,
}
REWARD_BENCH_SECTION_MAPPING = {
    "Chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "Reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}


def _load_rewardbench_items(
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
        chosen = normalize_text(raw_record.get("chosen"))
        rejected = normalize_text(raw_record.get("rejected"))
        if not chosen or not rejected:
            continue
        global_idx = len(items)
        example_id = str(raw_record.get("id") if raw_record.get("id") is not None else global_idx)
        items.append(
            RewardBenchItem(
                benchmark="reward-bench",
                example_id=example_id,
                prompt_text=prompt_text,
                prompt_messages=prompt_messages,
                chosen_responses=[chosen],
                rejected_responses=[rejected],
                subset=subset,
                meta={
                    "record_idx": record_idx,
                    "global_idx": global_idx,
                    "source_path": public_path(path),
                    "chosen_model": json_safe(raw_record.get("chosen_model")),
                    "rejected_model": json_safe(raw_record.get("rejected_model")),
                },
                raw_record=json_safe(raw_record),
            )
        )
        if limit is not None and len(items) >= limit:
            break
    return items


def _score_item(*, item: RewardBenchItem, engine, config) -> dict[str, Any]:
    oriented = orient_preference_pair(
        example_id=item.example_id,
        positive_response=item.chosen_responses[0],
        negative_response=item.rejected_responses[0],
        salt="rewardbench1",
    )
    prediction = strict_score_pair(
        example_id=item.example_id,
        prompt_text=item.prompt_text,
        prompt_messages=item.prompt_messages,
        response_a=oriented.response_a,
        response_b=oriented.response_b,
        engine=engine,
        config=config,
        meta=item.meta,
        raw_record=item.raw_record,
    )
    margin_vs_chosen = gold_aligned_margin(
        prediction.weighted_margin,
        oriented.gold_side,
    )
    payload = build_identity_payload(item)
    payload.update(
        {
            "skipped": False,
            "correct": prediction.preferred_side == oriented.gold_side,
            "preferred_position": preferred_position(
                prediction.preferred_side,
                gold_side=oriented.gold_side,
            ),
            "margin_vs_chosen": margin_vs_chosen,
            "svr": build_svr_payload(prediction, gold_side=oriented.gold_side),
        }
    )
    return payload


async def _ascore_item(*, item: RewardBenchItem, engine, config) -> dict[str, Any]:
    oriented = orient_preference_pair(
        example_id=item.example_id,
        positive_response=item.chosen_responses[0],
        negative_response=item.rejected_responses[0],
        salt="rewardbench1",
    )
    prediction = await astrict_score_pair(
        example_id=item.example_id,
        prompt_text=item.prompt_text,
        prompt_messages=item.prompt_messages,
        response_a=oriented.response_a,
        response_b=oriented.response_b,
        engine=engine,
        config=config,
        meta=item.meta,
        raw_record=item.raw_record,
    )
    margin_vs_chosen = gold_aligned_margin(
        prediction.weighted_margin,
        oriented.gold_side,
    )
    payload = build_identity_payload(item)
    payload.update(
        {
            "skipped": False,
            "correct": prediction.preferred_side == oriented.gold_side,
            "preferred_position": preferred_position(
                prediction.preferred_side,
                gold_side=oriented.gold_side,
            ),
            "margin_vs_chosen": margin_vs_chosen,
            "svr": build_svr_payload(prediction, gold_side=oriented.gold_side),
        }
    )
    return payload


def _compute_section_scores(metrics: dict[str, float]) -> dict[str, float]:
    section_scores: dict[str, float] = {}
    for section, tests in REWARD_BENCH_SECTION_MAPPING.items():
        total_weighted_score = 0.0
        total_examples = 0
        for test in tests:
            if test not in metrics:
                continue
            total_weighted_score += metrics[test] * REWARD_BENCH_SECTION_EXAMPLE_COUNTS[test]
            total_examples += REWARD_BENCH_SECTION_EXAMPLE_COUNTS[test]
        section_scores[section] = (
            total_weighted_score / total_examples if total_examples else 0.0
        )
    return section_scores


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
    summary = summarize_simple_pairwise(records)
    by_subset = group_summary(records, "subset", summarize_simple_pairwise)
    subset_metrics = {
        subset: float(payload.get("accuracy") or 0.0)
        for subset, payload in by_subset.items()
    }
    section_scores = _compute_section_scores(subset_metrics)
    covered_sections = [
        section
        for section, tests in REWARD_BENCH_SECTION_MAPPING.items()
        if any(test in subset_metrics for test in tests)
    ]
    core_score = (
        sum(section_scores[section] for section in covered_sections) / len(covered_sections)
        if covered_sections
        else 0.0
    )
    return {
        "benchmark": "reward-bench",
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
            "summary": summary,
            "by_subset": by_subset,
            "section_scores": section_scores,
            "core_reward_bench_score": core_score,
            "leaderboard_note": (
                "This core score covers RewardBench local subsets only. "
                "Prior preference test sets are not included."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SVR model on RewardBench using RubricBench-style strict judging."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--test-path",
        required=True,
        help="RewardBench parquet path or a directory containing it.",
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
        preferred_file_names=["filtered-00000-of-00001.parquet"],
    )
    subset_filters = {subset_key(item) for item in args.subset} if args.subset else None
    config = strict_eval_config_from_args(args)
    print(
        "[SVR] RewardBench eval start: "
        f"model_dir={args.model_dir} test_paths={[test_path]}",
        flush=True,
    )
    items = _load_rewardbench_items(
        path=test_path,
        subset_filters=subset_filters,
        limit=args.limit,
    )
    if not items:
        raise ValueError("No RewardBench items found for the requested inputs.")
    print(
        f"[SVR] loaded reward-bench items={len(items)} subsets={sorted({item.subset for item in items})}",
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
    records = run_eval_loop(
        items=items,
        sync_fn=lambda item: _score_item(item=item, engine=engine, config=config),
        async_fn=lambda item: _ascore_item(item=item, engine=engine, config=config),
        progress_label="reward-bench progress",
        progress_log_interval=args.progress_log_interval,
        max_concurrency=args.eval_max_concurrency,
        use_async=use_async,
    )

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
        "rewardbench_accuracy={:.4f} core_score={:.4f}".format(
            summary["svr"]["summary"]["accuracy"],
            summary["svr"]["core_reward_bench_score"],
        ),
        flush=True,
    )
    print(f"Saved RewardBench summary to {output_path}", flush=True)
    print(f"Saved RewardBench details to {details_path}", flush=True)


if __name__ == "__main__":
    main()
