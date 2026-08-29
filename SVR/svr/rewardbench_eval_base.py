from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

import numpy as np

from svr.inference import NoOpRubricRewriter, SVRInferenceEngine
from svr.rubricbench_eval import (
    _ascore_pair_strict_rubricbench,
    _score_pair_strict_rubricbench,
)
from svr.schema import PairwisePrediction, PreferenceExample
from svr.utils import (
    clean_text_block,
    dump_json,
    ensure_dir,
    extract_prompt_from_any,
    public_path,
    semantic_prediction_payload,
)

DEFAULT_EVAL_LLM_REASONING_EFFORT = "medium"
DEFAULT_EVAL_JUDGE_RUBRIC_CHUNK_SIZE = 3
DEFAULT_EVAL_SCORE_MAX_TOKENS = 8192
DEFAULT_EVAL_MAX_CONCURRENCY = 64


@dataclass
class RewardBenchItem:
    benchmark: str
    example_id: str
    prompt_text: str
    prompt_messages: list[dict[str, str]]
    chosen_responses: list[str]
    rejected_responses: list[str]
    subset: str
    meta: dict[str, Any] = field(default_factory=dict)
    raw_record: dict[str, Any] | None = None


@dataclass(frozen=True)
class StrictPairEvalConfig:
    top_k: int = 6
    required_rubrics: int = 6
    score_retry_times: int = 5
    max_top_k: int | None = None
    selection_pool_size: int = 18
    max_same_facet: int = 2
    diversity_similarity_threshold: float = 0.92
    diversity_lambda: float = 0.35
    low_evidence_max_effective_rubrics: int = 2


def add_shared_eval_args(parser: argparse.ArgumentParser) -> None:
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
        help=(
            "Maximum retries per single-rubric recovery when the initial full top_k "
            "judge returns partial coverage."
        ),
    )
    parser.add_argument(
        "--max-top-k",
        type=int,
        default=None,
        help="Increase top_k up to this value when rubric evidence is still tied.",
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
        help=(
            "When the selector gives at most this many non-zero-weight rubrics, "
            "fall back to rubric-vote/direct-compare aggregation."
        ),
    )
    parser.add_argument(
        "--eval-llm-reasoning-effort",
        default=DEFAULT_EVAL_LLM_REASONING_EFFORT,
        help="Override real-LLM reasoning effort during eval.",
    )
    parser.add_argument(
        "--eval-judge-rubric-chunk-size",
        type=int,
        default=DEFAULT_EVAL_JUDGE_RUBRIC_CHUNK_SIZE,
        help="Override scorer chunk size during eval.",
    )
    parser.add_argument(
        "--eval-score-max-tokens",
        type=int,
        default=DEFAULT_EVAL_SCORE_MAX_TOKENS,
        help="Override pairwise judge max tokens during eval.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress-log-interval", type=int, default=64)
    parser.add_argument(
        "--eval-max-concurrency",
        type=int,
        default=DEFAULT_EVAL_MAX_CONCURRENCY,
        help="Concurrent examples for eval.",
    )
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--details-path", default=None)
    parser.add_argument(
        "--disable-rubric-rewrite",
        action="store_true",
        help="Override model inference config and skip eval-time rubric rewriting.",
    )


def strict_eval_config_from_args(args: argparse.Namespace) -> StrictPairEvalConfig:
    return StrictPairEvalConfig(
        top_k=int(args.top_k),
        required_rubrics=int(args.required_rubrics),
        score_retry_times=int(args.score_retry_times),
        max_top_k=args.max_top_k,
        selection_pool_size=int(args.selection_pool_size),
        max_same_facet=int(args.max_same_facet),
        diversity_similarity_threshold=float(args.diversity_similarity_threshold),
        diversity_lambda=float(args.diversity_lambda),
        low_evidence_max_effective_rubrics=int(args.low_evidence_max_effective_rubrics),
    )


def subset_key(value: Any) -> str:
    if not isinstance(value, str):
        return "unknown"
    normalized = " ".join(value.strip().split()).lower()
    return normalized or "unknown"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return json_safe(value.tolist())
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def normalize_text(value: Any) -> str:
    value = json_safe(value)
    if value is None:
        return ""
    if isinstance(value, str):
        return clean_text_block(value)
    return clean_text_block(str(value))


def normalize_text_list(value: Any) -> list[str]:
    value = json_safe(value)
    if value is None:
        return []
    if isinstance(value, str):
        text = normalize_text(value)
        return [text] if text else []
    if not isinstance(value, list):
        raise ValueError(f"Expected list-like completion payload, got {type(value)}")
    results: list[str] = []
    for item in value:
        text = normalize_text(item)
        if text:
            results.append(text)
    return results


def resolve_input_path(
    raw_path: str | None,
    *,
    default_path: str | None = None,
    preferred_file_names: Sequence[str],
) -> str:
    if raw_path is None and default_path is None:
        raise ValueError("An input path is required.")
    candidate = Path(raw_path or default_path or "")
    if candidate.is_file():
        return str(candidate)
    if not candidate.exists():
        raise FileNotFoundError(f"Input path does not exist: {candidate}")
    if not candidate.is_dir():
        raise ValueError(f"Expected file or directory path, got: {candidate}")

    for preferred_name in preferred_file_names:
        matches = sorted(candidate.rglob(preferred_name))
        if matches:
            return str(matches[0])

    parquet_files = sorted(candidate.rglob("*.parquet"))
    if len(parquet_files) == 1:
        return str(parquet_files[0])
    if parquet_files:
        raise ValueError(
            "Found multiple parquet files under {}. Please pass an explicit file path.".format(
                candidate
            )
        )
    raise FileNotFoundError(f"No parquet files found under: {candidate}")


def build_identity_payload(item: RewardBenchItem) -> dict[str, Any]:
    payload = {
        "benchmark": item.benchmark,
        "example_id": item.example_id,
        "subset": item.subset,
        "record_idx": item.meta.get("record_idx"),
        "source_path": public_path(item.meta.get("source_path")),
    }
    payload.update({key: value for key, value in item.meta.items() if key not in payload})
    return payload


def build_svr_payload(
    prediction: PairwisePrediction,
    *,
    gold_side: str,
    positive_label: str = "chosen",
    negative_label: str = "rejected",
    margin_key: str = "margin_vs_chosen",
) -> dict[str, Any]:
    return semantic_prediction_payload(
        prediction,
        gold_side=gold_side,
        positive_label=positive_label,
        negative_label=negative_label,
        margin_key=margin_key,
    )


def summarize_simple_pairwise(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
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
        "total_items": len(records),
        "usable_items": usable_items,
        "skipped_items": len(records) - usable_items,
        "correct_items": correct_items,
        "incorrect_items": usable_items - correct_items,
        "accuracy": correct_items / usable_items if usable_items else 0.0,
        "tie_rate": tie_items / usable_items if usable_items else 0.0,
        "avg_margin": sum(margins) / usable_items if usable_items else 0.0,
    }


def summarize_best_of_n(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    usable_records = [item for item in records if not item.get("skipped")]
    usable_items = len(usable_records)
    correct_items = sum(int(bool(item.get("correct"))) for item in usable_records)
    pair_total = sum(int(item.get("pair_total") or 0) for item in usable_records)
    pair_correct = sum(int(item.get("pair_correct_count") or 0) for item in usable_records)
    min_margins = [
        float(item["min_margin"])
        for item in usable_records
        if isinstance(item.get("min_margin"), (int, float))
    ]
    return {
        "total_items": len(records),
        "usable_items": usable_items,
        "skipped_items": len(records) - usable_items,
        "correct_items": correct_items,
        "incorrect_items": usable_items - correct_items,
        "accuracy": correct_items / usable_items if usable_items else 0.0,
        "pair_total": pair_total,
        "pair_correct": pair_correct,
        "pair_accuracy": pair_correct / pair_total if pair_total else 0.0,
        "avg_min_margin": sum(min_margins) / usable_items if usable_items else 0.0,
    }


def group_summary(
    records: Sequence[dict[str, Any]],
    key_name: str,
    summarizer: Callable[[Sequence[dict[str, Any]]], dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = str(record.get(key_name) or "unknown")
        grouped.setdefault(key, []).append(record)
    return {
        key: summarizer(group_records)
        for key, group_records in sorted(grouped.items())
    }


def write_jsonl(path: str, records: Sequence[dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(json_safe(record), ensure_ascii=False))
            file_obj.write("\n")


def build_eval_engine(
    *,
    model_dir: str,
    device: str,
    eval_llm_reasoning_effort: str = DEFAULT_EVAL_LLM_REASONING_EFFORT,
    eval_judge_rubric_chunk_size: int = DEFAULT_EVAL_JUDGE_RUBRIC_CHUNK_SIZE,
    eval_score_max_tokens: int = DEFAULT_EVAL_SCORE_MAX_TOKENS,
    eval_max_concurrency: int = DEFAULT_EVAL_MAX_CONCURRENCY,
    disable_rubric_rewrite: bool = False,
) -> SVRInferenceEngine:
    engine = SVRInferenceEngine(
        model_dir=model_dir,
        device=device,
    )
    runner = getattr(getattr(engine, "scorer", None), "config", None)
    runner = getattr(runner, "runner", None)
    if runner is not None:
        if eval_llm_reasoning_effort:
            runner.config.reasoning_effort = str(eval_llm_reasoning_effort)
            engine.inference_config["llm_reasoning_effort"] = str(
                eval_llm_reasoning_effort
            )
        scorer_config = getattr(engine.scorer, "config", None)
        if scorer_config is not None:
            scorer_config.chunk_size = max(1, int(eval_judge_rubric_chunk_size))
            scorer_config.judge_max_tokens = max(512, int(eval_score_max_tokens))
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
    if disable_rubric_rewrite:
        engine.inference_config["rewrite_selected_rubrics"] = False
        engine.rewriter = NoOpRubricRewriter()
        print("[SVR] eval override: disable rubric rewrite", flush=True)
    if eval_max_concurrency > 1:
        print(
            f"[SVR] eval concurrency enabled: max_concurrency={eval_max_concurrency}",
            flush=True,
        )
    return engine


def build_inference_summary(
    *,
    engine: SVRInferenceEngine,
    config: StrictPairEvalConfig,
    device: str,
    max_concurrency: int,
) -> dict[str, Any]:
    return {
        "top_k": config.top_k,
        "required_rubrics": config.required_rubrics,
        "score_retry_times": config.score_retry_times,
        "max_top_k": config.max_top_k,
        "selection_pool_size": config.selection_pool_size,
        "max_same_facet": config.max_same_facet,
        "diversity_similarity_threshold": config.diversity_similarity_threshold,
        "diversity_lambda": config.diversity_lambda,
        "low_evidence_max_effective_rubrics": config.low_evidence_max_effective_rubrics,
        "device": device,
        "eval_max_concurrency": max_concurrency,
        "bank_size": len(engine.bank),
        "inference_config": dict(engine.inference_config),
    }


def strict_score_pair(
    *,
    example_id: str,
    prompt_text: str,
    prompt_messages: list[dict[str, str]],
    response_a: str,
    response_b: str,
    engine: SVRInferenceEngine,
    config: StrictPairEvalConfig,
    difficulty_analysis: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    raw_record: dict[str, Any] | None = None,
) -> PairwisePrediction:
    example = PreferenceExample(
        example_id=example_id,
        prompt_text=prompt_text,
        prompt_messages=prompt_messages,
        chosen_response=response_a,
        rejected_response=response_b,
        difficulty_analysis=difficulty_analysis,
        meta=dict(meta or {}),
        raw_record=raw_record,
    )
    return _score_pair_strict_rubricbench(
        example=example,
        engine=engine,
        initial_top_k=config.top_k,
        required_rubrics=config.required_rubrics,
        score_retry_times=config.score_retry_times,
        max_top_k=config.max_top_k,
        selection_pool_size=config.selection_pool_size,
        max_same_facet=config.max_same_facet,
        diversity_similarity_threshold=config.diversity_similarity_threshold,
        diversity_lambda=config.diversity_lambda,
        low_evidence_max_effective_rubrics=config.low_evidence_max_effective_rubrics,
    )


async def astrict_score_pair(
    *,
    example_id: str,
    prompt_text: str,
    prompt_messages: list[dict[str, str]],
    response_a: str,
    response_b: str,
    engine: SVRInferenceEngine,
    config: StrictPairEvalConfig,
    difficulty_analysis: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    raw_record: dict[str, Any] | None = None,
) -> PairwisePrediction:
    example = PreferenceExample(
        example_id=example_id,
        prompt_text=prompt_text,
        prompt_messages=prompt_messages,
        chosen_response=response_a,
        rejected_response=response_b,
        difficulty_analysis=difficulty_analysis,
        meta=dict(meta or {}),
        raw_record=raw_record,
    )
    return await _ascore_pair_strict_rubricbench(
        example=example,
        engine=engine,
        initial_top_k=config.top_k,
        required_rubrics=config.required_rubrics,
        score_retry_times=config.score_retry_times,
        max_top_k=config.max_top_k,
        selection_pool_size=config.selection_pool_size,
        max_same_facet=config.max_same_facet,
        diversity_similarity_threshold=config.diversity_similarity_threshold,
        diversity_lambda=config.diversity_lambda,
        low_evidence_max_effective_rubrics=config.low_evidence_max_effective_rubrics,
    )


def evaluate_loop_sync(
    *,
    items: Sequence[RewardBenchItem],
    fn: Callable[[RewardBenchItem], dict[str, Any]],
    progress_label: str,
    progress_log_interval: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    total = len(items)
    for idx, item in enumerate(items, start=1):
        records.append(fn(item))
        if progress_log_interval > 0 and (idx % progress_log_interval == 0 or idx == total):
            print(f"[SVR] {progress_label}: {idx}/{total}", flush=True)
    return records


async def evaluate_loop_async(
    *,
    items: Sequence[RewardBenchItem],
    fn: Callable[[RewardBenchItem], Awaitable[dict[str, Any]]],
    progress_label: str,
    progress_log_interval: int,
    max_concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
    records: list[dict[str, Any] | None] = [None] * len(items)

    async def process(idx: int, item: RewardBenchItem) -> tuple[int, dict[str, Any]]:
        async with semaphore:
            return idx, await fn(item)

    tasks = [asyncio.create_task(process(idx, item)) for idx, item in enumerate(items)]
    completed = 0
    total = len(tasks)
    for task in asyncio.as_completed(tasks):
        idx, record = await task
        records[idx] = record
        completed += 1
        if progress_log_interval > 0 and (
            completed % progress_log_interval == 0 or completed == total
        ):
            print(f"[SVR] {progress_label}: {completed}/{total}", flush=True)
    return [record for record in records if record is not None]


def run_eval_loop(
    *,
    items: Sequence[RewardBenchItem],
    sync_fn: Callable[[RewardBenchItem], dict[str, Any]],
    async_fn: Callable[[RewardBenchItem], Awaitable[dict[str, Any]]] | None,
    progress_label: str,
    progress_log_interval: int,
    max_concurrency: int,
    use_async: bool,
) -> list[dict[str, Any]]:
    if use_async and async_fn is not None and max_concurrency > 1:
        return asyncio.run(
            evaluate_loop_async(
                items=items,
                fn=async_fn,
                progress_label=progress_label,
                progress_log_interval=progress_log_interval,
                max_concurrency=max_concurrency,
            )
        )
    return evaluate_loop_sync(
        items=items,
        fn=sync_fn,
        progress_label=progress_label,
        progress_log_interval=progress_log_interval,
    )


__all__ = [
    "DEFAULT_EVAL_LLM_REASONING_EFFORT",
    "DEFAULT_EVAL_JUDGE_RUBRIC_CHUNK_SIZE",
    "DEFAULT_EVAL_MAX_CONCURRENCY",
    "DEFAULT_EVAL_SCORE_MAX_TOKENS",
    "RewardBenchItem",
    "StrictPairEvalConfig",
    "add_shared_eval_args",
    "astrict_score_pair",
    "build_eval_engine",
    "build_identity_payload",
    "build_inference_summary",
    "build_svr_payload",
    "dump_json",
    "evaluate_loop_async",
    "evaluate_loop_sync",
    "extract_prompt_from_any",
    "group_summary",
    "json_safe",
    "normalize_text",
    "normalize_text_list",
    "resolve_input_path",
    "run_eval_loop",
    "strict_eval_config_from_args",
    "strict_score_pair",
    "subset_key",
    "summarize_best_of_n",
    "summarize_simple_pairwise",
    "write_jsonl",
]
