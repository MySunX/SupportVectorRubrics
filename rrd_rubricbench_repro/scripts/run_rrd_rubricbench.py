from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rrd_rubricbench.data import load_rubricbench
from rrd_rubricbench.evaluate import summarize_results
from rrd_rubricbench.models import ExampleResult, RubricCandidate, RubricComparison, SampledResponse
from rrd_rubricbench.openai_runner import OpenAIChatRunner
from rrd_rubricbench.rrd import (
    PAPER_DECOMPOSITION_TRIGGER_YES_COUNT,
    PAPER_EVALUATION_MODE,
    PAPER_MAX_DECOMPOSITION_DEPTH,
    PAPER_SAMPLE_COUNT_PER_MODEL,
    PAPER_SAMPLE_MODELS,
    PAPER_SAMPLE_TEMPERATURE,
    PAPER_SAMPLE_TOP_P,
    PAPER_STRONG_REFERENCE_MODEL,
    PAPER_TERMINATION_THRESHOLD,
    PAPER_WEAK_REFERENCE_MODEL,
    PAPER_WEIGHT_MODE,
    RRDPipeline,
)
from rrd_rubricbench.utils import ensure_dir, write_json, write_jsonl

T = TypeVar("T")


CHECKPOINT_SCHEMA_VERSION = 3
DEFAULT_LLM_MODEL = "gpt-oss-120b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RRD on RubricBench")
    parser.add_argument("--benchmark-path", default="./data/rubricbench.json")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument(
        "--model",
        "--llm-model",
        dest="llm_model",
        default=DEFAULT_LLM_MODEL,
        help="LLM name used as rubric proposer and final judge.",
    )
    parser.add_argument("--base-url", default=os.getenv("RRD_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY") or os.getenv("RRD_OPENAI_API_KEY"))
    parser.add_argument(
        "--reasoning-effort",
        "--llm-reasoning-effort",
        dest="llm_reasoning_effort",
        default="high",
        help="Reasoning effort forwarded to the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--temperature",
        "--llm-temperature",
        dest="llm_temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--top-p",
        "--llm-top-p",
        dest="llm_top_p",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--max-tokens",
        "--llm-max-tokens",
        dest="llm_max_tokens",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--timeout",
        "--llm-request-timeout-sec",
        dest="llm_request_timeout_sec",
        type=float,
        default=900.0,
    )
    parser.add_argument(
        "--max-retries",
        "--llm-retry-times",
        dest="llm_retry_times",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--llm-max-concurrency",
        type=int,
        default=64,
        help="Max concurrent LLM calls for sampling/judging/rewrite steps.",
    )
    parser.add_argument(
        "--example-max-concurrency",
        type=int,
        default=4,
        help="How many benchmark examples to process in parallel per stage.",
    )
    parser.add_argument("--cache-dir", default="./.cache")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for resumable per-example checkpoints. Defaults to OUTPUT_DIR/checkpoints.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    return parser.parse_args()


def _safe_checkpoint_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    slug = slug.strip("._")
    return (slug or "case")[:80]


def _checkpoint_key(index: int, case_id: str) -> str:
    return f"{index:06d}_{_safe_checkpoint_slug(case_id)}"


def _checkpoint_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "data_ordering": "chosen_response_as_a_v1",
        "benchmark_path": str(args.benchmark_path),
        "start_index": args.start_index,
        "limit": args.limit,
        "llm_model": args.llm_model,
        "llm_reasoning_effort": args.llm_reasoning_effort,
        "llm_temperature": args.llm_temperature,
        "llm_top_p": args.llm_top_p,
        "llm_max_tokens": args.llm_max_tokens,
        "rrd_paper_config": {
            "sample_models": list(PAPER_SAMPLE_MODELS),
            "sample_count_per_model": PAPER_SAMPLE_COUNT_PER_MODEL,
            "sample_temperature": PAPER_SAMPLE_TEMPERATURE,
            "sample_top_p": PAPER_SAMPLE_TOP_P,
            "strong_reference_model": PAPER_STRONG_REFERENCE_MODEL,
            "weak_reference_model": PAPER_WEAK_REFERENCE_MODEL,
            "decomposition_trigger_yes_count": PAPER_DECOMPOSITION_TRIGGER_YES_COUNT,
            "max_decomposition_depth": PAPER_MAX_DECOMPOSITION_DEPTH,
            "termination_threshold": PAPER_TERMINATION_THRESHOLD,
            "weight_mode": PAPER_WEIGHT_MODE,
            "evaluation_mode": PAPER_EVALUATION_MODE,
            "rubric_judge_prompt": "single_response_yes_no",
        },
    }


def _checkpoint_signature(config: dict[str, Any]) -> str:
    raw = json.dumps(config, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _write_json_atomic(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_sampled(sampled: Sequence[SampledResponse]) -> list[dict[str, Any]]:
    return [asdict(item) for item in sampled]


def _load_sampled(payload: Any) -> list[SampledResponse]:
    return [SampledResponse(**item) for item in payload]


def _dump_build_result(build_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted_rubrics": [asdict(item) for item in build_result["accepted_rubrics"]],
        "stats": build_result["stats"],
        "trace": build_result["trace"],
    }


def _load_build_result(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted_rubrics": [RubricCandidate(**item) for item in payload["accepted_rubrics"]],
        "stats": payload["stats"],
        "trace": payload["trace"],
    }


def _dump_eval_result(eval_result: dict[str, Any]) -> dict[str, Any]:
    return {
        **eval_result,
        "comparisons": [asdict(item) for item in eval_result["comparisons"]],
    }


def _load_eval_result(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **payload,
        "comparisons": [RubricComparison(**item) for item in payload["comparisons"]],
    }


def _dump_result(result: ExampleResult) -> dict[str, Any]:
    return asdict(result)


def _load_result(payload: dict[str, Any]) -> ExampleResult:
    return ExampleResult(**payload)


def _run_checkpointed_stage(
    *,
    stage_name: str,
    examples: Sequence[Any],
    keys: Sequence[str],
    stage_dir: Path,
    max_workers: int,
    compute: Callable[[int], T],
    dump: Callable[[T], Any],
    load: Callable[[Any], T],
) -> list[T]:
    values: list[T | None] = [None] * len(examples)
    missing: list[int] = []
    for idx, key in enumerate(keys):
        path = stage_dir / f"{key}.json"
        if path.is_file():
            values[idx] = load(_read_json(path))
        else:
            missing.append(idx)

    print(
        f"[RRD] {stage_name}: loaded={len(examples) - len(missing)} "
        f"missing={len(missing)} checkpoint_dir={stage_dir}",
        flush=True,
    )

    if missing:
        ensure_dir(stage_dir)

        def _compute_and_save(idx: int) -> tuple[int, T]:
            print(
                f"[RRD] {stage_name}: start {idx + 1}/{len(examples)} key={keys[idx]}",
                flush=True,
            )
            value = compute(idx)
            _write_json_atomic(stage_dir / f"{keys[idx]}.json", dump(value))
            print(
                f"[RRD] {stage_name}: saved {idx + 1}/{len(examples)} key={keys[idx]}",
                flush=True,
            )
            return idx, value

        workers = max(1, min(int(max_workers), len(missing)))
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_compute_and_save, idx) for idx in missing]
            for future in as_completed(futures):
                idx, value = future.result()
                values[idx] = value
                completed += 1
                if completed == len(missing) or completed % 10 == 0:
                    print(
                        f"[RRD] {stage_name}: completed {completed}/{len(missing)} newly computed",
                        flush=True,
                    )

    return [value for value in values if value is not None]


def _make_result(
    *,
    example,
    sampled: Sequence[SampledResponse],
    build_result: dict[str, Any],
    eval_result: dict[str, Any],
    runner: OpenAIChatRunner,
    pipeline: RRDPipeline,
) -> ExampleResult:
    accepted = build_result["accepted_rubrics"]
    weights = eval_result["weights"]
    for rubric, weight in zip(accepted, weights):
        rubric.weight = float(weight)

    predicted = eval_result["predicted_candidate"]
    decision_source = eval_result["decision_source"]
    weighted_margin = eval_result["weighted_margin"]
    comparisons = eval_result["comparisons"]
    return ExampleResult(
        case_id=example.case_id,
        domain=example.domain,
        gold_candidate=example.gold_candidate,
        predicted_candidate=predicted,
        weighted_margin=weighted_margin,
        correct=(predicted == example.gold_candidate),
        rubric_count=len(accepted),
        sampled_responses=[asdict(s) for s in sampled],
        accepted_rubrics=[asdict(r) for r in accepted],
        rubric_comparisons=[asdict(c) for c in comparisons],
        trace={
            "sampling": {
                "sample_count": len(sampled),
                "model": runner.model,
                "temperature": pipeline.sample_temperature,
                "top_p": pipeline.sample_top_p,
                "reasoning_effort": pipeline.sample_reasoning_effort,
                "sample_models": pipeline.sample_models,
                "sample_count_per_model": pipeline.sample_count_per_model,
            },
            "rubric_build": build_result["trace"],
            "evaluation": {
                "evaluation_mode": pipeline.evaluation_mode,
                "weight_mode": pipeline.weight_mode,
                "decision_source": decision_source,
                "weighted_margin": weighted_margin,
                "weights": weights,
            },
        },
        stats={
            **build_result["stats"],
            "decision_source": decision_source,
            "weights_mode": pipeline.weight_mode,
            "evaluation_mode": pipeline.evaluation_mode,
            "sample_count": len(sampled),
            "sample_model": runner.model,
            "sample_models": pipeline.sample_models,
            "sample_count_per_model": pipeline.sample_count_per_model,
            "sample_temperature": pipeline.sample_temperature,
            "sample_top_p": pipeline.sample_top_p,
        },
    )


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    examples = load_rubricbench(args.benchmark_path)
    examples = examples[args.start_index :]
    if args.limit is not None:
        examples = examples[: args.limit]

    runner = OpenAIChatRunner(
        model=args.llm_model,
        api_key=args.api_key,
        base_url=args.base_url,
        reasoning_effort=args.llm_reasoning_effort,
        temperature=args.llm_temperature,
        top_p=args.llm_top_p,
        max_tokens=args.llm_max_tokens,
        request_timeout_sec=args.llm_request_timeout_sec,
        retry_times=args.llm_retry_times,
        max_concurrency=args.llm_max_concurrency,
        cache_dir=args.cache_dir,
    )
    pipeline = RRDPipeline(
        runner=runner,
    )
    pipeline.sample_judge_max_concurrency = args.llm_max_concurrency
    pipeline.final_judge_max_concurrency = args.llm_max_concurrency

    config = _checkpoint_config(args)
    signature = _checkpoint_signature(config)
    checkpoint_root = ensure_dir(args.checkpoint_dir or (output_dir / "checkpoints"))
    run_checkpoint_dir = ensure_dir(checkpoint_root / signature)
    _write_json_atomic(
        run_checkpoint_dir / "config.json",
        {
            "signature": signature,
            "config": config,
            "example_count": len(examples),
        },
    )
    keys = [_checkpoint_key(idx, example.case_id or f"example_{idx}") for idx, example in enumerate(examples)]
    max_workers = max(1, min(int(args.example_max_concurrency), len(examples)))
    print(
        f"[RRD] checkpoint signature={signature} root={run_checkpoint_dir}",
        flush=True,
    )

    sampled_list = _run_checkpointed_stage(
        stage_name="stage 1/4 sampling",
        examples=examples,
        keys=keys,
        stage_dir=run_checkpoint_dir / "samples",
        max_workers=max_workers,
        compute=lambda idx: pipeline.sample_example(examples[idx]),
        dump=_dump_sampled,
        load=_load_sampled,
    )

    build_list = _run_checkpointed_stage(
        stage_name="stage 2/4 rubric build",
        examples=examples,
        keys=keys,
        stage_dir=run_checkpoint_dir / "build",
        max_workers=max_workers,
        compute=lambda idx: pipeline.build_and_iterate_rubrics(examples[idx], sampled_list[idx]),
        dump=_dump_build_result,
        load=_load_build_result,
    )

    eval_list = _run_checkpointed_stage(
        stage_name="stage 3/4 evaluation",
        examples=examples,
        keys=keys,
        stage_dir=run_checkpoint_dir / "eval",
        max_workers=max_workers,
        compute=lambda idx: pipeline.evaluate_example(
            examples[idx],
            sampled_list[idx],
            build_list[idx]["accepted_rubrics"],
        ),
        dump=_dump_eval_result,
        load=_load_eval_result,
    )

    results = _run_checkpointed_stage(
        stage_name="stage 4/4 result assembly",
        examples=examples,
        keys=keys,
        stage_dir=run_checkpoint_dir / "results",
        max_workers=max_workers,
        compute=lambda idx: _make_result(
            example=examples[idx],
            sampled=sampled_list[idx],
            build_result=build_list[idx],
            eval_result=eval_list[idx],
            runner=runner,
            pipeline=pipeline,
        ),
        dump=_dump_result,
        load=_load_result,
    )

    summary = summarize_results(results)
    write_jsonl(output_dir / "predictions.jsonl", [_dump_result(r) for r in results])
    write_json(output_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
