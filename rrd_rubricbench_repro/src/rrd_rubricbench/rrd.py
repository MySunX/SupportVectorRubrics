from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import sys
from typing import Any, Sequence

from .models import BenchmarkExample, ExampleResult, RubricCandidate, RubricComparison, SampledResponse
from .openai_runner import OpenAIChatRunner
from .prompts import (
    build_conflict_prompt,
    build_decomposition_prompt,
    build_direct_pairwise_prompt,
    build_initial_rubric_prompt,
    build_overlap_prompt,
    build_rubric_judge_prompt,
)
from .sampling import sample_responses, sample_responses_from_models
from .utils import (
    bool_from_yes_no,
    dedupe_preserve_order,
    extract_first_json_object,
    extract_tagged_items,
    extract_yes_no_evaluation,
    normalize_rubric_text,
    whitened_uniform_weights,
)

PAPER_SAMPLE_MODELS = ("gpt-4o", "gemini-2.5-pro")
PAPER_SAMPLE_COUNT_PER_MODEL = 4
PAPER_SAMPLE_TEMPERATURE = 0.7
PAPER_SAMPLE_TOP_P = 0.95
PAPER_STRONG_REFERENCE_MODEL = "gpt-4o"
PAPER_WEAK_REFERENCE_MODEL = "qwen3-8b"
PAPER_REFERENCE_TEMPERATURE = 0.7
PAPER_REFERENCE_TOP_P = 0.95
PAPER_DECOMPOSITION_TRIGGER_YES_COUNT = 2
PAPER_MAX_DECOMPOSITION_DEPTH = 3
PAPER_TERMINATION_THRESHOLD = 15
PAPER_WEIGHT_MODE = "wu"
PAPER_EVALUATION_MODE = "aggregated"


class RRDPipeline:
    def __init__(
        self,
        *,
        runner: OpenAIChatRunner,
    ):
        self.runner = runner
        self.sample_models = list(PAPER_SAMPLE_MODELS)
        self.sample_count_per_model = PAPER_SAMPLE_COUNT_PER_MODEL
        self.sample_temperature = PAPER_SAMPLE_TEMPERATURE
        self.sample_top_p = PAPER_SAMPLE_TOP_P
        self.sample_reasoning_effort = None
        self.strong_reference_model = PAPER_STRONG_REFERENCE_MODEL
        self.weak_reference_model = PAPER_WEAK_REFERENCE_MODEL
        self.reference_temperature = PAPER_REFERENCE_TEMPERATURE
        self.reference_top_p = PAPER_REFERENCE_TOP_P
        self.reference_reasoning_effort = None
        self.decomposition_trigger_yes_count = PAPER_DECOMPOSITION_TRIGGER_YES_COUNT
        self.max_decomposition_depth = PAPER_MAX_DECOMPOSITION_DEPTH
        self.termination_threshold = PAPER_TERMINATION_THRESHOLD
        self.weight_mode = PAPER_WEIGHT_MODE
        self.evaluation_mode = PAPER_EVALUATION_MODE

        self.rubric_generation_max_tokens = 4096
        self.decomposition_max_tokens = 2048
        self.rubric_judge_max_tokens = 256
        self.direct_compare_max_tokens = 512
        self.sample_judge_max_concurrency = 64
        self.final_judge_max_concurrency = 64
        self._reference_response_cache: dict[str, tuple[SampledResponse, SampledResponse]] = {}

    def sample_example(self, example: BenchmarkExample) -> list[SampledResponse]:
        return sample_responses_from_models(
            runner=self.runner,
            case_id=example.case_id,
            prompt_messages=example.prompt_messages,
            models=self.sample_models,
            count_per_model=self.sample_count_per_model,
            temperature=self.sample_temperature,
            top_p=self.sample_top_p,
            reasoning_effort=self.sample_reasoning_effort,
            max_tokens=self.runner.max_tokens,
        )

    def build_and_iterate_rubrics(
        self,
        example: BenchmarkExample,
        sampled: Sequence[SampledResponse],
    ) -> dict[str, Any]:
        sampled_texts = [s.text for s in sampled]
        initial_prompt = build_initial_rubric_prompt(
            prompt_messages=example.prompt_messages,
            responses=sampled_texts,
        )
        raw, _ = self.runner.complete_text(
            namespace=f"{example.case_id}/initial",
            prompt=initial_prompt,
            max_tokens=self.rubric_generation_max_tokens,
        )
        initial_rubrics = dedupe_preserve_order(extract_tagged_items(raw))
        queue = [RubricCandidate(text=r, depth=0, source="initial") for r in initial_rubrics]

        accepted: list[RubricCandidate] = []
        seen: set[str] = set()
        rejected = 0
        decomp_calls = 0
        samplewise_rubric_checks = 0
        sample_rubric_judge_calls = 0
        zero_yes_rejects = 0
        misaligned_rejects = 0
        overlap_rejects = 0
        conflict_rejects = 0
        decomposition_failures = 0
        trace_steps: list[dict[str, Any]] = []

        while queue and rejected < self.termination_threshold:
            cand = queue.pop(0)
            norm = normalize_rubric_text(cand.text).lower()
            if not norm or norm in seen:
                rejected += 1
                trace_steps.append(
                    {
                        "action": "skip_duplicate_or_empty",
                        "rubric": cand.text,
                        "depth": cand.depth,
                        "source": cand.source,
                    }
                )
                continue
            seen.add(norm)

            sample_eval = self._judge_rubric_on_samples(example, cand.text, sampled)
            samplewise_rubric_checks += len(sampled)
            sample_rubric_judge_calls += int(sample_eval.get("judge_calls", len(sampled)))
            cand.sample_vote_rate = sample_eval["vote_rate"]
            cand.sample_yes_count = sample_eval["yes_count"]
            cand.sample_votes = sample_eval["votes"]
            cand.group_vote_rates = sample_eval["group_vote_rates"]
            cand.meta.update(
                {
                    "sample_vote_rate": cand.sample_vote_rate,
                    "sample_yes_count": cand.sample_yes_count,
                    "sample_votes": cand.sample_votes,
                    "sample_groups": cand.group_vote_rates,
                }
            )

            step_record: dict[str, Any] = {
                "rubric": cand.text,
                "depth": cand.depth,
                "source": cand.source,
                "sample_vote_rate": cand.sample_vote_rate,
                "sample_yes_count": cand.sample_yes_count,
                "sample_votes": cand.sample_votes,
                "group_vote_rates": cand.group_vote_rates,
                "decision": None,
                "children": [],
                "sample_judge_calls": sample_eval.get("judge_calls", len(sampled)),
            }

            if sample_eval["yes_count"] <= 0:
                rejected += 1
                zero_yes_rejects += 1
                step_record["decision"] = "reject_zero_yes"
                trace_steps.append(step_record)
                continue

            if (
                sample_eval["yes_count"] > self.decomposition_trigger_yes_count
                and cand.depth < self.max_decomposition_depth
            ):
                satisfied_sample_texts = [
                    sample.text for sample, vote in zip(sampled, sample_eval["votes"]) if vote
                ]
                new_rubrics = self._decompose_rubric(
                    example=example,
                    cand=cand,
                    accepted=accepted,
                    sampled_responses=satisfied_sample_texts or sampled_texts,
                )
                decomp_calls += 1
                if new_rubrics:
                    step_record["decision"] = "decompose"
                    step_record["children"] = new_rubrics
                    trace_steps.append(step_record)
                    self._enqueue_rubrics(
                        queue,
                        new_rubrics,
                        depth=cand.depth + 1,
                        source="decomposition",
                        parent=cand.text,
                    )
                    continue
                rejected += 1
                decomposition_failures += 1
                step_record["decision"] = "reject_decomposition_failure"
                trace_steps.append(step_record)
                continue

            if not self._passes_directionality_guardrail(example, cand.text):
                rejected += 1
                misaligned_rejects += 1
                step_record["decision"] = "reject_directionality"
                trace_steps.append(step_record)
                continue

            filter_reason = self._overlap_or_conflict_reason(
                cand.text,
                [r.text for r in accepted],
                namespace_prefix=f"{example.case_id}/filter",
            )
            if filter_reason is not None:
                rejected += 1
                if filter_reason == "overlap":
                    overlap_rejects += 1
                else:
                    conflict_rejects += 1
                step_record["decision"] = f"reject_{filter_reason}"
                trace_steps.append(step_record)
                continue

            accepted.append(cand)
            step_record["decision"] = "accept"
            trace_steps.append(step_record)

        return {
            "accepted_rubrics": accepted,
            "stats": {
                "initial_rubrics": initial_rubrics,
                "accepted_count": len(accepted),
                "rejected_count": rejected,
                "decomposition_calls": decomp_calls,
                "max_decomposition_depth": self.max_decomposition_depth,
                "decomposition_failures": decomposition_failures,
                "samplewise_rubric_checks": samplewise_rubric_checks,
                "sample_rubric_judge_calls": sample_rubric_judge_calls,
                "zero_yes_rejects": zero_yes_rejects,
                "misalignment_rejects": misaligned_rejects,
                "overlap_rejects": overlap_rejects,
                "conflict_rejects": conflict_rejects,
                "terminated_by_threshold": rejected >= self.termination_threshold,
            },
            "trace": {
                "initial_prompt": initial_prompt,
                "initial_rubric_output": raw,
                "initial_rubrics": initial_rubrics,
                "steps": trace_steps,
            },
        }

    def run_example(self, example: BenchmarkExample) -> ExampleResult:
        sampled = self.sample_example(example)
        build_result = self.build_and_iterate_rubrics(example, sampled)
        accepted = build_result["accepted_rubrics"]
        build_stats = build_result["stats"]
        eval_result = self.evaluate_example(example, sampled, accepted)
        predicted = eval_result["predicted_candidate"]
        decision_source = eval_result["decision_source"]
        weighted_margin = eval_result["weighted_margin"]
        comparisons = eval_result["comparisons"]
        weights = eval_result["weights"]

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
                    "model": self.runner.model,
                    "sample_models": self.sample_models,
                    "sample_count_per_model": self.sample_count_per_model,
                    "temperature": self.sample_temperature,
                    "top_p": self.sample_top_p,
                    "reasoning_effort": self.sample_reasoning_effort,
                },
                "rubric_build": build_result["trace"],
                "evaluation": {
                    "evaluation_mode": self.evaluation_mode,
                    "weight_mode": self.weight_mode,
                    "decision_source": decision_source,
                    "weighted_margin": weighted_margin,
                    "weights": weights,
                },
            },
            stats={
                **build_stats,
                "decision_source": decision_source,
                "weights_mode": self.weight_mode,
                "evaluation_mode": self.evaluation_mode,
                "sample_count": len(sampled),
                "sample_model": self.runner.model,
                "sample_models": self.sample_models,
                "sample_count_per_model": self.sample_count_per_model,
                "sample_temperature": self.sample_temperature,
                "sample_top_p": self.sample_top_p,
            },
        )

    def evaluate_example(
        self,
        example: BenchmarkExample,
        sampled: Sequence[SampledResponse],
        accepted: Sequence[RubricCandidate],
    ) -> dict[str, Any]:
        votes_matrix = self._sample_major_votes_matrix(accepted, len(sampled))
        weights = self._compute_weights([r.text for r in accepted], votes_matrix=votes_matrix)
        if len(weights) != len(accepted):
            weights = [1.0] * len(accepted)
        for rubric, weight in zip(accepted, weights):
            rubric.weight = float(weight)

        predicted, decision_source, weighted_margin, comparisons = self._aggregated_judge(
            example,
            list(accepted),
            weights,
        )

        return {
            "predicted_candidate": predicted,
            "decision_source": decision_source,
            "weighted_margin": weighted_margin,
            "comparisons": comparisons,
            "weights": weights,
            "sample_count": len(sampled),
        }

    @staticmethod
    def _sample_major_votes_matrix(
        accepted: Sequence[RubricCandidate],
        sample_count: int,
    ) -> list[list[bool]]:
        """
        Return votes as rows = sampled responses, cols = accepted rubrics.

        Rubric candidates store votes per rubric, so this transposes the
        rubric-major shape before WU weighting consumes the matrix.
        """
        if not accepted or sample_count <= 0:
            return []

        rubric_major: list[list[bool]] = []
        for rubric in accepted:
            raw_votes = rubric.sample_votes if rubric.sample_votes else (rubric.meta.get("sample_votes") or [])
            votes = [bool(vote) for vote in raw_votes]
            if len(votes) != sample_count:
                return []
            rubric_major.append(votes)

        return [list(sample_votes) for sample_votes in zip(*rubric_major)]

    def _enqueue_rubrics(
        self,
        queue: list[RubricCandidate],
        rubrics: Sequence[str],
        *,
        depth: int,
        source: str,
        parent: str = "",
    ) -> None:
        for text in rubrics:
            queue.append(
                RubricCandidate(
                    text=text,
                    depth=depth,
                    source=source,
                    parent=parent,
                )
            )

    def _judge_rubric_on_samples(
        self,
        example: BenchmarkExample,
        rubric: str,
        sampled: Sequence[SampledResponse],
    ) -> dict[str, Any]:
        if not sampled:
            return {
                "votes": [],
                "yes_count": 0,
                "vote_rate": 0.0,
                "group_vote_rates": {},
                "judge_calls": 0,
            }
        return self._judge_rubric_on_samples_per_sample(example, rubric, sampled)

    def _judge_rubric_on_samples_per_sample(
        self,
        example: BenchmarkExample,
        rubric: str,
        sampled: Sequence[SampledResponse],
    ) -> dict[str, Any]:
        def _judge_one(idx_sample: tuple[int, SampledResponse]) -> tuple[int, bool, str]:
            idx, sample = idx_sample
            val = self._judge_single_response_rubric(
                namespace=f"{example.case_id}/sample-judge/{idx}",
                response=sample.text,
                rubric=rubric,
            )
            return idx, bool(val), sample.group_tag

        items = list(enumerate(sampled))
        max_workers = max(1, min(len(items), self.sample_judge_max_concurrency))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            judged = list(executor.map(_judge_one, items))

        return self._sample_judge_result_from_judged(judged, judge_calls=len(items))

    @staticmethod
    def _sample_judge_result_from_judged(
        judged: list[tuple[int, bool, str]],
        *,
        judge_calls: int,
    ) -> dict[str, Any]:
        judged.sort(key=lambda item: item[0])
        votes = [vote for _, vote, _ in judged]
        group_votes: dict[str, list[bool]] = {}
        for _, vote, group_tag in judged:
            group_votes.setdefault(group_tag, []).append(vote)
        group_vote_rates = {
            group: (sum(vals) / len(vals) if vals else 0.0)
            for group, vals in group_votes.items()
        }
        yes_count = sum(1 for v in votes if v)
        return {
            "votes": votes,
            "yes_count": yes_count,
            "vote_rate": yes_count / max(1, len(votes)),
            "group_vote_rates": group_vote_rates,
            "judge_calls": judge_calls,
        }

    def _passes_directionality_guardrail(self, example: BenchmarkExample, rubric: str) -> bool:
        try:
            strong_response, weak_response = self._reference_responses(example)
        except Exception as exc:  # noqa: BLE001
            if "inappropriate content" not in str(exc).lower():
                raise
            print(
                "[RRD] weak/strong reference blocked by content safety; "
                f"skipping directionality guardrail for case={example.case_id}",
                file=sys.stderr,
                flush=True,
            )
            return True

        def _judge_one(sample: SampledResponse) -> bool:
            return self._judge_single_response_rubric(
                namespace=f"{example.case_id}/misalignment/{sample.group_tag}",
                response=sample.text,
                rubric=rubric,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            strong_future = executor.submit(_judge_one, strong_response)
            weak_future = executor.submit(_judge_one, weak_response)
            strong_yes = strong_future.result()
            weak_yes = weak_future.result()
        return not (weak_yes and not strong_yes)

    def _reference_responses(self, example: BenchmarkExample) -> tuple[SampledResponse, SampledResponse]:
        cached = self._reference_response_cache.get(example.case_id)
        if cached is not None:
            return cached
        prompt_messages = example.prompt_messages

        def _sample_reference(kind: str, model: str) -> SampledResponse:
            return sample_responses(
                runner=self.runner,
                case_id=f"{example.case_id}/{kind}-reference",
                prompt_messages=prompt_messages,
                model=model,
                count=1,
                temperature=self.reference_temperature,
                top_p=self.reference_top_p,
                reasoning_effort=self.reference_reasoning_effort,
                max_tokens=self.runner.max_tokens,
            )[0]

        with ThreadPoolExecutor(max_workers=2) as executor:
            strong_future = executor.submit(_sample_reference, "strong", self.strong_reference_model)
            weak_future = executor.submit(_sample_reference, "weak", self.weak_reference_model)
            strong = strong_future.result()
            weak = weak_future.result()

        strong = SampledResponse(
            index=0,
            group_index=0,
            group_tag="strong",
            model=self.strong_reference_model,
            text=strong.text,
            temperature=self.reference_temperature,
            top_p=self.reference_top_p,
            reasoning_effort=self.reference_reasoning_effort,
            meta={"namespace": f"{example.case_id}/strong-reference/sample/0"},
        )
        weak = SampledResponse(
            index=1,
            group_index=1,
            group_tag="weak",
            model=self.weak_reference_model,
            text=weak.text,
            temperature=self.reference_temperature,
            top_p=self.reference_top_p,
            reasoning_effort=self.reference_reasoning_effort,
            meta={"namespace": f"{example.case_id}/weak-reference/sample/0"},
        )
        self._reference_response_cache[example.case_id] = (strong, weak)
        return strong, weak

    def _overlap_or_conflict_reason(
        self,
        rubric: str,
        existing_rubrics: list[str],
        *,
        namespace_prefix: str = "filter",
    ) -> str | None:
        if not existing_rubrics:
            return None

        def _check_overlap() -> str:
            overlap_raw, _ = self.runner.complete_text(
                namespace=f"{namespace_prefix}/overlap",
                prompt=build_overlap_prompt(existing_rubrics=existing_rubrics, new_rubric=rubric),
                max_tokens=64,
            )
            return overlap_raw

        def _check_conflict() -> str:
            conflict_raw, _ = self.runner.complete_text(
                namespace=f"{namespace_prefix}/conflict",
                prompt=build_conflict_prompt(existing_rubrics=existing_rubrics, new_rubric=rubric),
                max_tokens=64,
            )
            return conflict_raw

        with ThreadPoolExecutor(max_workers=2) as executor:
            overlap_future = executor.submit(_check_overlap)
            conflict_future = executor.submit(_check_conflict)
            overlap_raw = overlap_future.result()
            if "YES" in overlap_raw.upper():
                return "overlap"
            conflict_raw = conflict_future.result()
            if "YES" in conflict_raw.upper():
                return "conflict"
        return None

    def _decompose_rubric(
        self,
        example: BenchmarkExample,
        cand: RubricCandidate,
        accepted: list[RubricCandidate],
        sampled_responses: Sequence[str],
    ) -> list[str]:
        other_rubrics = [r.text for r in accepted]
        prompt = build_decomposition_prompt(
            prompt_messages=example.prompt_messages,
            responses=list(sampled_responses),
            current_rubric=cand.text,
            other_rubrics=other_rubrics,
        )
        raw, _ = self.runner.complete_text(
            namespace=f"{example.case_id}/decompose/{cand.depth}",
            prompt=prompt,
            max_tokens=self.decomposition_max_tokens,
        )
        rubrics = dedupe_preserve_order(extract_tagged_items(raw))
        rubrics = [r for r in rubrics if normalize_rubric_text(r)]
        return rubrics[:2]

    def _compute_weights(
        self,
        rubrics: list[str],
        *,
        votes_matrix: list[list[bool]] | None = None,
    ) -> list[float]:
        if not rubrics:
            return []
        return whitened_uniform_weights(votes_matrix or [])

    def _aggregated_judge(
        self,
        example: BenchmarkExample,
        accepted: list[RubricCandidate],
        weights: list[float],
    ) -> tuple[str, str, float, list[RubricComparison]]:
        if not accepted:
            predicted, decision_source = self._direct_compare(example)
            return predicted, decision_source, 0.0, []

        def _judge_one(item: tuple[int, RubricCandidate]) -> tuple[int, dict[str, bool]]:
            idx, rubric = item
            return idx, self._judge_rubric_pairwise(example, rubric.text, idx)

        items = list(enumerate(accepted))
        max_workers = max(1, min(len(items), self.final_judge_max_concurrency))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            judged = list(executor.map(_judge_one, items))

        judged.sort(key=lambda item: item[0])
        comparisons: list[RubricComparison] = []
        weighted_margin = 0.0
        for idx, pair_eval in judged:
            rubric = accepted[idx]
            a_score = 1.0 if pair_eval["response_a"] else 0.0
            b_score = 1.0 if pair_eval["response_b"] else 0.0
            delta = a_score - b_score
            weighted_margin += weights[idx] * delta
            comparisons.append(
                RubricComparison(
                    rubric=rubric.text,
                    weight=weights[idx],
                    response_a_yes=bool(pair_eval["response_a"]),
                    response_b_yes=bool(pair_eval["response_b"]),
                    response_a_score=a_score,
                    response_b_score=b_score,
                    delta=delta,
                    source=rubric.source,
                    meta={
                        "depth": rubric.depth,
                        "parent": rubric.parent,
                        "sample_vote_rate": rubric.sample_vote_rate,
                        "sample_yes_count": rubric.sample_yes_count,
                    },
                )
            )
        if weighted_margin > 0:
            return "a", "rrd_margin", weighted_margin, comparisons
        if weighted_margin < 0:
            return "b", "rrd_margin", weighted_margin, comparisons
        predicted, decision_source = self._direct_compare(example)
        return predicted, decision_source, weighted_margin, comparisons

    def _judge_single_response_rubric(self, *, namespace: str, response: str, rubric: str) -> bool:
        raw, _ = self.runner.complete_text(
            namespace=namespace,
            prompt=build_rubric_judge_prompt(response=response, rubric=rubric),
            max_tokens=self.rubric_judge_max_tokens,
        )
        obj = extract_first_json_object(raw)
        val = bool_from_yes_no(obj.get("evaluation")) if obj else None
        if val is None:
            val = extract_yes_no_evaluation(raw)
        return bool(val)

    def _judge_rubric_pairwise(self, example: BenchmarkExample, rubric: str, rubric_idx: int) -> dict[str, bool]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            response_a_future = executor.submit(
                self._judge_single_response_rubric,
                namespace=f"{example.case_id}/judge/{rubric_idx}/response_a",
                response=example.response_a,
                rubric=rubric,
            )
            response_b_future = executor.submit(
                self._judge_single_response_rubric,
                namespace=f"{example.case_id}/judge/{rubric_idx}/response_b",
                response=example.response_b,
                rubric=rubric,
            )
        return {
            "response_a": response_a_future.result(),
            "response_b": response_b_future.result(),
        }

    def _direct_compare(self, example: BenchmarkExample) -> tuple[str, str]:
        prompt = build_direct_pairwise_prompt(
            prompt_messages=example.prompt_messages,
            response_a=example.response_a,
            response_b=example.response_b,
        )
        raw, _ = self.runner.complete_text(
            namespace=f"{example.case_id}/direct",
            prompt=prompt,
            max_tokens=self.direct_compare_max_tokens,
        )
        obj = extract_first_json_object(raw) or {}
        pref = str(obj.get("preferred_candidate", "")).strip().upper()
        if pref == "A":
            return "a", "direct_compare"
        if pref == "B":
            return "b", "direct_compare"
        if "A" in raw.upper() and "B" not in raw.upper():
            return "a", "direct_compare_fallback"
        if "B" in raw.upper() and "A" not in raw.upper():
            return "b", "direct_compare_fallback"
        return "a", "direct_compare_fallback"
