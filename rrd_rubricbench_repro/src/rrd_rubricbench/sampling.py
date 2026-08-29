from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from .models import SampledResponse
from .openai_runner import OpenAIChatRunner
from .prompts import build_sampling_prompt


def sample_responses(
    *,
    runner: OpenAIChatRunner,
    case_id: str,
    prompt_messages: Sequence[dict[str, str]],
    model: str,
    count: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.95,
    reasoning_effort: str | None = None,
    max_tokens: int,
) -> list[SampledResponse]:
    """Sample multiple candidate responses from one model for a single prompt."""
    prompt = build_sampling_prompt(prompt_messages=prompt_messages)
    total = max(0, int(count))
    if total == 0:
        return []

    def _sample_one(idx: int) -> SampledResponse:
        raw, _ = runner.complete_text(
            namespace=f"{case_id}/sample/{idx}",
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
        )
        return SampledResponse(
            index=idx,
            group_index=0,
            group_tag=model,
            model=model,
            text=raw.strip(),
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            meta={"namespace": f"{case_id}/sample/{idx}"},
        )

    max_workers = max(
        1,
        min(
            total,
            int(getattr(runner, "max_concurrency", total) or total),
            16,
        ),
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        sampled = list(executor.map(_sample_one, range(total)))
    return sampled


def sample_responses_from_models(
    *,
    runner: OpenAIChatRunner,
    case_id: str,
    prompt_messages: Sequence[dict[str, str]],
    models: Sequence[str],
    count_per_model: int = 4,
    temperature: float = 0.7,
    top_p: float = 0.95,
    reasoning_effort: str | None = None,
    max_tokens: int,
) -> list[SampledResponse]:
    """Sample candidate responses from multiple models for one prompt."""
    model_list = [str(model).strip() for model in models if str(model).strip()]
    if not model_list:
        return []

    def _sample_group(group_item: tuple[int, str]) -> tuple[int, str, list[SampledResponse]]:
        group_index, model = group_item
        group_samples = sample_responses(
            runner=runner,
            case_id=f"{case_id}/sample-model/{group_index}",
            prompt_messages=prompt_messages,
            model=model,
            count=count_per_model,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
        return group_index, model, group_samples

    max_workers = max(
        1,
        min(
            len(model_list),
            int(getattr(runner, "max_concurrency", len(model_list)) or len(model_list)),
        ),
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        groups = list(executor.map(_sample_group, enumerate(model_list)))

    all_sampled: list[SampledResponse] = []
    for group_index, model, group_samples in sorted(groups, key=lambda item: item[0]):
        for sample in group_samples:
            global_index = len(all_sampled)
            all_sampled.append(
                SampledResponse(
                    index=global_index,
                    group_index=group_index,
                    group_tag=model,
                    model=model,
                    text=sample.text,
                    temperature=sample.temperature,
                    top_p=sample.top_p,
                    reasoning_effort=sample.reasoning_effort,
                    meta={
                        **sample.meta,
                        "namespace": f"{case_id}/sample-model/{group_index}/sample/{sample.index}",
                        "model_local_index": sample.index,
                    },
                )
            )
    return all_sampled
