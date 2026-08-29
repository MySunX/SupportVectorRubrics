from __future__ import annotations

import copy
import random
from typing import Any, Sequence

from svr.schema import PreferenceExample, RubricItem
from svr.utils import (
    ALLOWED_FACETS,
    ALLOWED_IMPORTANCE,
    clean_text_block,
    extract_prompt_from_any,
    iter_json_records,
    maybe_parse_json_string,
    maybe_get_first,
    normalize_text_signature,
    normalize_whitespace,
    parse_rubric_text,
    stable_hash,
)


PROMPT_FIELD_CANDIDATES = (
    "prompt",
    "context",
    "conversation",
    "conversations",
    "request_conv",
    "conversation_history",
    "messages",
    "instruction",
    "question",
)

CHOSEN_FIELD_CANDIDATES = (
    "chosen",
    "chosen_response",
    "chosen_answer",
    "preferred_response",
    "response_chosen",
)

REJECTED_FIELD_CANDIDATES = (
    "rejected",
    "rejected_response",
    "rejected_answer",
    "dispreferred_response",
    "response_rejected",
)

DEFAULT_SELF_RUBRIC_FIELD_CANDIDATES = (
    "prompt_wise_rubrics",
    "self_generated_rubrics",
    "self_rubrics",
)

DEFAULT_REFERENCE_RUBRIC_FIELD_CANDIDATES = (
    "reference_rubrics",
    "oracle_rubrics",
    "benchmark_rubrics",
)

DIFFICULTY_FIELD_CANDIDATES = (
    "prompt_wise_difficulty_analysis",
    "difficulty_analysis",
)

CANDIDATE_RESPONSE_FIELD_CANDIDATES = (
    "candidate_responses",
    "response_samples",
    "sampled_responses",
    "responses",
)


def _normalize_facet(value: Any) -> str:
    if not isinstance(value, str):
        return "correctness"
    normalized = normalize_whitespace(value).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "readability": "coherence",
        "organization": "coherence",
        "clarity": "coherence",
        "completeness": "coverage",
        "detail": "coverage",
        "details": "coverage",
        "tone": "style",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in ALLOWED_FACETS else "correctness"


def _normalize_importance(value: Any) -> str:
    if not isinstance(value, str):
        return "major"
    normalized = normalize_whitespace(value).lower()
    return normalized if normalized in ALLOWED_IMPORTANCE else "major"


def _normalize_rubric_item(item: Any, source: str) -> RubricItem | None:
    if isinstance(item, str):
        text = parse_rubric_text(item)
        if not text:
            return None
        return RubricItem(text=text, source=source)

    if not isinstance(item, dict):
        return None

    text = parse_rubric_text(
        maybe_get_first(item, ("rubric", "text", "criterion", "requirement"))
    )
    if not text:
        return None

    grounding = parse_rubric_text(
        maybe_get_first(item, ("grounding", "rationale", "reason", "note"))
    )
    return RubricItem(
        text=text,
        facet=_normalize_facet(item.get("facet")),
        importance=_normalize_importance(item.get("importance")),
        source=normalize_whitespace(str(item.get("source") or source)).lower() or source,
        grounding=grounding,
        anchor_id=item.get("anchor_id") if isinstance(item.get("anchor_id"), str) else None,
        metadata={
            key: value
            for key, value in item.items()
            if key
            not in {
                "rubric",
                "text",
                "criterion",
                "requirement",
                "grounding",
                "rationale",
                "reason",
                "note",
                "facet",
                "importance",
                "source",
                "anchor_id",
            }
        },
    )


def _parse_rubrics_from_record(
    record: dict[str, Any], field_names: Sequence[str], source: str
) -> list[RubricItem]:
    for field_name in field_names:
        if field_name not in record:
            continue

        payload = maybe_parse_json_string(record[field_name])
        if isinstance(payload, dict) and "prompt_wise_rubrics" in payload:
            payload = payload["prompt_wise_rubrics"]
        if isinstance(payload, dict):
            payload = [payload]

        if not isinstance(payload, list):
            continue

        rubrics: list[RubricItem] = []
        for item in payload:
            normalized = _normalize_rubric_item(item, source=source)
            if normalized is not None:
                rubrics.append(normalized)

        deduped: dict[str, RubricItem] = {}
        for rubric in rubrics:
            signature = normalize_text_signature(rubric.text)
            if not signature or signature in deduped:
                continue
            deduped[signature] = rubric
        return list(deduped.values())

    return []


def _extract_prompt(record: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    prompt_value = maybe_get_first(record, PROMPT_FIELD_CANDIDATES)
    if prompt_value is None:
        raise ValueError(
            f"Missing prompt/conversation field. candidates={PROMPT_FIELD_CANDIDATES}"
        )
    return extract_prompt_from_any(prompt_value)


def _extract_chosen_rejected(record: dict[str, Any]) -> tuple[str, str]:
    chosen = maybe_get_first(record, CHOSEN_FIELD_CANDIDATES)
    rejected = maybe_get_first(record, REJECTED_FIELD_CANDIDATES)
    if isinstance(chosen, str) and isinstance(rejected, str):
        return clean_text_block(chosen), clean_text_block(rejected)

    response1 = record.get("response1")
    response2 = record.get("response2")
    overall_preference = record.get("overall_preference")
    if (
        isinstance(response1, str)
        and isinstance(response2, str)
        and isinstance(overall_preference, (int, float))
    ):
        if overall_preference < 0:
            return clean_text_block(response1), clean_text_block(response2)
        if overall_preference > 0:
            return clean_text_block(response2), clean_text_block(response1)
        raise ValueError("overall_preference is neutral/tie (0); skip this record")

    response_a = record.get("response_a")
    response_b = record.get("response_b")
    chosen_candidate = record.get("chosen_candidate")
    if (
        isinstance(response_a, str)
        and isinstance(response_b, str)
        and isinstance(chosen_candidate, str)
    ):
        normalized = chosen_candidate.strip().lower()
        if normalized == "a":
            return clean_text_block(response_a), clean_text_block(response_b)
        if normalized == "b":
            return clean_text_block(response_b), clean_text_block(response_a)

    raise ValueError(
        "Unable to extract chosen/rejected responses. "
        f"direct_fields={CHOSEN_FIELD_CANDIDATES}/{REJECTED_FIELD_CANDIDATES}"
    )


def _extract_candidate_responses(record: dict[str, Any]) -> list[str]:
    candidate_values: list[str] = []
    response1 = record.get("response1")
    response2 = record.get("response2")
    overall_preference = record.get("overall_preference")
    if (
        isinstance(response1, str)
        and isinstance(response2, str)
        and isinstance(overall_preference, (int, float))
    ):
        if overall_preference < 0:
            candidate_values.append(clean_text_block(response2))
        elif overall_preference > 0:
            candidate_values.append(clean_text_block(response1))
    nested_sources = [record]
    for container_key in ("meta", "meta_info", "ground_truth"):
        value = record.get(container_key)
        if isinstance(value, dict):
            nested_sources.append(value)

    for source in nested_sources:
        for field_name in CANDIDATE_RESPONSE_FIELD_CANDIDATES:
            if field_name not in source:
                continue
            payload = maybe_parse_json_string(source.get(field_name))
            if isinstance(payload, dict):
                payload = [payload]
            if not isinstance(payload, list):
                continue
            for item in payload:
                if isinstance(item, str) and item.strip():
                    candidate_values.append(clean_text_block(item))
                    continue
                if isinstance(item, dict):
                    for key in ("response", "response_text", "text", "content", "value"):
                        value = item.get(key)
                        if isinstance(value, str) and value.strip():
                            candidate_values.append(clean_text_block(value))
                            break

    deduped: list[str] = []
    seen: set[str] = set()
    for value in candidate_values:
        if not value:
            continue
        signature = normalize_text_signature(value)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(value)
    return deduped


def load_preference_examples(
    *,
    paths: Sequence[str],
    self_rubric_fields: Sequence[str] = DEFAULT_SELF_RUBRIC_FIELD_CANDIDATES,
    reference_rubric_fields: Sequence[str] = (),
    limit: int | None = None,
) -> list[PreferenceExample]:
    examples: list[PreferenceExample] = []
    for idx, raw_record in enumerate(iter_json_records(paths)):
        if limit is not None and len(examples) >= limit:
            break

        record = copy.deepcopy(raw_record)
        try:
            prompt_text, prompt_messages = _extract_prompt(record)
            chosen_response, rejected_response = _extract_chosen_rejected(record)
        except ValueError as exc:
            if "neutral/tie" in str(exc):
                continue
            raise
        self_rubrics = _parse_rubrics_from_record(
            record, self_rubric_fields, source="self_generated"
        )
        reference_rubrics = _parse_rubrics_from_record(
            record, reference_rubric_fields, source="reference"
        )
        candidate_responses = _extract_candidate_responses(record)
        difficulty_analysis = maybe_get_first(record, DIFFICULTY_FIELD_CANDIDATES)
        difficulty_analysis = (
            maybe_parse_json_string(difficulty_analysis)
            if difficulty_analysis is not None
            else None
        )

        example_id = None
        candidate_id = maybe_get_first(record, ("example_id", "id", "uid", "uuid"))
        if isinstance(candidate_id, str) and candidate_id.strip():
            example_id = candidate_id.strip()
        if example_id is None:
            example_id = stable_hash(
                f"{prompt_text}\n<chosen>\n{chosen_response}\n<rejected>\n{rejected_response}",
                prefix="ex_",
            )

        examples.append(
            PreferenceExample(
                example_id=example_id,
                prompt_text=prompt_text,
                prompt_messages=prompt_messages,
                chosen_response=chosen_response,
                rejected_response=rejected_response,
                candidate_responses=candidate_responses,
                self_rubrics=self_rubrics,
                reference_rubrics=reference_rubrics,
                difficulty_analysis=(
                difficulty_analysis if isinstance(difficulty_analysis, dict) else None
                ),
                meta={
                    "record_idx": idx,
                    "overall_preference": record.get("overall_preference"),
                    "domain": record.get("domain"),
                    "language": record.get("language"),
                },
                raw_record=record,
            )
        )

    return examples


def split_train_dev(
    examples: Sequence[PreferenceExample],
    *,
    dev_ratio: float,
    seed: int,
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    if not 0.0 < dev_ratio < 1.0:
        raise ValueError(f"dev_ratio must be in (0, 1), got {dev_ratio}")
    if len(examples) < 2:
        raise ValueError("Need at least two examples to split train/dev")

    shuffled = list(examples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    dev_count = max(1, int(round(len(shuffled) * dev_ratio)))
    dev_examples = shuffled[:dev_count]
    train_examples = shuffled[dev_count:]
    if not train_examples:
        raise ValueError("dev split consumed all training examples")
    return train_examples, dev_examples
