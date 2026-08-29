from __future__ import annotations

from dataclasses import dataclass
import gzip
import html
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


ALLOWED_IMPORTANCE = {"critical", "major", "minor"}
ALLOWED_FACETS = {
    "correctness",
    "format",
    "coverage",
    "grounding",
    "tool_use",
    "coherence",
    "safety",
    "style",
    "language",
    "conciseness",
    "reasoning",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "use",
    "using",
    "with",
    "your",
}

TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def stable_hash(value: str, prefix: str = "") -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}" if prefix else digest


def public_path(path: str | os.PathLike[str] | None) -> str | None:
    if path is None:
        return None
    text = str(path)
    if not text:
        return text
    path_obj = Path(text)
    if not path_obj.is_absolute():
        return text
    try:
        return str(path_obj.relative_to(Path.cwd()))
    except ValueError:
        return path_obj.name


@dataclass(frozen=True)
class OrientedPreferencePair:
    response_a: str
    response_b: str
    gold_side: str


def orient_preference_pair(
    *,
    example_id: str,
    positive_response: str,
    negative_response: str,
    salt: str = "eval",
) -> OrientedPreferencePair:
    key = "\n".join([salt, str(example_id), positive_response, negative_response])
    put_positive_in_b = int(stable_hash(key), 16) % 2 == 1
    if put_positive_in_b:
        return OrientedPreferencePair(
            response_a=negative_response,
            response_b=positive_response,
            gold_side="B",
        )
    return OrientedPreferencePair(
        response_a=positive_response,
        response_b=negative_response,
        gold_side="A",
    )


def gold_aligned_margin(weighted_margin: float, gold_side: str) -> float:
    margin = float(weighted_margin)
    return -margin if gold_side == "B" else margin


def preferred_position(
    preferred_side: str,
    *,
    gold_side: str,
    positive_label: str = "chosen",
    negative_label: str = "rejected",
) -> str:
    if preferred_side == "tie":
        return "tie"
    if preferred_side == gold_side:
        return positive_label
    if preferred_side in {"A", "B"}:
        return negative_label
    return "unknown"


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        payload = value.to_dict()
        return dict(payload) if isinstance(payload, dict) else {}
    return dict(value) if isinstance(value, dict) else {}


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def semantic_rubric_comparison_payload(
    comparison: Any,
    *,
    gold_side: str,
    positive_label: str = "chosen",
    negative_label: str = "rejected",
) -> dict[str, Any]:
    raw = _as_dict(comparison)
    score_a = _maybe_float(raw.get("score_a"))
    score_b = _maybe_float(raw.get("score_b"))
    raw_delta = _maybe_float(raw.get("delta"))
    if gold_side == "B":
        positive_score = score_b
        negative_score = score_a
        delta_vs_positive = -raw_delta if raw_delta is not None else None
    else:
        positive_score = score_a
        negative_score = score_b
        delta_vs_positive = raw_delta

    payload = {
        "rubric_id": raw.get("rubric_id"),
        "rubric_text": raw.get("rubric_text"),
        "facet": raw.get("facet"),
        "importance": raw.get("importance"),
        "weight": raw.get("weight"),
        f"score_{positive_label}": positive_score,
        f"score_{negative_label}": negative_score,
        f"delta_vs_{positive_label}": delta_vs_positive,
        "better_position": preferred_position(
            str(raw.get("better") or ""),
            gold_side=gold_side,
            positive_label=positive_label,
            negative_label=negative_label,
        ),
    }
    return {key: value for key, value in payload.items() if value is not None}


def semantic_prediction_payload(
    prediction: Any,
    *,
    gold_side: str,
    positive_label: str = "chosen",
    negative_label: str = "rejected",
    margin_key: str = "margin_vs_chosen",
) -> dict[str, Any]:
    return {
        "preferred_position": preferred_position(
            str(getattr(prediction, "preferred_side", "")),
            gold_side=gold_side,
            positive_label=positive_label,
            negative_label=negative_label,
        ),
        margin_key: gold_aligned_margin(
            float(getattr(prediction, "weighted_margin", 0.0)),
            gold_side,
        ),
        "decision_source": getattr(prediction, "decision_source", None),
        "selected_rubrics": [
            dict(item)
            for item in (getattr(prediction, "selected_rubrics", None) or [])
            if isinstance(item, dict)
        ],
        "rubric_comparisons": [
            semantic_rubric_comparison_payload(
                item,
                gold_side=gold_side,
                positive_label=positive_label,
                negative_label=negative_label,
            )
            for item in (getattr(prediction, "rubric_comparisons", None) or [])
        ],
    }


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def maybe_parse_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def canonicalize_cache_model_name(model_name: str | None) -> str:
    normalized = normalize_whitespace(str(model_name or ""))
    lowered = normalized.lower()
    if lowered == "gpt-oss-120b":
        return "gpt-oss-120b"
    return normalized


def cache_compatible_model_names(model_name: str | None) -> tuple[str, ...]:
    normalized = normalize_whitespace(str(model_name or ""))
    if not normalized:
        return ()
    canonical = canonicalize_cache_model_name(normalized)
    if canonical != "gpt-oss-120b":
        return (normalized,)

    compatible_names = [canonical, normalized]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in compatible_names:
        value = normalize_whitespace(item)
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return tuple(deduped)


def clean_text_block(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def normalize_text_signature(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return normalize_whitespace(lowered)


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def importance_weight(importance: str) -> float:
    return {"critical": 3.0, "major": 2.0, "minor": 1.0}.get(importance, 2.0)


def dump_json(path: str, payload: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_json_records(paths: Sequence[str]) -> Iterator[dict[str, Any]]:
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        if path.is_dir():
            child_paths = sorted(
                child
                for child in path.iterdir()
                if child.is_file()
                and (
                    child.suffix in {".json", ".jsonl"}
                    or child.name.endswith(".jsonl.gz")
                )
            )
            for child in child_paths:
                yield from iter_json_records([str(child)])
            continue

        if path.name.endswith(".jsonl.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Failed to parse jsonl.gz record at {path}:{line_no}"
                        ) from exc
                    if not isinstance(payload, dict):
                        raise ValueError(
                            f"Expected dict record at {path}:{line_no}, got {type(payload)}"
                        )
                    yield payload
            continue

        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Failed to parse jsonl record at {path}:{line_no}"
                        ) from exc
                    if not isinstance(payload, dict):
                        raise ValueError(
                            f"Expected dict record at {path}:{line_no}, got {type(payload)}"
                        )
                    yield payload
            continue

        if path.suffix == ".json":
            payload = load_json(str(path))
            if isinstance(payload, dict):
                yield payload
                continue
            if isinstance(payload, list):
                for idx, item in enumerate(payload):
                    if not isinstance(item, dict):
                        raise ValueError(
                            f"Expected dict record at {path}[{idx}], got {type(item)}"
                        )
                    yield item
                continue
            raise ValueError(f"Unsupported top-level JSON payload in {path}: {type(payload)}")

        raise ValueError(f"Unsupported input path suffix: {path}")


def parse_rubric_text(value: Any) -> str:
    if isinstance(value, str):
        return normalize_whitespace(value)
    return ""


def maybe_get_first(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key not in mapping:
            continue
        value = mapping[key]
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        return value
    return None


def _message_role(value: Any) -> str:
    role = normalize_whitespace(str(value or "user")).lower()
    if role in {"assistant", "model", "bot"}:
        return "assistant"
    if role in {"system", "developer"}:
        return "system"
    return "user"


def _message_content(message: dict[str, Any]) -> str:
    for key in ("content", "text", "value"):
        value = message.get(key)
        if isinstance(value, str):
            return clean_text_block(value)
        if isinstance(value, list):
            parts: list[str] = []
            for part in value:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    text = part.get("text") or part.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            if parts:
                return clean_text_block("\n".join(parts))
    return ""


def _format_prompt_messages(messages: Sequence[dict[str, str]]) -> str:
    return clean_text_block(
        "\n".join(f"{item['role']}: {item['content']}" for item in messages)
    )


def _normalize_prompt_messages(items: Sequence[Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, str):
            content = clean_text_block(item)
            role = "user"
        elif isinstance(item, dict):
            content = _message_content(item)
            role = _message_role(item.get("role") or item.get("from") or item.get("speaker"))
        else:
            continue
        if content:
            messages.append({"role": role, "content": content})
    return messages


def extract_prompt_from_any(value: Any) -> tuple[str, list[dict[str, str]]]:
    value = maybe_parse_json_string(value)
    if isinstance(value, list):
        prompt_messages = _normalize_prompt_messages(value)
        if not prompt_messages:
            raise ValueError("Unable to extract prompt messages from empty list")
        prompt_text = _format_prompt_messages(prompt_messages)
        return prompt_text, prompt_messages

    if isinstance(value, str):
        prompt_text = clean_text_block(value)
        return prompt_text, [{"role": "user", "content": prompt_text}]

    if isinstance(value, dict):
        if "prompt" in value:
            return extract_prompt_from_any(value["prompt"])
        if "conversation" in value:
            return extract_prompt_from_any(value["conversation"])
        if "messages" in value:
            return extract_prompt_from_any(value["messages"])
        content = _message_content(value)
        if content:
            role = _message_role(value.get("role") or value.get("from") or value.get("speaker"))
            return content, [{"role": role, "content": content}]

    raise ValueError(f"Unable to extract prompt from value of type {type(value)}")
