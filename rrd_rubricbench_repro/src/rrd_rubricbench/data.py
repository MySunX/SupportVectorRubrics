from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import BenchmarkExample
from .utils import normalize_whitespace


def _prompt_messages_from_record(record: dict[str, Any]) -> list[dict[str, str]]:
    prompt = record.get("prompt")
    if isinstance(prompt, list) and prompt:
        out = []
        for msg in prompt:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "user"))
            content = normalize_whitespace(str(msg.get("content", "")))
            if content:
                out.append({"role": role, "content": content})
        if out:
            return out
    instruction = normalize_whitespace(str(record.get("instruction") or record.get("prompt_text") or ""))
    return [{"role": "user", "content": instruction}]


def _prompt_text_from_messages(messages: list[dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        parts.append(f"{msg['role']}: {msg['content']}")
    return "\n".join(parts).strip()


def _gold_candidate_from_record(record: dict[str, Any]) -> str:
    gold = str(record.get("preferred_candidate") or record.get("chosen_candidate") or "").strip().lower()
    if gold:
        return gold
    label = record.get("label")
    return "a" if int(label) == 0 else "b"


def _chosen_rejected_from_record(record: dict[str, Any]) -> tuple[str, str]:
    response_a = str(record.get("response_a") or "")
    response_b = str(record.get("response_b") or "")
    gold = _gold_candidate_from_record(record)
    if gold == "a":
        return response_a, response_b
    if gold == "b":
        return response_b, response_a
    raise ValueError(f"Unsupported preferred candidate: {gold!r}")


def load_rubricbench(path: str | Path) -> list[BenchmarkExample]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".jsonl":
        raw = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
            raw = raw["data"]
    examples: list[BenchmarkExample] = []
    for record in raw:
        if not isinstance(record, dict):
            continue
        messages = _prompt_messages_from_record(record)
        prompt_text = _prompt_text_from_messages(messages)
        chosen_response, rejected_response = _chosen_rejected_from_record(record)
        examples.append(
            BenchmarkExample(
                case_id=str(record.get("case_id") or record.get("id") or ""),
                domain=str(record.get("domain") or "unknown"),
                prompt_text=prompt_text,
                prompt_messages=messages,
                response_a=chosen_response,
                response_b=rejected_response,
                gold_candidate="a",
                source=str(record.get("source") or ""),
                reference_rubrics=[str(r) for r in record.get("reference_rubrics", []) if str(r).strip()],
                raw_record=record,
            )
        )
    return examples
