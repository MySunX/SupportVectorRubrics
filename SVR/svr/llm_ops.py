from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence

import numpy as np

from svr.rubric_quality import (
    normalize_candidate_rubric_text,
    rubric_text_quality_issue,
)
from svr.schema import (
    BankEntry,
    HardNegativeResult,
    PairwisePrediction,
    PairwiseRubricComparison,
    PreferenceExample,
    RubricItem,
)
from svr.utils import (
    canonicalize_cache_model_name,
    cache_compatible_model_names,
    dump_json,
    ensure_dir,
    importance_weight,
    load_json,
    normalize_text_signature,
    normalize_whitespace,
    stable_hash,
)


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
ALLOWED_IMPORTANCE = {"critical", "major", "minor"}


def _normalize_llm_base_url(raw_addr: str | None) -> str:
    addr = (raw_addr or "").strip()
    if not addr:
        addr = (
            os.getenv("LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
        )
    if not addr:
        raise ValueError(
            "Missing LLM endpoint. Set LLM_BASE_URL or pass --llm-base-url."
        )
    if addr.startswith("http://") or addr.startswith("https://"):
        base_url = addr.rstrip("/")
    else:
        base_url = f"http://{addr}".rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def _extract_completion_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    choice0 = choices[0]
    if isinstance(choice0, dict):
        text = choice0.get("text")
        if isinstance(text, str) and text:
            return text
        message = choice0.get("message", {})
        if isinstance(message, dict):
            content = message.get("content", "")
            if isinstance(content, str) and content:
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        item_text = item.get("text")
                        if isinstance(item_text, str) and item_text.strip():
                            parts.append(item_text.strip())
                if parts:
                    return "\n".join(parts)
            model_extra = message.get("model_extra", {}) or {}
            for key in ("reasoning_content", "text", "response", "answer"):
                value = model_extra.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return str(choice0)

def _parse_json_response(response_text: str) -> dict[str, Any]:
    try:
        import json_repair  # type: ignore

        try:
            parsed = json_repair.loads(response_text)
            if isinstance(parsed, dict):
                return parsed
            return {
                "error": (
                    "JSON parsed successfully but top-level type is not object: "
                    f"{type(parsed).__name__}"
                ),
                "parsed_type": type(parsed).__name__,
                "parsed_value_preview": str(parsed)[:500],
            }
        except Exception:
            pass
    except Exception:
        pass

    try:
        parsed = json.loads(response_text)
        if isinstance(parsed, dict):
            return parsed
        return {
            "error": (
                "JSON parsed successfully but top-level type is not object: "
                f"{type(parsed).__name__}"
            ),
            "parsed_type": type(parsed).__name__,
            "parsed_value_preview": str(parsed)[:500],
        }
    except Exception:
        pass

    import re

    json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if not json_match:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1) if json_match.groups() else json_match.group())
            if isinstance(parsed, dict):
                return parsed
            return {
                "error": (
                    "Regex-extracted JSON parsed but top-level type is not object: "
                    f"{type(parsed).__name__}"
                ),
                "parsed_type": type(parsed).__name__,
                "parsed_value_preview": str(parsed)[:500],
            }
        except Exception:
            pass
    return {"error": f"无法解析响应: {response_text[:200]}..."}


def _safe_parse_structured_response(
    *,
    parser: Any,
    raw_text: str,
) -> dict[str, Any]:
    try:
        parsed = parser.parse_response(raw_text)
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"parse_response_exception: {type(exc).__name__}: {exc}",
        }
    if not isinstance(parsed, dict):
        return {
            "error": (
                "parse_response returned non-dict value: "
                f"{type(parsed).__name__}"
            )
        }
    return parsed


def _messages_to_text(prompt_messages: Sequence[dict[str, str]]) -> str:
    return "\n".join(
        f"{item.get('role', 'user')}: {item.get('content', '')}"
        for item in prompt_messages
    ).strip()


def _build_pairwise_judge_prompt(
    *,
    prompt_messages: list[dict[str, str]],
    rubrics: list[dict[str, Any]],
    response_a: str,
    response_b: str,
    difficulty_analysis: dict[str, Any] | None,
    rubric_note: str,
) -> str:
    payload: dict[str, Any] = {
        "prompt_messages": prompt_messages,
        "rubrics": rubrics,
        "response_a": response_a,
        "response_b": response_b,
    }
    if difficulty_analysis:
        payload["difficulty_analysis"] = difficulty_analysis
    if rubric_note:
        payload["rubric_note"] = rubric_note
    return (
        "You are judging two assistant responses with prompt-specific rubrics.\n"
        "For each rubric, decide whether candidate A and candidate B pass or fail, "
        "then choose the better candidate for that rubric.\n\n"
        "Return only JSON with this schema:\n"
        "{\n"
        '  "preferred_candidate": "A",\n'
        '  "rubric_comparisons": [\n'
        "    {\n"
        '      "candidate_a_verdict": "pass",\n'
        '      "candidate_b_verdict": "fail",\n'
        '      "better": "A",\n'
        '      "rationale": "<brief reason>"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Use exactly one comparison per rubric in the given order. "
        "Use only `pass` or `fail` for verdicts and only `A` or `B` for better/preferred fields.\n\n"
        "<pairwise_input>\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        "</pairwise_input>\n"
    )


def _build_prompt_analysis_prompt(prompt_messages: list[dict[str, str]]) -> str:
    return (
        "Analyze the evaluation challenges in the following user prompt or conversation.\n"
        "Return only JSON with concise fields such as task_type, key_requirements, "
        "likely_failure_modes, and difficulty_notes.\n\n"
        "<prompt_conversation>\n"
        f"{_messages_to_text(prompt_messages)}\n"
        "</prompt_conversation>\n"
    )


def _build_prompt_rubric_prompt(
    *,
    prompt_messages: list[dict[str, str]],
    difficulty_analysis: dict[str, Any] | None,
) -> str:
    payload = {
        "prompt_messages": prompt_messages,
        "difficulty_analysis": difficulty_analysis or {},
    }
    return (
        "Propose prompt-wise evaluation rubrics for judging assistant responses.\n\n"
        "Rules:\n"
        "1. Produce 4 to 8 rubrics.\n"
        "2. Each rubric must be atomic, reusable, and independently checkable.\n"
        "3. Focus on correctness, completeness, instruction following, reasoning, "
        "format, safety, and grounding when relevant.\n"
        "4. Do not include answer keys or mention a preferred/dispreferred response.\n\n"
        "Return only JSON:\n"
        "{\n"
        '  "prompt_wise_rubrics": [\n'
        "    {\n"
        '      "facet": "correctness",\n'
        '      "importance": "major",\n'
        '      "rubric": "<one reusable rubric>",\n'
        '      "grounding": "<why this matters for the prompt>"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "<rubric_input>\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        "</rubric_input>\n"
    )


def _as_prompt_messages(
    prompt_messages: list[dict[str, str]] | None,
    prompt_text: str | None,
) -> list[dict[str, str]]:
    if isinstance(prompt_messages, list) and prompt_messages:
        return prompt_messages
    if isinstance(prompt_text, str) and prompt_text.strip():
        return [{"role": "user", "content": prompt_text}]
    raise ValueError("Missing prompt_messages/prompt_text")


def _to_score_rubric_dict(item: RubricItem | BankEntry | dict[str, Any]) -> dict[str, Any]:
    if isinstance(item, BankEntry):
        rubric_text = item.text
        importance = item.importance
        facet = item.facet
        grounding = item.grounding
        source = item.source
        point_id = item.bank_id
    elif isinstance(item, RubricItem):
        rubric_text = item.text
        importance = item.importance
        facet = item.facet
        grounding = item.grounding
        source = item.source
        point_id = item.metadata.get("point_id")
    else:
        rubric_text = str(item.get("text") or item.get("rubric") or "").strip()
        importance = str(item.get("importance") or "major").strip().lower()
        facet = str(item.get("facet") or "correctness").strip().lower()
        grounding = str(item.get("grounding") or "").strip()
        source = str(item.get("source") or "svr_bank").strip()
        point_id = item.get("point_id") or item.get("bank_id")
    return {
        "rubric": rubric_text,
        "importance": importance if importance in ALLOWED_IMPORTANCE else "major",
        "is_implicit": False,
        "facet": facet if facet in ALLOWED_FACETS else "correctness",
        "grounding": grounding,
        "source": source,
        "point_id": point_id,
    }


@dataclass
class OpenAICompatibleLLMConfig:
    model: str = "gpt-oss-120b"
    reasoning_effort: str = "high"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 4096
    retry_times: int = 25
    retry_backoff_seconds: float = 1.0
    request_timeout_sec: int = 900
    llm_base_url: str | None = None
    cache_dir: str | None = None
    max_concurrency: int = 64
    progress_log_interval: int = 64


class OpenAICompatibleCompletionRunner:
    def __init__(self, config: OpenAICompatibleLLMConfig):
        self.config = config
        self.base_url = _normalize_llm_base_url(config.llm_base_url)
        self._cache_stats: dict[str, dict[str, int]] = {}
        api_key = (
            os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or "EMPTY"
        )
        from openai import OpenAI
        from openai import AsyncOpenAI

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=float(self.config.request_timeout_sec),
        )
        self.async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=float(self.config.request_timeout_sec),
        )
        self.cache_dir = config.cache_dir
        if self.cache_dir:
            ensure_dir(self.cache_dir)
            self.error_dir = os.path.join(os.path.dirname(self.cache_dir), "llm_errors")
            ensure_dir(self.error_dir)
        else:
            self.error_dir = None
        print(
            "[SVR] LLM runner configured: "
            f"base_url={self.base_url} model={self.config.model} "
            f"reasoning_effort={self.config.reasoning_effort} "
            f"max_tokens={self.config.max_tokens} retry_times={self.config.retry_times}",
            flush=True,
        )

    def complete_text(
        self,
        *,
        prompt: str,
        namespace: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
            "temperature": self.config.temperature if temperature is None else temperature,
            "top_p": self.config.top_p if top_p is None else top_p,
            "reasoning_effort": self.config.reasoning_effort,
        }
        if extra_payload:
            payload.update(extra_payload)

        cache_path, cache_paths = self._cache_paths(namespace=namespace, payload=payload)
        for candidate_cache_path in cache_paths:
            if not os.path.isfile(candidate_cache_path):
                continue
            self._record_cache_event(namespace=namespace, event="hit")
            cached = load_json(candidate_cache_path)
            if cache_path and candidate_cache_path != cache_path:
                self._write_cache(
                    cache_path,
                    text=str(cached.get("text", "")),
                    response=cached.get("response", {}),
                )
            return str(cached.get("text", "")), cached.get("response", {})

        self._record_cache_event(namespace=namespace, event="miss")
        try:
            response_json = self._request_with_retry(payload)
            text = _extract_completion_text(response_json).strip()
        except Exception as exc:  # noqa: BLE001
            self.record_error(
                namespace=namespace,
                error_kind="request_failure",
                message=str(exc),
                prompt=prompt,
                extra={
                    "payload": payload,
                    "cache_path": cache_path,
                },
            )
            raise
        self._write_cache(cache_path, text=text, response=response_json)
        return text, response_json

    async def acomplete_text(
        self,
        *,
        prompt: str,
        namespace: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
            "temperature": self.config.temperature if temperature is None else temperature,
            "top_p": self.config.top_p if top_p is None else top_p,
            "reasoning_effort": self.config.reasoning_effort,
        }
        if extra_payload:
            payload.update(extra_payload)

        cache_path, cache_paths = self._cache_paths(namespace=namespace, payload=payload)
        for candidate_cache_path in cache_paths:
            if not os.path.isfile(candidate_cache_path):
                continue
            self._record_cache_event(namespace=namespace, event="hit")
            cached = load_json(candidate_cache_path)
            if cache_path and candidate_cache_path != cache_path:
                self._write_cache(
                    cache_path,
                    text=str(cached.get("text", "")),
                    response=cached.get("response", {}),
                )
            return str(cached.get("text", "")), cached.get("response", {})

        self._record_cache_event(namespace=namespace, event="miss")
        try:
            response_json = await self._arequest_with_retry(payload)
            text = _extract_completion_text(response_json).strip()
        except Exception as exc:  # noqa: BLE001
            self.record_error(
                namespace=namespace,
                error_kind="request_failure",
                message=str(exc),
                prompt=prompt,
                extra={
                    "payload": payload,
                    "cache_path": cache_path,
                },
            )
            raise
        self._write_cache(cache_path, text=text, response=response_json)
        return text, response_json

    def _request_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.config.retry_times + 1):
            try:
                response = self.client.chat.completions.create(
                    model=str(payload["model"]),
                    messages=[{"role": "user", "content": str(payload["prompt"])}],
                    max_tokens=int(payload["max_tokens"]),
                    temperature=float(payload["temperature"]),
                    top_p=float(payload["top_p"]),
                    reasoning_effort=payload.get("reasoning_effort"),
                )
                return response.model_dump()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.config.retry_times:
                    time.sleep(
                        min(
                            self.config.retry_backoff_seconds * (2 ** (attempt - 1)),
                            30.0,
                        )
                    )
        assert last_error is not None
        raise last_error

    async def _arequest_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.config.retry_times + 1):
            try:
                response = await self.async_client.chat.completions.create(
                    model=str(payload["model"]),
                    messages=[{"role": "user", "content": str(payload["prompt"])}],
                    max_tokens=int(payload["max_tokens"]),
                    temperature=float(payload["temperature"]),
                    top_p=float(payload["top_p"]),
                    reasoning_effort=payload.get("reasoning_effort"),
                    stream=False,
                )
                return response.model_dump()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.config.retry_times:
                    await asyncio.sleep(
                        min(
                            self.config.retry_backoff_seconds * (2 ** (attempt - 1)),
                            30.0,
                        )
                    )
        assert last_error is not None
        raise last_error

    async def parallel_map(
        self,
        items: Sequence[Any],
        worker: Callable[[Any, int], Awaitable[Any]],
        *,
        description: str,
        max_concurrency: int | None = None,
        progress_log_interval: int | None = None,
    ) -> list[Any]:
        total = len(items)
        if total == 0:
            return []

        concurrency = max(1, int(max_concurrency or self.config.max_concurrency))
        interval = max(
            1,
            int(progress_log_interval or self.config.progress_log_interval),
        )
        print(
            f"[SVR] {description}: submitting {total} tasks with max_concurrency={concurrency}",
            flush=True,
        )
        semaphore = asyncio.Semaphore(concurrency)
        results: list[Any] = [None] * total

        async def _run_one(idx: int, item: Any) -> tuple[int, Any]:
            async with semaphore:
                return idx, await worker(item, idx)

        items_iter = iter(enumerate(items))
        pending: set[asyncio.Task] = set()

        def _submit_next() -> bool:
            try:
                idx, item = next(items_iter)
            except StopIteration:
                return False
            pending.add(asyncio.create_task(_run_one(idx, item)))
            return True

        for _ in range(min(concurrency, total)):
            _submit_next()

        completed = 0
        while pending:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                idx, result = await task
                results[idx] = result
                completed += 1
                if (
                    completed == 1
                    or completed == total
                    or completed % interval == 0
                ):
                    print(
                        f"[SVR] {description} progress: {completed}/{total}",
                        flush=True,
                    )
                _submit_next()
        return results

    def _write_cache(
        self,
        cache_path: str | None,
        *,
        text: str,
        response: dict[str, Any],
    ) -> None:
        if not cache_path:
            return
        dump_json(
            cache_path,
            {
                "text": text,
                "response": response,
            },
        )

    def _record_cache_event(self, *, namespace: str, event: str) -> None:
        if not self.cache_dir:
            return
        if event not in {"hit", "miss"}:
            raise ValueError(f"unsupported cache event: {event}")
        stats = self._cache_stats.setdefault(namespace, {"hit": 0, "miss": 0})
        stats[event] += 1
        event_count = stats[event]
        interval = max(1, int(self.config.progress_log_interval))
        if event_count == 1 or event_count % interval == 0:
            total = stats["hit"] + stats["miss"]
            hit_rate = stats["hit"] / total if total else 0.0
            print(
                "[SVR] llm_cache "
                f"namespace={namespace} hit={stats['hit']} miss={stats['miss']} "
                f"hit_rate={hit_rate:.3f}",
                flush=True,
            )

    def log_cache_summary(self) -> None:
        if not self.cache_dir or not self._cache_stats:
            return
        total_hit = 0
        total_miss = 0
        print("[SVR] llm_cache summary start", flush=True)
        for namespace in sorted(self._cache_stats):
            stats = self._cache_stats[namespace]
            total = stats["hit"] + stats["miss"]
            hit_rate = stats["hit"] / total if total else 0.0
            total_hit += stats["hit"]
            total_miss += stats["miss"]
            print(
                "[SVR] llm_cache summary "
                f"namespace={namespace} hit={stats['hit']} miss={stats['miss']} "
                f"hit_rate={hit_rate:.3f}",
                flush=True,
            )
        total = total_hit + total_miss
        total_hit_rate = total_hit / total if total else 0.0
        print(
            "[SVR] llm_cache summary total "
            f"hit={total_hit} miss={total_miss} hit_rate={total_hit_rate:.3f}",
            flush=True,
        )

    def record_error(
        self,
        *,
        namespace: str,
        error_kind: str,
        message: str,
        prompt: str | None = None,
        raw_text: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self.error_dir:
            return
        ensure_dir(os.path.join(self.error_dir, namespace))
        error_payload = {
            "error_kind": error_kind,
            "message": message,
            "prompt_preview": (prompt[:2000] if isinstance(prompt, str) else None),
            "raw_text_preview": (raw_text[:4000] if isinstance(raw_text, str) else None),
            "extra": extra or {},
            "timestamp": time.time(),
        }
        key_source = json.dumps(
            {
                "namespace": namespace,
                "error_kind": error_kind,
                "message": message,
                "prompt": error_payload["prompt_preview"],
                "raw_text": error_payload["raw_text_preview"],
                "extra": error_payload["extra"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        file_name = f"{error_kind}_{stable_hash(key_source)}.json"
        dump_json(os.path.join(self.error_dir, namespace, file_name), error_payload)

    def _cache_path(
        self,
        *,
        namespace: str,
        payload: dict[str, Any],
        normalize_model: bool = True,
    ) -> str | None:
        if not self.cache_dir:
            return None
        normalized_payload = dict(payload)
        model_name = normalized_payload.get("model")
        if normalize_model and isinstance(model_name, str):
            normalized_payload["model"] = canonicalize_cache_model_name(model_name)
        key = normalize_text_signature(
            json.dumps(normalized_payload, ensure_ascii=False, sort_keys=True)
        )
        file_name = f"{namespace}_{stable_hash(key)}.json"
        return os.path.join(self.cache_dir, namespace, file_name)

    def _cache_paths(
        self,
        *,
        namespace: str,
        payload: dict[str, Any],
    ) -> tuple[str | None, list[str]]:
        canonical_path = self._cache_path(
            namespace=namespace,
            payload=payload,
            normalize_model=True,
        )
        if not self.cache_dir:
            return canonical_path, []

        compatible_paths: list[str] = []
        seen_paths: set[str] = set()
        model_name = payload.get("model")
        if isinstance(model_name, str):
            for compatible_model in cache_compatible_model_names(model_name):
                compatible_payload = dict(payload)
                compatible_payload["model"] = compatible_model
                cache_path = self._cache_path(
                    namespace=namespace,
                    payload=compatible_payload,
                    normalize_model=False,
                )
                if cache_path is None or cache_path in seen_paths:
                    continue
                seen_paths.add(cache_path)
                compatible_paths.append(cache_path)
        elif canonical_path is not None:
            compatible_paths.append(canonical_path)
        return canonical_path, compatible_paths


@dataclass
class RealScorerConfig:
    runner: OpenAICompatibleCompletionRunner
    chunk_size: int = 8
    judge_max_tokens: int = 8192
    tie_margin: float = 0.05


class LLMPairwiseRubricScorer:
    def __init__(self, config: RealScorerConfig):
        self.config = config

    def score_pairwise_features(
        self,
        *,
        prompt_messages: list[dict[str, str]] | None,
        prompt_text: str | None,
        chosen_response: str,
        rejected_response: str,
        bank: Sequence[BankEntry],
        bank_ids: Sequence[int] | None = None,
        difficulty_analysis: dict[str, Any] | None = None,
        rubric_note: str = "",
    ) -> np.ndarray:
        full_scores = np.zeros(len(bank), dtype=np.float32)
        candidate_bank_ids = (
            list(range(len(bank))) if bank_ids is None else sorted(set(int(i) for i in bank_ids))
        )
        if not candidate_bank_ids:
            return full_scores

        prompt_messages = _as_prompt_messages(prompt_messages, prompt_text)
        for chunk_start in range(0, len(candidate_bank_ids), self.config.chunk_size):
            chunk_ids = candidate_bank_ids[chunk_start : chunk_start + self.config.chunk_size]
            chunk_rubrics = [_to_score_rubric_dict(bank[idx]) for idx in chunk_ids]
            parsed = self._judge_chunk(
                prompt_messages=prompt_messages,
                rubrics=chunk_rubrics,
                response_a=chosen_response,
                response_b=rejected_response,
                difficulty_analysis=difficulty_analysis,
                rubric_note=rubric_note,
                namespace="pairwise_features",
            )
            if "error" in parsed:
                continue
            comparisons = parsed.get("rubric_comparisons", [])
            for local_idx, comparison in enumerate(comparisons):
                if local_idx >= len(chunk_ids):
                    break
                full_scores[chunk_ids[local_idx]] = self._comparison_to_delta(comparison)
        return full_scores

    async def ascore_pairwise_features(
        self,
        *,
        prompt_messages: list[dict[str, str]] | None,
        prompt_text: str | None,
        chosen_response: str,
        rejected_response: str,
        bank: Sequence[BankEntry],
        bank_ids: Sequence[int] | None = None,
        difficulty_analysis: dict[str, Any] | None = None,
        rubric_note: str = "",
    ) -> np.ndarray:
        full_scores = np.zeros(len(bank), dtype=np.float32)
        candidate_bank_ids = (
            list(range(len(bank))) if bank_ids is None else sorted(set(int(i) for i in bank_ids))
        )
        if not candidate_bank_ids:
            return full_scores

        prompt_messages = _as_prompt_messages(prompt_messages, prompt_text)
        for chunk_start in range(0, len(candidate_bank_ids), self.config.chunk_size):
            chunk_ids = candidate_bank_ids[chunk_start : chunk_start + self.config.chunk_size]
            chunk_rubrics = [_to_score_rubric_dict(bank[idx]) for idx in chunk_ids]
            parsed = await self._ajudge_chunk(
                prompt_messages=prompt_messages,
                rubrics=chunk_rubrics,
                response_a=chosen_response,
                response_b=rejected_response,
                difficulty_analysis=difficulty_analysis,
                rubric_note=rubric_note,
                namespace="pairwise_features",
            )
            if "error" in parsed:
                continue
            comparisons = parsed.get("rubric_comparisons", [])
            for local_idx, comparison in enumerate(comparisons):
                if local_idx >= len(chunk_ids):
                    break
                full_scores[chunk_ids[local_idx]] = self._comparison_to_delta(comparison)
        return full_scores

    def compare_responses(
        self,
        *,
        prompt_messages: list[dict[str, str]] | None,
        prompt_text: str | None,
        response_a: str,
        response_b: str,
        selected_rubrics: Sequence[dict[str, Any] | RubricItem | BankEntry],
        difficulty_analysis: dict[str, Any] | None = None,
        rubric_note: str = "",
    ) -> PairwisePrediction:
        prompt_messages = _as_prompt_messages(prompt_messages, prompt_text)
        selected_payloads: list[dict[str, Any]] = []
        comparisons: list[PairwiseRubricComparison] = []
        weighted_margin = 0.0
        preferred_hint_margin = 0.0

        selected = list(selected_rubrics)
        for chunk_start in range(0, len(selected), self.config.chunk_size):
            chunk = selected[chunk_start : chunk_start + self.config.chunk_size]
            chunk_rubrics = [_to_score_rubric_dict(item) for item in chunk]
            parsed = self._judge_chunk(
                prompt_messages=prompt_messages,
                rubrics=chunk_rubrics,
                response_a=response_a,
                response_b=response_b,
                difficulty_analysis=difficulty_analysis,
                rubric_note=rubric_note,
                namespace="pairwise_compare",
            )
            if "error" in parsed:
                continue
            preferred_candidate_hint = self._normalize_preferred_candidate_hint(
                parsed.get("preferred_candidate")
            )
            if preferred_candidate_hint == "A":
                preferred_hint_margin += sum(
                    float(
                        item.get(
                            "selection_weight",
                            importance_weight(str(item.get("importance", "major"))),
                        )
                    )
                    if isinstance(item, dict)
                    else importance_weight(item.importance)
                    for item in chunk
                )
            elif preferred_candidate_hint == "B":
                preferred_hint_margin -= sum(
                    float(
                        item.get(
                            "selection_weight",
                            importance_weight(str(item.get("importance", "major"))),
                        )
                    )
                    if isinstance(item, dict)
                    else importance_weight(item.importance)
                    for item in chunk
                )
            for local_idx, comparison in enumerate(parsed.get("rubric_comparisons", [])):
                if local_idx >= len(chunk):
                    break
                comparison = self._normalize_comparison_no_tie(
                    comparison=comparison,
                    preferred_candidate_hint=preferred_candidate_hint,
                )
                if comparison is None:
                    self.config.runner.record_error(
                        namespace="pairwise_compare",
                        error_kind="parse_error",
                        message="missing no-tie better field for rubric comparison",
                        extra={
                            "chunk_start": chunk_start,
                            "local_idx": local_idx,
                            "preferred_candidate_hint": preferred_candidate_hint,
                        },
                    )
                    continue
                source_item = chunk[local_idx]
                if isinstance(source_item, (BankEntry, RubricItem)):
                    bank_id = source_item.bank_id if isinstance(source_item, BankEntry) else int(source_item.metadata.get("bank_id", local_idx))
                    selection_weight = importance_weight(source_item.importance)
                    rubric_text = source_item.text
                    facet = source_item.facet
                    importance = source_item.importance
                    source = source_item.source
                    grounding = source_item.grounding
                else:
                    bank_id = int(source_item.get("bank_id", local_idx))
                    selection_weight = float(
                        source_item.get(
                            "selection_weight",
                            importance_weight(str(source_item.get("importance", "major"))),
                        )
                    )
                    rubric_text = str(source_item.get("text") or source_item.get("rubric") or "")
                    facet = str(source_item.get("facet", "correctness"))
                    importance = str(source_item.get("importance", "major"))
                    source = str(source_item.get("source", "svr_bank"))
                    grounding = str(source_item.get("grounding", ""))

                delta = self._comparison_to_delta(comparison)
                weighted_margin += selection_weight * delta
                selected_payloads.append(
                    {
                        "bank_id": bank_id,
                        "text": rubric_text,
                        "facet": facet,
                        "importance": importance,
                        "source": source,
                        "grounding": grounding,
                        "selection_weight": selection_weight,
                    }
                )
                comparisons.append(
                    PairwiseRubricComparison(
                        rubric_id=bank_id,
                        rubric_text=rubric_text,
                        facet=facet,
                        importance=importance,
                        weight=selection_weight,
                        score_a=max(delta, 0.0),
                        score_b=max(-delta, 0.0),
                        delta=delta,
                        better=str(comparison.get("better", "tie")),
                    )
                )

        if weighted_margin > self.config.tie_margin:
            preferred_side = "A"
            decision_source = "rubric_margin"
        elif weighted_margin < -self.config.tie_margin:
            preferred_side = "B"
            decision_source = "rubric_margin"
        elif preferred_hint_margin > 0:
            preferred_side = "A"
            decision_source = "preferred_candidate_hint"
        elif preferred_hint_margin < 0:
            preferred_side = "B"
            decision_source = "preferred_candidate_hint"
        else:
            preferred_side, decision_source = self._direct_compare_no_tie(
                prompt_messages=prompt_messages,
                response_a=response_a,
                response_b=response_b,
                difficulty_analysis=difficulty_analysis,
            )
        return PairwisePrediction(
            preferred_side=preferred_side,
            weighted_margin=weighted_margin,
            selected_rubrics=selected_payloads,
            rubric_comparisons=comparisons,
            decision_source=decision_source,
        )

    async def acompare_responses(
        self,
        *,
        prompt_messages: list[dict[str, str]] | None,
        prompt_text: str | None,
        response_a: str,
        response_b: str,
        selected_rubrics: Sequence[dict[str, Any] | RubricItem | BankEntry],
        difficulty_analysis: dict[str, Any] | None = None,
        rubric_note: str = "",
    ) -> PairwisePrediction:
        prompt_messages = _as_prompt_messages(prompt_messages, prompt_text)
        selected_payloads: list[dict[str, Any]] = []
        comparisons: list[PairwiseRubricComparison] = []
        weighted_margin = 0.0
        preferred_hint_margin = 0.0

        selected = list(selected_rubrics)
        for chunk_start in range(0, len(selected), self.config.chunk_size):
            chunk = selected[chunk_start : chunk_start + self.config.chunk_size]
            chunk_rubrics = [_to_score_rubric_dict(item) for item in chunk]
            parsed = await self._ajudge_chunk(
                prompt_messages=prompt_messages,
                rubrics=chunk_rubrics,
                response_a=response_a,
                response_b=response_b,
                difficulty_analysis=difficulty_analysis,
                rubric_note=rubric_note,
                namespace="pairwise_compare",
            )
            if "error" in parsed:
                continue
            preferred_candidate_hint = self._normalize_preferred_candidate_hint(
                parsed.get("preferred_candidate")
            )
            if preferred_candidate_hint == "A":
                preferred_hint_margin += sum(
                    float(
                        item.get(
                            "selection_weight",
                            importance_weight(str(item.get("importance", "major"))),
                        )
                    )
                    if isinstance(item, dict)
                    else importance_weight(item.importance)
                    for item in chunk
                )
            elif preferred_candidate_hint == "B":
                preferred_hint_margin -= sum(
                    float(
                        item.get(
                            "selection_weight",
                            importance_weight(str(item.get("importance", "major"))),
                        )
                    )
                    if isinstance(item, dict)
                    else importance_weight(item.importance)
                    for item in chunk
                )
            for local_idx, comparison in enumerate(parsed.get("rubric_comparisons", [])):
                if local_idx >= len(chunk):
                    break
                comparison = self._normalize_comparison_no_tie(
                    comparison=comparison,
                    preferred_candidate_hint=preferred_candidate_hint,
                )
                if comparison is None:
                    self.config.runner.record_error(
                        namespace="pairwise_compare",
                        error_kind="parse_error",
                        message="missing no-tie better field for rubric comparison",
                        extra={
                            "chunk_start": chunk_start,
                            "local_idx": local_idx,
                            "preferred_candidate_hint": preferred_candidate_hint,
                        },
                    )
                    continue
                source_item = chunk[local_idx]
                if isinstance(source_item, (BankEntry, RubricItem)):
                    bank_id = source_item.bank_id if isinstance(source_item, BankEntry) else int(source_item.metadata.get("bank_id", local_idx))
                    selection_weight = importance_weight(source_item.importance)
                    rubric_text = source_item.text
                    facet = source_item.facet
                    importance = source_item.importance
                    source = source_item.source
                    grounding = source_item.grounding
                else:
                    bank_id = int(source_item.get("bank_id", local_idx))
                    selection_weight = float(
                        source_item.get(
                            "selection_weight",
                            importance_weight(str(source_item.get("importance", "major"))),
                        )
                    )
                    rubric_text = str(source_item.get("text") or source_item.get("rubric") or "")
                    facet = str(source_item.get("facet", "correctness"))
                    importance = str(source_item.get("importance", "major"))
                    source = str(source_item.get("source", "svr_bank"))
                    grounding = str(source_item.get("grounding", ""))

                delta = self._comparison_to_delta(comparison)
                weighted_margin += selection_weight * delta
                selected_payloads.append(
                    {
                        "bank_id": bank_id,
                        "text": rubric_text,
                        "facet": facet,
                        "importance": importance,
                        "source": source,
                        "grounding": grounding,
                        "selection_weight": selection_weight,
                    }
                )
                comparisons.append(
                    PairwiseRubricComparison(
                        rubric_id=bank_id,
                        rubric_text=rubric_text,
                        facet=facet,
                        importance=importance,
                        weight=selection_weight,
                        score_a=max(delta, 0.0),
                        score_b=max(-delta, 0.0),
                        delta=delta,
                        better=str(comparison.get("better", "tie")),
                    )
                )

        if weighted_margin > self.config.tie_margin:
            preferred_side = "A"
            decision_source = "rubric_margin"
        elif weighted_margin < -self.config.tie_margin:
            preferred_side = "B"
            decision_source = "rubric_margin"
        elif preferred_hint_margin > 0:
            preferred_side = "A"
            decision_source = "preferred_candidate_hint"
        elif preferred_hint_margin < 0:
            preferred_side = "B"
            decision_source = "preferred_candidate_hint"
        else:
            preferred_side, decision_source = await self._adirect_compare_no_tie(
                prompt_messages=prompt_messages,
                response_a=response_a,
                response_b=response_b,
                difficulty_analysis=difficulty_analysis,
            )
        return PairwisePrediction(
            preferred_side=preferred_side,
            weighted_margin=weighted_margin,
            selected_rubrics=selected_payloads,
            rubric_comparisons=comparisons,
            decision_source=decision_source,
        )

    def _judge_chunk(
        self,
        *,
        prompt_messages: list[dict[str, str]],
        rubrics: list[dict[str, Any]],
        response_a: str,
        response_b: str,
        difficulty_analysis: dict[str, Any] | None,
        rubric_note: str,
        namespace: str,
    ) -> dict[str, Any]:
        prompt = self._append_no_tie_instruction(
            _build_pairwise_judge_prompt(
            prompt_messages=prompt_messages,
            rubrics=rubrics,
                response_a=response_a,
                response_b=response_b,
            difficulty_analysis=difficulty_analysis,
            rubric_note=rubric_note,
        )
        )
        raw_text, _ = self.config.runner.complete_text(
            prompt=prompt,
            namespace=namespace,
            max_tokens=self.config.judge_max_tokens,
            temperature=0.0,
        )
        parsed = _parse_json_response(raw_text)
        if "error" in parsed:
            self.config.runner.record_error(
                namespace=namespace,
                error_kind="parse_error",
                message=str(parsed["error"]),
                prompt=prompt,
                raw_text=raw_text,
                extra={"num_rubrics": len(rubrics)},
            )
        return parsed

    async def _ajudge_chunk(
        self,
        *,
        prompt_messages: list[dict[str, str]],
        rubrics: list[dict[str, Any]],
        response_a: str,
        response_b: str,
        difficulty_analysis: dict[str, Any] | None,
        rubric_note: str,
        namespace: str,
    ) -> dict[str, Any]:
        prompt = self._append_no_tie_instruction(
            _build_pairwise_judge_prompt(
            prompt_messages=prompt_messages,
            rubrics=rubrics,
                response_a=response_a,
                response_b=response_b,
            difficulty_analysis=difficulty_analysis,
            rubric_note=rubric_note,
        )
        )
        raw_text, _ = await self.config.runner.acomplete_text(
            prompt=prompt,
            namespace=namespace,
            max_tokens=self.config.judge_max_tokens,
            temperature=0.0,
        )
        parsed = _parse_json_response(raw_text)
        if "error" in parsed:
            self.config.runner.record_error(
                namespace=namespace,
                error_kind="parse_error",
                message=str(parsed["error"]),
                prompt=prompt,
                raw_text=raw_text,
                extra={"num_rubrics": len(rubrics)},
            )
        return parsed

    @staticmethod
    def _comparison_to_delta(comparison: dict[str, Any]) -> float:
        verdict_map = {"pass": 1.0, "fail": 0.0}
        score_a = verdict_map.get(str(comparison.get("candidate_a_verdict", "")).lower(), 0.0)
        score_b = verdict_map.get(str(comparison.get("candidate_b_verdict", "")).lower(), 0.0)
        delta = score_a - score_b
        better = str(comparison.get("better", "tie")).upper()
        if better == "A":
            delta += 0.25
        elif better == "B":
            delta -= 0.25
        return float(delta)

    @staticmethod
    def _append_no_tie_instruction(prompt: str) -> str:
        return (
            f"{prompt}\n\n"
            "SVR local eval hard rule:\n"
            "- You must never output `tie` for `preferred_candidate`.\n"
            "- You must never output `tie` for any rubric-level `better` field.\n"
            "- If the candidates are close, still choose the slightly better side.\n"
            "- Break close calls by this order: core task success, correctness, instruction following, safety, then relevance.\n"
            "- If they still look equivalent after all checks, choose `A`.\n"
        )

    @staticmethod
    def _normalize_preferred_candidate_hint(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().upper()
        if normalized in {"A", "B"}:
            return normalized
        return None

    @staticmethod
    def _build_direct_compare_prompt(
        *,
        prompt_messages: list[dict[str, str]],
        response_a: str,
        response_b: str,
        difficulty_analysis: dict[str, Any] | None,
    ) -> str:
        payload: dict[str, Any] = {
            "prompt_messages": prompt_messages,
            "response_a": response_a,
            "response_b": response_b,
        }
        if difficulty_analysis:
            payload["difficulty_analysis"] = difficulty_analysis
        return (
            "You are making a final no-tie decision between two assistant responses.\n\n"
            "Return only JSON:\n"
            "{\n"
            '  "preferred_candidate": "A",\n'
            '  "reason": "<brief reason>"\n'
            "}\n\n"
            "Rules:\n"
            "- You must output exactly one of `A` or `B`.\n"
            "- You must never output `tie`.\n"
            "- Prioritize: core task success, correctness, instruction following, safety, then relevance and clarity.\n"
            "- If one side is even slightly better on the highest-priority differentiator, choose that side.\n"
            "- Do not default to A unless it is actually better overall.\n\n"
            "<pairwise_compare_input>\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
            "</pairwise_compare_input>\n"
        )

    def _direct_compare_no_tie(
        self,
        *,
        prompt_messages: list[dict[str, str]],
        response_a: str,
        response_b: str,
        difficulty_analysis: dict[str, Any] | None,
    ) -> tuple[str, str]:
        prompt = self._build_direct_compare_prompt(
            prompt_messages=prompt_messages,
            response_a=response_a,
            response_b=response_b,
            difficulty_analysis=difficulty_analysis,
        )
        raw_text, _ = self.config.runner.complete_text(
            prompt=prompt,
            namespace="pairwise_compare_direct_judge",
            max_tokens=self.config.judge_max_tokens,
            temperature=0.0,
        )
        parsed = _parse_json_response(raw_text)
        preferred_side = self._normalize_preferred_candidate_hint(
            parsed.get("preferred_candidate")
        )
        if preferred_side in {"A", "B"}:
            return preferred_side, "direct_overall_judge"
        self.config.runner.record_error(
            namespace="pairwise_compare_direct_judge",
            error_kind="parse_error",
            message="missing preferred_candidate in direct judge",
            prompt=prompt,
            raw_text=raw_text,
            extra={
                "has_reason": isinstance(parsed.get("reason"), str),
            },
        )
        raise RuntimeError("direct compare judge did not return preferred_candidate")

    async def _adirect_compare_no_tie(
        self,
        *,
        prompt_messages: list[dict[str, str]],
        response_a: str,
        response_b: str,
        difficulty_analysis: dict[str, Any] | None,
    ) -> tuple[str, str]:
        prompt = self._build_direct_compare_prompt(
            prompt_messages=prompt_messages,
            response_a=response_a,
            response_b=response_b,
            difficulty_analysis=difficulty_analysis,
        )
        raw_text, _ = await self.config.runner.acomplete_text(
            prompt=prompt,
            namespace="pairwise_compare_direct_judge",
            max_tokens=self.config.judge_max_tokens,
            temperature=0.0,
        )
        parsed = _parse_json_response(raw_text)
        preferred_side = self._normalize_preferred_candidate_hint(
            parsed.get("preferred_candidate")
        )
        if preferred_side in {"A", "B"}:
            return preferred_side, "direct_overall_judge"
        self.config.runner.record_error(
            namespace="pairwise_compare_direct_judge",
            error_kind="parse_error",
            message="missing preferred_candidate in direct judge",
            prompt=prompt,
            raw_text=raw_text,
            extra={
                "has_reason": isinstance(parsed.get("reason"), str),
            },
        )
        raise RuntimeError("direct compare judge did not return preferred_candidate")

    @classmethod
    def _normalize_comparison_no_tie(
        cls,
        *,
        comparison: dict[str, Any],
        preferred_candidate_hint: str | None,
    ) -> dict[str, Any] | None:
        normalized = dict(comparison)
        better = str(normalized.get("better", "")).strip().upper()
        if better in {"A", "B"}:
            normalized["better"] = better
            return normalized

        verdict_a = str(normalized.get("candidate_a_verdict", "")).strip().lower()
        verdict_b = str(normalized.get("candidate_b_verdict", "")).strip().lower()
        if verdict_a == "pass" and verdict_b == "fail":
            normalized["better"] = "A"
        elif verdict_a == "fail" and verdict_b == "pass":
            normalized["better"] = "B"
        elif preferred_candidate_hint in {"A", "B"}:
            normalized["better"] = preferred_candidate_hint
        else:
            return None
        return normalized


@dataclass
class RealMinerConfig:
    runner: OpenAICompatibleCompletionRunner
    prompt_only_max_tokens: int = 8192
    contrastive_max_tokens: int = 4096


class LLMCandidateMiner:
    def __init__(self, config: RealMinerConfig):
        self.config = config

    def mine(self, example: PreferenceExample) -> list[RubricItem]:
        candidates: list[RubricItem] = []
        prompt_only_rubrics, difficulty_analysis = self._prompt_only_propose(example)
        if difficulty_analysis is not None:
            example.difficulty_analysis = difficulty_analysis
        candidates.extend(prompt_only_rubrics)
        candidates.extend(
            self._contrastive_probe(
                example=example,
                positive_response=example.chosen_response,
                negative_response=example.rejected_response,
                source="contrastive_probe",
            )
        )
        return self._dedupe(candidates)

    async def amine(self, example: PreferenceExample) -> list[RubricItem]:
        candidates: list[RubricItem] = []
        prompt_only_rubrics, difficulty_analysis = await self._aprompt_only_propose(example)
        if difficulty_analysis is not None:
            example.difficulty_analysis = difficulty_analysis
        candidates.extend(prompt_only_rubrics)
        candidates.extend(
            await self._acontrastive_probe(
                example=example,
                positive_response=example.chosen_response,
                negative_response=example.rejected_response,
                source="contrastive_probe",
            )
        )
        return self._dedupe(candidates)

    def mine_from_pair(
        self,
        *,
        example: PreferenceExample,
        positive_response: str,
        negative_response: str,
        source: str,
    ) -> list[RubricItem]:
        return self._dedupe(
            self._contrastive_probe(
                example=example,
                positive_response=positive_response,
                negative_response=negative_response,
                source=source,
            )
        )

    async def amine_from_pair(
        self,
        *,
        example: PreferenceExample,
        positive_response: str,
        negative_response: str,
        source: str,
    ) -> list[RubricItem]:
        return self._dedupe(
            await self._acontrastive_probe(
                example=example,
                positive_response=positive_response,
                negative_response=negative_response,
                source=source,
            )
        )

    def _validated_rubric_item(
        self,
        *,
        example: PreferenceExample,
        namespace: str,
        rubric_text: str,
        facet: str,
        importance: str,
        source: str,
        grounding: str = "",
        anchor_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RubricItem | None:
        normalized_text = normalize_candidate_rubric_text(rubric_text)
        quality_issue = rubric_text_quality_issue(normalized_text)
        if quality_issue is not None:
            self.config.runner.record_error(
                namespace=namespace,
                error_kind="invalid_rubric_text",
                message=f"{quality_issue}: {normalized_text or rubric_text}",
                raw_text=rubric_text,
                extra={
                    "example_id": example.example_id,
                    "source": source,
                    "facet": facet,
                    "importance": importance,
                },
            )
            return None
        return RubricItem(
            text=normalized_text,
            facet=facet if facet in ALLOWED_FACETS else "correctness",
            importance=importance if importance in ALLOWED_IMPORTANCE else "major",
            source=source,
            grounding=normalize_whitespace(grounding),
            anchor_id=anchor_id,
            metadata=dict(metadata or {}),
        )

    def _prompt_only_propose(
        self, example: PreferenceExample
    ) -> tuple[list[RubricItem], dict[str, Any] | None]:
        try:
            analysis_prompt = _build_prompt_analysis_prompt(example.prompt_messages)
            analysis_raw, _ = self.config.runner.complete_text(
                prompt=analysis_prompt,
                namespace="prompt_only_analysis",
                max_tokens=self.config.prompt_only_max_tokens,
                temperature=0.0,
            )
            difficulty_analysis = _parse_json_response(analysis_raw)
            if "error" in difficulty_analysis:
                self.config.runner.record_error(
                    namespace="prompt_only_analysis",
                    error_kind="parse_error",
                    message=str(difficulty_analysis["error"]),
                    prompt=analysis_prompt,
                    raw_text=analysis_raw,
                    extra={"example_id": example.example_id},
                )
                return [], None

            rubric_prompt = _build_prompt_rubric_prompt(
                prompt_messages=example.prompt_messages,
                difficulty_analysis=difficulty_analysis,
            )
            rubric_raw, _ = self.config.runner.complete_text(
                prompt=rubric_prompt,
                namespace="prompt_only_rubrics",
                max_tokens=self.config.prompt_only_max_tokens,
                temperature=0.0,
            )
            rubric_result = _parse_json_response(rubric_raw)
            if "error" in rubric_result:
                self.config.runner.record_error(
                    namespace="prompt_only_rubrics",
                    error_kind="parse_error",
                    message=str(rubric_result["error"]),
                    prompt=rubric_prompt,
                    raw_text=rubric_raw,
                    extra={"example_id": example.example_id},
                )
                return [], difficulty_analysis
            rubrics: list[RubricItem] = []
            for item in rubric_result.get("prompt_wise_rubrics", []):
                if not isinstance(item, dict):
                    continue
                rubric = self._validated_rubric_item(
                    example=example,
                    namespace="prompt_only_rubrics",
                    rubric_text=str(item.get("rubric", "")),
                    facet=str(item.get("facet", "correctness")).strip().lower(),
                    importance=str(item.get("importance", "major")).strip().lower(),
                    source=str(item.get("source", "prompt_only_propose")),
                    grounding=str(item.get("grounding", "")),
                    anchor_id=item.get("anchor_id")
                    if isinstance(item.get("anchor_id"), str)
                    else None,
                    metadata={"point_id": item.get("point_id")},
                )
                if rubric is not None:
                    rubrics.append(rubric)
            return rubrics, difficulty_analysis
        except Exception:
            return [], None

    async def _aprompt_only_propose(
        self, example: PreferenceExample
    ) -> tuple[list[RubricItem], dict[str, Any] | None]:
        try:
            analysis_prompt = _build_prompt_analysis_prompt(example.prompt_messages)
            analysis_raw, _ = await self.config.runner.acomplete_text(
                prompt=analysis_prompt,
                namespace="prompt_only_analysis",
                max_tokens=self.config.prompt_only_max_tokens,
                temperature=0.0,
            )
            difficulty_analysis = _parse_json_response(analysis_raw)
            if "error" in difficulty_analysis:
                self.config.runner.record_error(
                    namespace="prompt_only_analysis",
                    error_kind="parse_error",
                    message=str(difficulty_analysis["error"]),
                    prompt=analysis_prompt,
                    raw_text=analysis_raw,
                    extra={"example_id": example.example_id},
                )
                return [], None

            rubric_prompt = _build_prompt_rubric_prompt(
                prompt_messages=example.prompt_messages,
                difficulty_analysis=difficulty_analysis,
            )
            rubric_raw, _ = await self.config.runner.acomplete_text(
                prompt=rubric_prompt,
                namespace="prompt_only_rubrics",
                max_tokens=self.config.prompt_only_max_tokens,
                temperature=0.0,
            )
            rubric_result = _parse_json_response(rubric_raw)
            if "error" in rubric_result:
                self.config.runner.record_error(
                    namespace="prompt_only_rubrics",
                    error_kind="parse_error",
                    message=str(rubric_result["error"]),
                    prompt=rubric_prompt,
                    raw_text=rubric_raw,
                    extra={"example_id": example.example_id},
                )
                return [], difficulty_analysis
            rubrics: list[RubricItem] = []
            for item in rubric_result.get("prompt_wise_rubrics", []):
                if not isinstance(item, dict):
                    continue
                rubric = self._validated_rubric_item(
                    example=example,
                    namespace="prompt_only_rubrics",
                    rubric_text=str(item.get("rubric", "")),
                    facet=str(item.get("facet", "correctness")).strip().lower(),
                    importance=str(item.get("importance", "major")).strip().lower(),
                    source=str(item.get("source", "prompt_only_propose")),
                    grounding=str(item.get("grounding", "")),
                    anchor_id=item.get("anchor_id")
                    if isinstance(item.get("anchor_id"), str)
                    else None,
                    metadata={"point_id": item.get("point_id")},
                )
                if rubric is not None:
                    rubrics.append(rubric)
            return rubrics, difficulty_analysis
        except Exception:
            return [], None

    def _contrastive_probe(
        self,
        *,
        example: PreferenceExample,
        positive_response: str,
        negative_response: str,
        source: str,
    ) -> list[RubricItem]:
        feedback_text = self._build_feedback_excerpt(example)
        prompt = f"""You are extracting discriminative evaluation rubrics from a human preference comparison.

Use the prompt-side conversation, the preferred response, the dispreferred response, and the optional human preference notes.

Your goal is to identify the smallest reusable rubric criteria that explain why the preferred response is better.

Rules:
1. Produce 2 to 6 rubrics.
2. Each rubric must be atomic and independently checkable.
3. Rubrics must be reusable evaluation criteria, not direct answer keys.
4. Focus on difference-making factors, not generic quality praise.
5. Prefer criteria about correctness, completeness, instruction following, continuity, safety, format, and reasoning when those actually explain the preference.
6. Do not mention "preferred" or "dispreferred" inside the rubric text.
7. Each rubric must be self-contained and readable in isolation. Do not output bare fragments like "The", "All", or unfinished question stems.

Return only JSON:
{{
  "contrastive_rubrics": [
    {{
      "facet": "correctness|format|coverage|grounding|tool_use|coherence|safety|style|language|conciseness|reasoning",
      "importance": "critical|major|minor",
      "rubric": "<one reusable rubric>",
      "grounding": "<why this rubric explains the preference gap>"
    }}
  ]
}}

<prompt_conversation>
{example.prompt_text}
</prompt_conversation>

<preferred_response>
{positive_response}
</preferred_response>

<dispreferred_response>
{negative_response}
</dispreferred_response>

<human_preference_notes>
{feedback_text}
</human_preference_notes>
"""
        raw_text, _ = self.config.runner.complete_text(
            prompt=prompt,
            namespace="contrastive_probe",
            max_tokens=self.config.contrastive_max_tokens,
            temperature=0.0,
        )
        parsed = _parse_json_response(raw_text)
        items = parsed.get("contrastive_rubrics")
        if not isinstance(items, list):
            self.config.runner.record_error(
                namespace="contrastive_probe",
                error_kind="parse_error",
                message="missing contrastive_rubrics list",
                prompt=prompt,
                raw_text=raw_text,
                extra={"example_id": example.example_id},
            )
            return []
        rubrics: list[RubricItem] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            rubric = self._validated_rubric_item(
                example=example,
                namespace="contrastive_probe",
                rubric_text=str(item.get("rubric", "")),
                facet=str(item.get("facet", "correctness")).strip().lower(),
                importance=str(item.get("importance", "major")).strip().lower(),
                source=source,
                grounding=str(item.get("grounding", "")),
            )
            if rubric is not None:
                rubrics.append(rubric)
        return rubrics

    async def _acontrastive_probe(
        self,
        *,
        example: PreferenceExample,
        positive_response: str,
        negative_response: str,
        source: str,
    ) -> list[RubricItem]:
        feedback_text = self._build_feedback_excerpt(example)
        prompt = f"""You are extracting discriminative evaluation rubrics from a human preference comparison.

Use the prompt-side conversation, the preferred response, the dispreferred response, and the optional human preference notes.

Your goal is to identify the smallest reusable rubric criteria that explain why the preferred response is better.

Rules:
1. Produce 2 to 6 rubrics.
2. Each rubric must be atomic and independently checkable.
3. Rubrics must be reusable evaluation criteria, not direct answer keys.
4. Focus on difference-making factors, not generic quality praise.
5. Prefer criteria about correctness, completeness, instruction following, continuity, safety, format, and reasoning when those actually explain the preference.
6. Do not mention "preferred" or "dispreferred" inside the rubric text.
7. Each rubric must be self-contained and readable in isolation. Do not output bare fragments like "The", "All", or unfinished question stems.

Return only JSON:
{{
  "contrastive_rubrics": [
    {{
      "facet": "correctness|format|coverage|grounding|tool_use|coherence|safety|style|language|conciseness|reasoning",
      "importance": "critical|major|minor",
      "rubric": "<one reusable rubric>",
      "grounding": "<why this rubric explains the preference gap>"
    }}
  ]
}}

<prompt_conversation>
{example.prompt_text}
</prompt_conversation>

<preferred_response>
{positive_response}
</preferred_response>

<dispreferred_response>
{negative_response}
</dispreferred_response>

<human_preference_notes>
{feedback_text}
</human_preference_notes>
"""
        raw_text, _ = await self.config.runner.acomplete_text(
            prompt=prompt,
            namespace="contrastive_probe",
            max_tokens=self.config.contrastive_max_tokens,
            temperature=0.0,
        )
        parsed = _parse_json_response(raw_text)
        items = parsed.get("contrastive_rubrics")
        if not isinstance(items, list):
            self.config.runner.record_error(
                namespace="contrastive_probe",
                error_kind="parse_error",
                message="missing contrastive_rubrics list",
                prompt=prompt,
                raw_text=raw_text,
                extra={"example_id": example.example_id},
            )
            return []
        rubrics: list[RubricItem] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            rubric = self._validated_rubric_item(
                example=example,
                namespace="contrastive_probe",
                rubric_text=str(item.get("rubric", "")),
                facet=str(item.get("facet", "correctness")).strip().lower(),
                importance=str(item.get("importance", "major")).strip().lower(),
                source=source,
                grounding=str(item.get("grounding", "")),
            )
            if rubric is not None:
                rubrics.append(rubric)
        return rubrics

    @staticmethod
    def _build_feedback_excerpt(example: PreferenceExample) -> str:
        raw_record = example.raw_record or {}
        preferences = raw_record.get("individual_preference")
        if not isinstance(preferences, list):
            return "[none]"
        parts = []
        for item in preferences[:3]:
            if not isinstance(item, dict):
                continue
            for key in ("reasoning", "feedback1", "feedback2"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
        return "\n\n".join(parts[:6]) if parts else "[none]"

    @staticmethod
    def _dedupe(items: list[RubricItem]) -> list[RubricItem]:
        deduped: dict[str, RubricItem] = {}
        for item in items:
            signature = normalize_text_signature(item.text)
            if signature and signature not in deduped:
                deduped[signature] = item
        return list(deduped.values())


@dataclass
class RealAdversarialConfig:
    runner: OpenAICompatibleCompletionRunner
    scorer: LLMPairwiseRubricScorer
    max_candidates: int = 6
    max_tokens: int = 4096


class LLMAdversarialProbe:
    def __init__(self, config: RealAdversarialConfig):
        self.config = config

    def mine_hard_negative(
        self,
        *,
        example: PreferenceExample,
        selected_rubrics: list[dict[str, Any]],
    ) -> HardNegativeResult | None:
        candidates = self._generate_candidates(example)
        pool = []
        if example.rejected_response.strip():
            pool.append(("rejected_response", example.rejected_response))
        for idx, response_text in enumerate(example.candidate_responses[: self.config.max_candidates]):
            pool.append((f"dataset_candidate_{idx}", response_text))
        pool.extend(candidates)

        best_result = None
        best_margin = float("-inf")
        for source, response_text in pool:
            prediction = self.config.scorer.compare_responses(
                prompt_messages=example.prompt_messages,
                prompt_text=example.prompt_text,
                response_a=response_text,
                response_b=example.chosen_response,
                selected_rubrics=selected_rubrics,
                difficulty_analysis=example.difficulty_analysis,
            )
            if prediction.weighted_margin > best_margin:
                best_margin = prediction.weighted_margin
                best_result = (source, response_text, prediction)

        if best_result is None:
            return None
        source, response_text, prediction = best_result
        return HardNegativeResult(
            example_id=example.example_id,
            response_text=response_text,
            source=source,
            weighted_margin_vs_chosen=best_margin,
            candidate_count=len(pool),
            selected_rubrics=prediction.selected_rubrics,
        )

    async def amine_hard_negative(
        self,
        *,
        example: PreferenceExample,
        selected_rubrics: list[dict[str, Any]],
    ) -> HardNegativeResult | None:
        candidates = await self._agenerate_candidates(example)
        pool = []
        if example.rejected_response.strip():
            pool.append(("rejected_response", example.rejected_response))
        for idx, response_text in enumerate(example.candidate_responses[: self.config.max_candidates]):
            pool.append((f"dataset_candidate_{idx}", response_text))
        pool.extend(candidates)

        best_result = None
        best_margin = float("-inf")
        for source, response_text in pool:
            prediction = await self.config.scorer.acompare_responses(
                prompt_messages=example.prompt_messages,
                prompt_text=example.prompt_text,
                response_a=response_text,
                response_b=example.chosen_response,
                selected_rubrics=selected_rubrics,
                difficulty_analysis=example.difficulty_analysis,
            )
            if prediction.weighted_margin > best_margin:
                best_margin = prediction.weighted_margin
                best_result = (source, response_text, prediction)

        if best_result is None:
            return None
        source, response_text, prediction = best_result
        return HardNegativeResult(
            example_id=example.example_id,
            response_text=response_text,
            source=source,
            weighted_margin_vs_chosen=best_margin,
            candidate_count=len(pool),
            selected_rubrics=prediction.selected_rubrics,
        )

    def _generate_candidates(self, example: PreferenceExample) -> list[tuple[str, str]]:
        prompt = f"""You are generating adversarial hard-negative responses for preference learning.

Given the conversation context and a strong preferred response, generate 3 to 5 plausible but worse responses.

Rules:
1. Each candidate should look superficially reasonable.
2. Each candidate must be materially worse than the preferred response.
3. Vary the failure modes: incomplete answer, shallow reasoning, format drift, continuity error, overly generic answer, etc.
4. Keep candidates in the same language as the prompt when possible.
5. Return only JSON.

JSON format:
{{
  "candidates": [
    {{
      "response": "<candidate response>",
      "failure_mode": "<short label>"
    }}
  ]
}}

<prompt_conversation>
{example.prompt_text}
</prompt_conversation>

<preferred_response>
{example.chosen_response}
</preferred_response>
"""
        raw_text, _ = self.config.runner.complete_text(
            prompt=prompt,
            namespace="hard_negative_generate",
            max_tokens=self.config.max_tokens,
            temperature=0.0,
        )
        parsed = _parse_json_response(raw_text)
        items = parsed.get("candidates")
        if not isinstance(items, list):
            self.config.runner.record_error(
                namespace="hard_negative_generate",
                error_kind="parse_error",
                message="missing candidates list",
                prompt=prompt,
                raw_text=raw_text,
                extra={"example_id": example.example_id},
            )
            return []
        candidates: list[tuple[str, str]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            response_text = str(item.get("response", "")).strip()
            failure_mode = str(item.get("failure_mode", f"generated_{idx}")).strip()
            if not response_text:
                continue
            candidates.append((failure_mode, response_text))
        return candidates[: self.config.max_candidates]

    async def _agenerate_candidates(self, example: PreferenceExample) -> list[tuple[str, str]]:
        prompt = f"""You are generating adversarial hard-negative responses for preference learning.

Given the conversation context and a strong preferred response, generate 3 to 5 plausible but worse responses.

Rules:
1. Each candidate should look superficially reasonable.
2. Each candidate must be materially worse than the preferred response.
3. Vary the failure modes: incomplete answer, shallow reasoning, format drift, continuity error, overly generic answer, etc.
4. Keep candidates in the same language as the prompt when possible.
5. Return only JSON.

JSON format:
{{
  "candidates": [
    {{
      "response": "<candidate response>",
      "failure_mode": "<short label>"
    }}
  ]
}}

<prompt_conversation>
{example.prompt_text}
</prompt_conversation>

<preferred_response>
{example.chosen_response}
</preferred_response>
"""
        raw_text, _ = await self.config.runner.acomplete_text(
            prompt=prompt,
            namespace="hard_negative_generate",
            max_tokens=self.config.max_tokens,
            temperature=0.0,
        )
        parsed = _parse_json_response(raw_text)
        items = parsed.get("candidates")
        if not isinstance(items, list):
            self.config.runner.record_error(
                namespace="hard_negative_generate",
                error_kind="parse_error",
                message="missing candidates list",
                prompt=prompt,
                raw_text=raw_text,
                extra={"example_id": example.example_id},
            )
            return []
        candidates: list[tuple[str, str]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            response_text = str(item.get("response", "")).strip()
            failure_mode = str(item.get("failure_mode", f"generated_{idx}")).strip()
            if not response_text:
                continue
            candidates.append((failure_mode, response_text))
        return candidates[: self.config.max_candidates]


@dataclass
class RealRewriteConfig:
    runner: OpenAICompatibleCompletionRunner
    max_tokens: int = 4096


class LLMRubricRewriter:
    def __init__(self, config: RealRewriteConfig):
        self.config = config

    def rewrite(
        self,
        *,
        prompt_text: str,
        selected_rubrics: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not selected_rubrics:
            return []
        prompt = f"""You are rewriting evaluation rubrics to sound natural and prompt-specific.

You must preserve the original evaluation meaning of each rubric.
Do not invent new criteria. Do not delete any criterion.

Return only JSON:
{{
  "rewritten_rubrics": [
    {{
      "bank_id": 1,
      "text": "<rewritten rubric text>",
      "grounding": "<optional prompt-specific grounding>"
    }}
  ]
}}

<prompt_conversation>
{prompt_text}
</prompt_conversation>

<selected_bank_rubrics>
{json.dumps(selected_rubrics, ensure_ascii=False, indent=2)}
</selected_bank_rubrics>
"""
        raw_text, _ = self.config.runner.complete_text(
            prompt=prompt,
            namespace="rewrite_rubrics",
            max_tokens=self.config.max_tokens,
            temperature=0.0,
        )
        parsed = _parse_json_response(raw_text)
        items = parsed.get("rewritten_rubrics")
        if not isinstance(items, list):
            self.config.runner.record_error(
                namespace="rewrite_rubrics",
                error_kind="parse_error",
                message="missing rewritten_rubrics list",
                prompt=prompt,
                raw_text=raw_text,
                extra={"selected_count": len(selected_rubrics)},
            )
            return [dict(item) for item in selected_rubrics]
        rewritten_by_id = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            bank_id = item.get("bank_id")
            if not isinstance(bank_id, int):
                continue
            rewritten_by_id[bank_id] = item
        rewritten = []
        for original in selected_rubrics:
            payload = dict(original)
            bank_id = int(payload.get("bank_id", -1))
            replacement = rewritten_by_id.get(bank_id)
            if replacement is not None:
                new_text = str(replacement.get("text", "")).strip()
                if new_text:
                    payload["bank_text"] = payload.get("text", "")
                    payload["text"] = new_text
                grounding = str(replacement.get("grounding", "")).strip()
                if grounding:
                    payload["grounding"] = grounding
            rewritten.append(payload)
        return rewritten

    async def arewrite(
        self,
        *,
        prompt_text: str,
        selected_rubrics: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not selected_rubrics:
            return []
        prompt = f"""You are rewriting evaluation rubrics to sound natural and prompt-specific.

You must preserve the original evaluation meaning of each rubric.
Do not invent new criteria. Do not delete any criterion.

Return only JSON:
{{
  "rewritten_rubrics": [
    {{
      "bank_id": 1,
      "text": "<rewritten rubric text>",
      "grounding": "<optional prompt-specific grounding>"
    }}
  ]
}}

<prompt_conversation>
{prompt_text}
</prompt_conversation>

<selected_bank_rubrics>
{json.dumps(selected_rubrics, ensure_ascii=False, indent=2)}
</selected_bank_rubrics>
"""
        raw_text, _ = await self.config.runner.acomplete_text(
            prompt=prompt,
            namespace="rewrite_rubrics",
            max_tokens=self.config.max_tokens,
            temperature=0.0,
        )
        parsed = _parse_json_response(raw_text)
        items = parsed.get("rewritten_rubrics")
        if not isinstance(items, list):
            self.config.runner.record_error(
                namespace="rewrite_rubrics",
                error_kind="parse_error",
                message="missing rewritten_rubrics list",
                prompt=prompt,
                raw_text=raw_text,
                extra={"selected_count": len(selected_rubrics)},
            )
            return [dict(item) for item in selected_rubrics]
        rewritten_by_id = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            bank_id = item.get("bank_id")
            if not isinstance(bank_id, int):
                continue
            rewritten_by_id[bank_id] = item
        rewritten = []
        for original in selected_rubrics:
            payload = dict(original)
            bank_id = int(payload.get("bank_id", -1))
            replacement = rewritten_by_id.get(bank_id)
            if replacement is not None:
                new_text = str(replacement.get("text", "")).strip()
                if new_text:
                    payload["bank_text"] = payload.get("text", "")
                    payload["text"] = new_text
                grounding = str(replacement.get("grounding", "")).strip()
                if grounding:
                    payload["grounding"] = grounding
            rewritten.append(payload)
        return rewritten
