from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Any

from .utils import cache_key, ensure_dir

CACHE_SCHEMA_VERSION = 3
DEFAULT_LLM_MODEL = "gpt-oss-120b"


def _openai_client_class() -> type[Any]:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The OpenAI Python SDK is required to run LLM calls. "
            "Install project dependencies with `pip install -e .` first."
        ) from exc
    return OpenAI


def canonicalize_cache_model_name(model_name: str | None) -> str:
    normalized = str(model_name or "").strip()
    lowered = normalized.lower()
    if lowered == "gpt-oss-120b" or lowered.startswith("gpt-oss-120b-"):
        return "gpt-oss-120b"
    return normalized


def cache_compatible_model_names(model_name: str | None) -> tuple[str, ...]:
    normalized = str(model_name or "").strip()
    if not normalized:
        return ()
    canonical = canonicalize_cache_model_name(normalized)
    if canonical != "gpt-oss-120b":
        return (normalized,)

    compatible_names = [canonical, normalized]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in compatible_names:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return tuple(deduped)


def _normalize_router_addr(raw_addr: str | None) -> str:
    addr = (raw_addr or "").strip()
    if not addr:
        addr = (
            os.getenv("RRD_OPENAI_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
    if addr.startswith("http://") or addr.startswith("https://"):
        base_url = addr.rstrip("/")
    else:
        base_url = f"http://{addr}".rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def extract_completion_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    choice0 = choices[0]
    if isinstance(choice0, dict):
        text = choice0.get("text")
        if isinstance(text, str) and text:
            return text.strip()
        message = choice0.get("message", {})
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_text = item.get("text")
                    if isinstance(item_text, str) and item_text.strip():
                        parts.append(item_text.strip())
                if parts:
                    return "\n".join(parts)
    return ""


def extract_reasoning_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    choice0 = choices[0]
    if isinstance(choice0, dict):
        message = choice0.get("message", {})
        if isinstance(message, dict):
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                return reasoning.strip()
            model_extra = message.get("model_extra", {}) or {}
            for key in ("reasoning_content", "reasoning", "cot"):
                value = model_extra.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return ""


def _extract_completion_text(response: dict[str, Any]) -> str:
    return extract_completion_text(response)


class OpenAIChatRunner:
    def __init__(
        self,
        *,
        model: str = DEFAULT_LLM_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        reasoning_effort: str | None = "high",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        request_timeout_sec: int = 900,
        retry_times: int = 8,
        retry_backoff_seconds: float = 1.0,
        max_concurrency: int = 64,
        cache_dir: str | Path | None = None,
    ):
        self.model = str(model).strip() or DEFAULT_LLM_MODEL
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.request_timeout_sec = request_timeout_sec
        self.retry_times = retry_times
        self.retry_backoff_seconds = retry_backoff_seconds
        self.max_concurrency = max(1, int(max_concurrency))
        self.base_url = _normalize_router_addr(base_url)
        self.api_key = (
            api_key
            or os.getenv("RRD_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or "EMPTY"
        )
        self._concurrency_semaphore = threading.Semaphore(self.max_concurrency)
        self.cache_dir = ensure_dir(cache_dir or ".cache")
        openai_client = _openai_client_class()
        self.client = openai_client(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=float(self.request_timeout_sec),
        )
        print(
            "[RRD] OpenAI-compatible runner configured: "
            f"base_url={self.base_url} model={self.model} "
            f"reasoning_effort={self.reasoning_effort} "
            f"max_tokens={self.max_tokens} retry_times={self.retry_times} "
            f"max_concurrency={self.max_concurrency}",
            flush=True,
        )

    def _cache_path(
        self,
        *,
        namespace: str,
        payload: dict[str, Any],
        normalize_model: bool = True,
    ) -> Path | None:
        if not self.cache_dir:
            return None
        normalized_payload = dict(payload)
        model_name = normalized_payload.get("model")
        if normalize_model and isinstance(model_name, str):
            normalized_payload["model"] = canonicalize_cache_model_name(model_name)
        key = cache_key(namespace, normalized_payload)
        return Path(self.cache_dir) / namespace / f"{key}.json"

    def _cache_paths(
        self,
        *,
        namespace: str,
        payload: dict[str, Any],
    ) -> tuple[Path | None, list[Path]]:
        canonical_path = self._cache_path(namespace=namespace, payload=payload, normalize_model=True)
        if not self.cache_dir:
            return canonical_path, []

        compatible_paths: list[Path] = []
        seen_paths: set[Path] = set()
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

    def complete_text(
        self,
        *,
        prompt: str,
        namespace: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        reasoning_effort: str | None = None,
        model: str | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        payload = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "model": str(model or self.model).strip(),
            "prompt": prompt,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p if top_p is None else top_p,
            "reasoning_effort": self.reasoning_effort if reasoning_effort is None else reasoning_effort,
        }
        if extra_payload:
            payload.update(extra_payload)

        cache_path, cache_paths = self._cache_paths(namespace=namespace, payload=payload)
        for candidate_cache_path in cache_paths:
            if not candidate_cache_path.is_file():
                continue
            cached = json.loads(candidate_cache_path.read_text(encoding="utf-8"))
            if cache_path and candidate_cache_path != cache_path:
                ensure_dir(cache_path.parent)
                cache_path.write_text(
                    json.dumps(
                        {
                            "text": str(cached.get("text", "")),
                            "response": cached.get("response", {}),
                            "payload": cached.get("payload", payload),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            return str(cached.get("text", "")), cached.get("response", {})

        if cache_path is not None:
            ensure_dir(cache_path.parent)

        last_error: Exception | None = None
        for attempt in range(1, max(1, self.retry_times) + 1):
            try:
                with self._concurrency_semaphore:
                    response = self.client.chat.completions.create(
                        model=str(payload["model"]),
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=int(payload["max_tokens"]),
                        temperature=float(payload["temperature"]),
                        top_p=float(payload["top_p"]),
                        reasoning_effort=payload.get("reasoning_effort"),
                        stream=False,
                    )
                    response_json = response.model_dump()
                text = extract_completion_text(response_json).strip()
                reasoning = extract_reasoning_text(response_json)
                if cache_path is not None:
                    ensure_dir(cache_path.parent)
                    cache_path.write_text(
                        json.dumps(
                            {
                                "text": text,
                                "reasoning": reasoning,
                                "response": response_json,
                                "payload": payload,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                return text, response_json
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < max(1, self.retry_times):
                    print(
                        "[RRD] LLM call failed; retrying "
                        f"attempt={attempt}/{max(1, self.retry_times)} "
                        f"model={payload['model']} namespace={namespace} "
                        f"error={type(exc).__name__}: {str(exc)[:300]}",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(
                        min(
                            self.retry_backoff_seconds * (2 ** (attempt - 1)),
                            30.0,
                        )
                    )
        assert last_error is not None
        raise last_error

    def complete(
        self,
        *,
        namespace: str,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        reasoning_effort: str | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> str:
        text, _ = self.complete_text(
            namespace=namespace,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            extra_payload=extra_payload,
        )
        return text

    async def acomplete_text(
        self,
        *,
        prompt: str,
        namespace: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        reasoning_effort: str | None = None,
        model: str | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        return await asyncio.to_thread(
            self.complete_text,
            prompt=prompt,
            namespace=namespace,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            model=model,
            extra_payload=extra_payload,
        )

    async def acomplete(
        self,
        *,
        namespace: str,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        reasoning_effort: str | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> str:
        text, _ = await self.acomplete_text(
            namespace=namespace,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            extra_payload=extra_payload,
        )
        return text
