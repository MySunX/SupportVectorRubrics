from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_rubric_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = re.sub(r"^\s*(?:rubric|new rubric)\s*\d+\s*[:.\-]?\s*", "", text, flags=re.I)
    text = re.sub(r"^\s*[-*•]\s*", "", text)
    return normalize_whitespace(text)


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        norm = normalize_rubric_text(item).lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(normalize_rubric_text(item))
    return out


def extract_tagged_items(text: str, tag: str = "RUBRIC") -> list[str]:
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    items = [normalize_rubric_text(m.group(1)) for m in pattern.finditer(text or "")]
    if items:
        return dedupe_preserve_order(items)

    lines = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(?:\d+[\.\)]|[-*•])\s*", "", line)
        if not line or line.lower().startswith(("output ", "role:", "inputs:", "goal:", "requirements:")):
            continue
        if len(line) < 8:
            continue
        lines.append(normalize_rubric_text(line))
    return dedupe_preserve_order(lines)


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    candidates = []
    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    candidates.extend(fenced)
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{") : text.rfind("}") + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def bool_from_yes_no(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if lowered.startswith("y"):
        return True
    if lowered.startswith("n"):
        return False
    return None


def extract_yes_no_evaluation(text: str) -> bool | None:
    if not text:
        return None
    m = re.search(r"<EVALUATION>\s*(YES|NO)\s*</EVALUATION>", text, flags=re.I | re.S)
    if m:
        return m.group(1).strip().upper() == "YES"
    yes = re.search(r"\bYES\b", text, flags=re.I)
    no = re.search(r"\bNO\b", text, flags=re.I)
    if yes and not no:
        return True
    if no and not yes:
        return False
    return None


def cache_key(namespace: str, payload: dict[str, Any]) -> str:
    raw = json.dumps({"namespace": namespace, "payload": payload}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def whitened_uniform_weights(votes_matrix: list[list[bool | int | float]], ridge: float = 1e-3) -> list[float]:
    """
    Approximate the paper's whitened-uniform weights from rubric vote vectors.

    rows = sampled responses, cols = rubrics
    """
    if not votes_matrix:
        return []
    x = np.asarray(votes_matrix, dtype=float)
    if x.ndim != 2 or x.shape[1] == 0:
        return []
    if x.shape[0] == 1:
        return [1.0] * x.shape[1]
    x = x - x.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / max(1, x.shape[0] - 1)
    cov = cov + ridge * np.eye(cov.shape[0], dtype=float)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, ridge, None)
        inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        w = inv_sqrt @ np.ones(cov.shape[0], dtype=float)
    except Exception:  # noqa: BLE001
        return [1.0] * x.shape[1]
    w = np.maximum(w, 0.0)
    if not np.any(w):
        w = np.ones_like(w)
    w = w / float(w.sum())
    return [float(v * len(w)) for v in w]
