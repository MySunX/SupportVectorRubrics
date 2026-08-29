from __future__ import annotations

from collections import defaultdict
from typing import Any

from .models import ExampleResult


def summarize_results(results: list[ExampleResult]) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    by_domain: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_domain[r.domain].append(r.correct)
    domain_accuracy = {
        domain: (sum(vals) / len(vals) if vals else 0.0)
        for domain, vals in sorted(by_domain.items())
    }
    return {
        "num_examples": total,
        "correct": correct,
        "accuracy": (correct / total if total else 0.0),
        "domain_accuracy": domain_accuracy,
        "prediction_counts": {
            "a": sum(1 for r in results if r.predicted_candidate == "a"),
            "b": sum(1 for r in results if r.predicted_candidate == "b"),
        },
        "rubric_count_mean": (sum(r.rubric_count for r in results) / total if total else 0.0),
        "rubric_count_median": _median([r.rubric_count for r in results]),
    }


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return (values[mid - 1] + values[mid]) / 2.0

