from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkExample:
    case_id: str
    domain: str
    prompt_text: str
    prompt_messages: list[dict[str, str]]
    response_a: str
    response_b: str
    gold_candidate: str
    source: str = ""
    reference_rubrics: list[str] = field(default_factory=list)
    raw_record: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampledResponse:
    index: int
    group_index: int
    group_tag: str
    model: str
    text: str
    temperature: float
    top_p: float
    reasoning_effort: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RubricCandidate:
    text: str
    weight: float = 1.0
    depth: int = 0
    source: str = "initial"
    parent: str = ""
    sample_yes_count: int = 0
    sample_vote_rate: float = 0.0
    sample_votes: list[bool] = field(default_factory=list)
    group_vote_rates: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RubricComparison:
    rubric: str
    weight: float
    response_a_yes: bool
    response_b_yes: bool
    delta: float
    response_a_score: float = 0.0
    response_b_score: float = 0.0
    source: str = "rrd"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExampleResult:
    case_id: str
    domain: str
    gold_candidate: str
    predicted_candidate: str
    weighted_margin: float
    correct: bool
    rubric_count: int
    sampled_responses: list[dict[str, Any]] = field(default_factory=list)
    accepted_rubrics: list[dict[str, Any]] = field(default_factory=list)
    rubric_comparisons: list[dict[str, Any]] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
