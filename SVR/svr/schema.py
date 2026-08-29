from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RubricItem:
    text: str
    facet: str = "correctness"
    importance: str = "major"
    source: str = "unknown"
    grounding: str = ""
    anchor_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "text": self.text,
            "facet": self.facet,
            "importance": self.importance,
            "source": self.source,
            "grounding": self.grounding,
            "anchor_id": self.anchor_id,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass
class PreferenceExample:
    example_id: str
    prompt_text: str
    prompt_messages: list[dict[str, str]]
    chosen_response: str
    rejected_response: str
    candidate_responses: list[str] = field(default_factory=list)
    self_rubrics: list[RubricItem] = field(default_factory=list)
    reference_rubrics: list[RubricItem] = field(default_factory=list)
    difficulty_analysis: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    raw_record: dict[str, Any] | None = None


@dataclass
class BankEntry:
    bank_id: int
    text: str
    facet: str
    importance: str
    source: str
    grounding: str = ""
    aliases: list[str] = field(default_factory=list)
    observed_count: int = 0
    support_count: int = 0
    activation_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_rubric_item(self) -> RubricItem:
        return RubricItem(
            text=self.text,
            facet=self.facet,
            importance=self.importance,
            source=self.source,
            grounding=self.grounding,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bank_id": self.bank_id,
            "text": self.text,
            "facet": self.facet,
            "importance": self.importance,
            "source": self.source,
            "grounding": self.grounding,
            "aliases": list(self.aliases),
            "observed_count": self.observed_count,
            "support_count": self.support_count,
            "activation_count": self.activation_count,
            "metadata": dict(self.metadata),
        }


@dataclass
class PairwiseRubricComparison:
    rubric_id: int
    rubric_text: str
    facet: str
    importance: str
    weight: float
    score_a: float
    score_b: float
    delta: float
    better: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rubric_id": self.rubric_id,
            "rubric_text": self.rubric_text,
            "facet": self.facet,
            "importance": self.importance,
            "weight": self.weight,
            "score_a": self.score_a,
            "score_b": self.score_b,
            "delta": self.delta,
            "better": self.better,
        }


@dataclass
class PairwisePrediction:
    preferred_side: str
    weighted_margin: float
    selected_rubrics: list[dict[str, Any]]
    rubric_comparisons: list[PairwiseRubricComparison]
    decision_source: str = "rubric_margin"

    def to_dict(self) -> dict[str, Any]:
        return {
            "preferred_side": self.preferred_side,
            "weighted_margin": self.weighted_margin,
            "decision_source": self.decision_source,
            "selected_rubrics": self.selected_rubrics,
            "rubric_comparisons": [
                item.to_dict() for item in self.rubric_comparisons
            ],
        }


@dataclass
class HardNegativeResult:
    example_id: str
    response_text: str
    source: str
    weighted_margin_vs_chosen: float
    candidate_count: int
    selected_rubrics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "response_text": self.response_text,
            "source": self.source,
            "weighted_margin_vs_chosen": self.weighted_margin_vs_chosen,
            "candidate_count": self.candidate_count,
            "selected_rubrics": list(self.selected_rubrics),
        }


@dataclass
class SupportPairRecord:
    example_id: str
    clean_margin: float
    is_misclassified: bool
    has_hard_negative: bool
    adv_margin: float | None = None
    selected_bank_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "clean_margin": self.clean_margin,
            "is_misclassified": self.is_misclassified,
            "has_hard_negative": self.has_hard_negative,
            "adv_margin": self.adv_margin,
            "selected_bank_ids": list(self.selected_bank_ids),
        }
